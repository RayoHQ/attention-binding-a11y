"""Minimal C5: Test if high-EB* heads are necessary for behavior.

Ablates top-binding heads vs. random heads on a single checkpoint
(160m step120000) and compares accuracy drops on recognition prompts.

Uses TransformerLens hook infrastructure for clean head ablation and
reuses the existing behavioral evaluation pipeline (scoring.py).
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils_model import load_pythia_with_checkpoint
from extract_attention import extract_binding_for_prompt
from scoring import score_recognition_logprob

# ── Config ──────────────────────────────────────────────────────────
MODEL_SIZE = "2.8b"
CHECKPOINT = "step143000"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_HEADS_TO_ABLATE = 4
N_RANDOM_BASELINES = 5
PROMPTS_FILE = Path("data/prompts/pilot_terms.jsonl")
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── Helpers ─────────────────────────────────────────────────────────

def load_prompts(path=PROMPTS_FILE):
    with open(path) as f:
        return [json.loads(line) for line in f]


def compute_per_head_bsi(model, prompt_text, term, tokenizer):
    """Compute BSI for every (layer, head) pair on a single prompt.

    Returns dict mapping (layer, head) -> bsi_score.
    Reuses the span-finding logic from extract_attention.py and
    computes BSI identically (mean later→earlier within-span attention).
    """
    from extract_attention import TERM_ALIASES

    tokens = model.to_tokens(prompt_text, prepend_bos=True)
    seq_len = tokens.shape[1]

    # ── Locate term span (same logic as extract_binding_for_prompt) ──
    search_terms = [term]
    search_terms.extend(TERM_ALIASES.get(term.lower(), []))

    term_variants = []
    for t in search_terms:
        for form in [t, t.capitalize(), t.title()]:
            term_variants.append(tokenizer.encode(form, add_special_tokens=False))
            term_variants.append(tokenizer.encode(" " + form, add_special_tokens=False))
    seen = set()
    unique_variants = []
    for v in term_variants:
        key = tuple(v)
        if key not in seen:
            seen.add(key)
            unique_variants.append(v)

    full_token_ids = tokens[0].tolist()
    span_start = None
    term_tokens = unique_variants[0]
    for variant in unique_variants:
        for i in range(len(full_token_ids) - len(variant) + 1):
            if full_token_ids[i : i + len(variant)] == variant:
                span_start = i
                term_tokens = variant
                break
        if span_start is not None:
            break

    n_term_tokens = len(term_tokens)

    if span_start is None:
        decoded_tokens = [tokenizer.decode([t]) for t in full_token_ids]
        joined = "".join(decoded_tokens)
        char_pos = joined.lower().find(term.lower())
        if char_pos >= 0:
            cum_len = 0
            for idx, dt in enumerate(decoded_tokens):
                if cum_len >= char_pos:
                    span_start = idx
                    n_term_tokens = len(unique_variants[0])
                    break
                cum_len += len(dt)
        if span_start is None:
            span_start = max(0, seq_len - n_term_tokens - 5)

    span_indices = list(range(span_start, span_start + n_term_tokens))

    # ── Extract BSI per (layer, head) ───────────────────────────────
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    head_scores = {}

    for layer_idx in range(n_layers):
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=[f"blocks.{layer_idx}.attn.hook_pattern"],
                stop_at_layer=layer_idx + 1,
            )

        attn = cache[f"blocks.{layer_idx}.attn.hook_pattern"]  # [1, n_heads, seq, seq]

        for head_idx in range(n_heads):
            head_attn = attn[0, head_idx]
            later_to_earlier = []
            for dest_idx in span_indices:
                for src_idx in span_indices:
                    if dest_idx > src_idx:
                        later_to_earlier.append(head_attn[dest_idx, src_idx].item())

            bsi = sum(later_to_earlier) / len(later_to_earlier) if later_to_earlier else 0.0
            head_scores[(layer_idx, head_idx)] = bsi

        del cache
        torch.cuda.empty_cache()

    return head_scores


def find_top_binding_heads(model, prompts, n_heads=N_HEADS_TO_ABLATE):
    """Identify (layer, head) pairs with highest average BSI across prompts."""
    aggregated = defaultdict(list)

    for prompt in tqdm(prompts, desc="Computing per-head BSI"):
        scores = compute_per_head_bsi(
            model, prompt["template"], prompt["term"], model.tokenizer
        )
        for key, bsi in scores.items():
            aggregated[key].append(bsi)

    avg_scores = [
        (layer, head, float(np.mean(vals)))
        for (layer, head), vals in aggregated.items()
    ]
    avg_scores.sort(key=lambda x: x[2], reverse=True)
    return avg_scores[:n_heads]


def evaluate_recognition_with_ablation(model, prompts, ablate_heads=None):
    """Evaluate recognition accuracy, optionally zeroing out specific heads.

    Uses TransformerLens fwd_hooks (temporary, context-managed) to zero
    the attention pattern for selected heads during every forward pass.
    Then calls the same log-prob scoring used by the behavioral pipeline.
    """
    hooks_list = []

    if ablate_heads:
        # Group heads by layer for efficiency
        heads_by_layer = defaultdict(list)
        for layer, head in ablate_heads:
            heads_by_layer[layer].append(head)

        for layer, head_indices in heads_by_layer.items():
            hook_name = f"blocks.{layer}.attn.hook_pattern"

            def make_hook(heads_to_zero):
                def hook_fn(activation, hook):
                    # activation: [batch, n_heads, seq, seq]
                    for h in heads_to_zero:
                        activation[:, h, :, :] = 0.0
                    return activation
                return hook_fn

            hooks_list.append((hook_name, make_hook(head_indices)))

    correct = 0
    total = 0
    per_prompt = []

    for prompt in prompts:
        if prompt["task"] != "recognition":
            continue

        choices = prompt.get("choices")
        answer_idx = prompt.get("answer_idx")
        if choices is None or answer_idx is None:
            continue

        # Run scoring with hooks active
        if hooks_list:
            with model.hooks(fwd_hooks=hooks_list):
                result = score_recognition_logprob(
                    model, prompt["template"], choices, answer_idx
                )
        else:
            result = score_recognition_logprob(
                model, prompt["template"], choices, answer_idx
            )

        total += 1
        if result["is_correct"]:
            correct += 1

        per_prompt.append({
            "term": prompt["term"],
            "prompt_id": prompt["prompt_id"],
            "is_correct": result["is_correct"],
            "predicted_idx": result["predicted_idx"],
            "log_probs": result["log_probs"],
        })

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total, per_prompt


def evaluate_generation_with_ablation(model, prompts, ablate_heads=None):
    """Evaluate generation quality, optionally zeroing out specific heads.

    Uses the same keyword rubric as the behavioral pipeline.
    """
    from scoring import score_generation

    hooks_list = []

    if ablate_heads:
        heads_by_layer = defaultdict(list)
        for layer, head in ablate_heads:
            heads_by_layer[layer].append(head)

        for layer, head_indices in heads_by_layer.items():
            hook_name = f"blocks.{layer}.attn.hook_pattern"

            def make_hook(heads_to_zero):
                def hook_fn(activation, hook):
                    for h in heads_to_zero:
                        activation[:, h, :, :] = 0.0
                    return activation
                return hook_fn

            hooks_list.append((hook_name, make_hook(head_indices)))

    scores = []
    per_prompt = []

    for prompt in prompts:
        if prompt["task"] != "generation":
            continue

        tokens = model.to_tokens(prompt["template"])
        max_tokens = prompt.get("max_tokens", 20)

        with torch.no_grad():
            if hooks_list:
                with model.hooks(fwd_hooks=hooks_list):
                    output = model.generate(
                        tokens,
                        max_new_tokens=max_tokens,
                        temperature=0.0,
                        do_sample=False,
                    )
            else:
                output = model.generate(
                    tokens,
                    max_new_tokens=max_tokens,
                    temperature=0.0,
                    do_sample=False,
                )

        text_out = model.tokenizer.decode(output[0], skip_special_tokens=True)
        template = prompt["template"]
        generated = text_out[len(template):].strip() if text_out.startswith(template) else text_out.strip()

        score = score_generation(generated, prompt["term"])
        scores.append(score)

        per_prompt.append({
            "term": prompt["term"],
            "prompt_id": prompt["prompt_id"],
            "generated": generated,
            "score": score,
        })

    mean_score = float(np.mean(scores)) if scores else 0.0
    return mean_score, per_prompt


# ── Main ────────────────────────────────────────────────────────────

def main():
    print(f"{'=' * 70}")
    print(f"  Minimal C5: Causal test — {MODEL_SIZE} @ {CHECKPOINT}")
    print(f"{'=' * 70}\n")

    # Load model
    print("Loading model …")
    model = load_pythia_with_checkpoint(MODEL_SIZE, CHECKPOINT, DEVICE)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    print(f"  {n_layers} layers × {n_heads} heads = {n_layers * n_heads} total heads\n")

    prompts = load_prompts()
    rec_prompts = [p for p in prompts if p["task"] == "recognition"]
    gen_prompts = [p for p in prompts if p["task"] == "generation"]
    print(f"  {len(rec_prompts)} recognition prompts, {len(gen_prompts)} generation prompts\n")

    # ── Step 1: Identify top-binding heads ──────────────────────────
    print("Step 1: Computing per-head BSI across all prompts …")
    top_heads_data = find_top_binding_heads(model, prompts, N_HEADS_TO_ABLATE)
    top_heads = [(h[0], h[1]) for h in top_heads_data]
    print(f"\n  Top {N_HEADS_TO_ABLATE} binding heads (layer, head, avg BSI):")
    for layer, head, bsi in top_heads_data:
        print(f"    L{layer:2d} H{head:2d}  BSI = {bsi:.4f}")

    # ── Step 2: Evaluate conditions ─────────────────────────────────
    all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]

    conditions = [
        ("BASELINE (no ablation)", None),
        (f"TOP-{N_HEADS_TO_ABLATE} BINDING HEADS ablated", top_heads),
    ]

    # Random baselines
    for i in range(N_RANDOM_BASELINES):
        random_heads = random.sample(
            [h for h in all_heads if h not in top_heads],
            N_HEADS_TO_ABLATE,
        )
        conditions.append((f"RANDOM heads ablated (trial {i + 1})", random_heads))

    # Also add bottom-binding heads for comparison
    aggregated = defaultdict(list)
    for prompt in prompts:
        scores = compute_per_head_bsi(
            model, prompt["template"], prompt["term"], model.tokenizer
        )
        for key, bsi in scores.items():
            aggregated[key].append(bsi)
    all_avg = sorted(
        [(l, h, float(np.mean(v))) for (l, h), v in aggregated.items()],
        key=lambda x: x[2],
    )
    bottom_heads = [(h[0], h[1]) for h in all_avg[:N_HEADS_TO_ABLATE]]
    conditions.append((f"BOTTOM-{N_HEADS_TO_ABLATE} BINDING HEADS ablated", bottom_heads))

    print(f"\n{'=' * 70}")
    print("Step 2: Evaluating conditions\n")

    rec_results = {}
    gen_results = {}

    for name, heads in conditions:
        print(f"\n--- {name} ---")
        if heads:
            print(f"  Heads: {heads}")

        rec_acc, rec_correct, rec_total, rec_detail = evaluate_recognition_with_ablation(
            model, rec_prompts, heads
        )
        gen_score, gen_detail = evaluate_generation_with_ablation(
            model, gen_prompts, heads
        )

        rec_results[name] = {
            "accuracy": rec_acc,
            "correct": rec_correct,
            "total": rec_total,
            "detail": rec_detail,
        }
        gen_results[name] = {
            "mean_score": gen_score,
            "detail": gen_detail,
        }

        print(f"  Recognition: {rec_correct}/{rec_total} = {rec_acc:.3f}")
        print(f"  Generation:  mean score = {gen_score:.3f}")

    # ── Step 3: Summary ─────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}\n")

    print(f"{'Condition':<45} {'RecAcc':>8}  {'GenScore':>8}")
    print("-" * 65)
    for name in [c[0] for c in conditions]:
        ra = rec_results[name]["accuracy"]
        gs = gen_results[name]["mean_score"]
        print(f"{name:<45} {ra:>8.3f}  {gs:>8.3f}")

    # Compute drops
    baseline_rec = rec_results[conditions[0][0]]["accuracy"]
    baseline_gen = gen_results[conditions[0][0]]["mean_score"]

    top_name = conditions[1][0]
    top_rec_drop = baseline_rec - rec_results[top_name]["accuracy"]
    top_gen_drop = baseline_gen - gen_results[top_name]["mean_score"]

    random_rec_drops = []
    random_gen_drops = []
    for name, heads in conditions[2:-1]:  # Skip bottom condition
        random_rec_drops.append(baseline_rec - rec_results[name]["accuracy"])
        random_gen_drops.append(baseline_gen - gen_results[name]["mean_score"])

    mean_random_rec_drop = float(np.mean(random_rec_drops)) if random_rec_drops else 0
    mean_random_gen_drop = float(np.mean(random_gen_drops)) if random_gen_drops else 0

    bottom_name = conditions[-1][0]
    bottom_rec_drop = baseline_rec - rec_results[bottom_name]["accuracy"]
    bottom_gen_drop = baseline_gen - gen_results[bottom_name]["mean_score"]

    print(f"\n{'Accuracy Drops from Baseline':}")
    print("-" * 65)
    print(f"  Top-binding ablated:     Rec Δ = {top_rec_drop:+.3f}   Gen Δ = {top_gen_drop:+.3f}")
    print(f"  Random ablated (mean):   Rec Δ = {mean_random_rec_drop:+.3f}   Gen Δ = {mean_random_gen_drop:+.3f}")
    print(f"  Bottom-binding ablated:  Rec Δ = {bottom_rec_drop:+.3f}   Gen Δ = {bottom_gen_drop:+.3f}")

    # ── Interpretation ──────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("INTERPRETATION")
    print(f"{'=' * 70}\n")

    # Use combined metric (rec + gen)
    top_combined_drop = (top_rec_drop + top_gen_drop) / 2
    random_combined_drop = (mean_random_rec_drop + mean_random_gen_drop) / 2
    specificity = top_combined_drop - random_combined_drop

    print(f"  Top combined drop:      {top_combined_drop:+.3f}")
    print(f"  Random combined drop:   {random_combined_drop:+.3f}")
    print(f"  Specificity (Δtop - Δrandom): {specificity:+.3f}\n")

    if specificity > 0.10:
        print("  ✅ C5 SUPPORTED: Top binding heads are specifically necessary")
        print("     → Ablating high-BSI heads degrades performance more than random")
        print("     → Binding is mechanistically implicated in behavior")
    elif specificity > 0.0:
        print("  ⚠️  C5 WEAKLY SUPPORTED: Some specificity, but not decisive")
        print("     → Binding correlates with importance, but effect is modest")
    else:
        print("  ❌ C5 NOT SUPPORTED: No specificity of top heads over random")
        print("     → Binding may be correlated but not uniquely causal")

    print(f"\n{'=' * 70}")

    # ── Save results ────────────────────────────────────────────────
    output_dir = Path("data/results/causal")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{MODEL_SIZE}_{CHECKPOINT}_causal.json"

    save_data = {
        "model": f"pythia-{MODEL_SIZE}-deduped",
        "checkpoint": CHECKPOINT,
        "n_heads_ablated": N_HEADS_TO_ABLATE,
        "n_random_trials": N_RANDOM_BASELINES,
        "top_heads": [
            {"layer": h[0], "head": h[1], "avg_bsi": h[2]}
            for h in top_heads_data
        ],
        "bottom_heads": [
            {"layer": h[0], "head": h[1], "avg_bsi": h[2]}
            for h in all_avg[:N_HEADS_TO_ABLATE]
        ],
        "recognition": {
            name: {
                "accuracy": rec_results[name]["accuracy"],
                "correct": rec_results[name]["correct"],
                "total": rec_results[name]["total"],
            }
            for name in [c[0] for c in conditions]
        },
        "generation": {
            name: {"mean_score": gen_results[name]["mean_score"]}
            for name in [c[0] for c in conditions]
        },
        "drops": {
            "top_rec_drop": top_rec_drop,
            "top_gen_drop": top_gen_drop,
            "mean_random_rec_drop": mean_random_rec_drop,
            "mean_random_gen_drop": mean_random_gen_drop,
            "bottom_rec_drop": bottom_rec_drop,
            "bottom_gen_drop": bottom_gen_drop,
            "specificity": specificity,
        },
    }

    with open(output_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved results to {output_file}")

    # Cleanup
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
