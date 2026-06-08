"""C5 causal head ablation for CRFM, SmolLM3, OLMo, and Qwen2.5.

Uses zero-ablation of top-4 / bottom-4 / random-4 binding heads.
CRFM uses TransformerLens hooks; SmolLM3/OLMo/Qwen use HF output_attentions + manual zeroing.

Usage:
    python src/run_c5_new_models.py --model crfm   # seed1 only, final ck
    python src/run_c5_new_models.py --model smollm3
    python src/run_c5_new_models.py --model olmo
    python src/run_c5_new_models.py --model qwen
    python src/run_c5_new_models.py --all
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from scoring import score_generation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_HEADS = 4
N_RAND = 5
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

OUTPUT_DIR = Path("data/results/causal")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_FILE = Path("data/prompts/canonical_45terms.jsonl")


# ── Shared: load 45-term recognition prompts ─────────────────────────────────
def load_rec_prompts(prompts_file=None):
    with open(prompts_file or PROMPTS_FILE) as f:
        return [json.loads(l) for l in f if json.loads(l)["task"] == "recognition"]


# ── Shared: HF-based ablation harness ────────────────────────────────────────
def _hf_score_recognition(model, tokenizer, prompt, choices, answer_idx):
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    plen = prompt_ids.shape[1]
    scores = []
    for ch in choices:
        ch_ids = tokenizer.encode(" " + ch, add_special_tokens=False)
        full = torch.cat([prompt_ids, torch.tensor([ch_ids], device=DEVICE)], dim=1)
        with torch.no_grad():
            lp = torch.log_softmax(model(full).logits.float(), dim=-1)
        scores.append(sum(lp[0, plen - 1 + i, tid].item()
                          for i, tid in enumerate(ch_ids)) / max(1, len(ch_ids)))
    pred = int(torch.tensor(scores).argmax())
    return 1.0 if pred == answer_idx else 0.0


def _hf_ablation_c5(model, tokenizer, prompts, n_layers, n_heads_per_layer,
                    top_heads, rand_head_sets, bot_heads, model_name, ck):
    """Compute baseline, top-ablated, random-ablated, bottom-ablated rec accuracy.
    Returns aggregate result dict plus per_term breakdown.
    """
    def run_with_ablation(ablate_heads):
        """Returns (agg_acc, {term: mean_acc})."""
        hooks = []
        for layer_idx, head_idx in ablate_heads:
            layer = model.model.layers[layer_idx]
            attn = layer.self_attn
            head_dim = attn.head_dim if hasattr(attn, "head_dim") else (
                model.config.hidden_size // model.config.num_attention_heads)

            def make_attn_hook(h_idx, h_dim):
                def attn_hook(module, args, kwargs, output):
                    if isinstance(output, tuple):
                        attn_out = output[0]
                    else:
                        attn_out = output
                    attn_out = attn_out.clone()
                    attn_out[:, :, h_idx * h_dim:(h_idx + 1) * h_dim] = 0.0
                    if isinstance(output, tuple):
                        return (attn_out,) + output[1:]
                    return attn_out
                return attn_hook

            h = layer.self_attn.register_forward_hook(
                make_attn_hook(head_idx, head_dim), with_kwargs=True
            )
            hooks.append(h)

        term_scores = defaultdict(list)
        for p in prompts:
            s = _hf_score_recognition(model, tokenizer, p["template"],
                                      p["choices"], p["answer_idx"])
            term_scores[p["term"]].append(s)
        for h in hooks:
            h.remove()
        term_means = {t: sum(v) / len(v) for t, v in term_scores.items()}
        agg = sum(s for v in term_scores.values() for s in v) / max(
            1, sum(len(v) for v in term_scores.values()))
        return agg, term_means

    baseline, baseline_t = run_with_ablation([])
    top_acc, top_t = run_with_ablation(top_heads)
    bot_acc, bot_t = run_with_ablation(bot_heads)
    rand_results = [run_with_ablation(rh) for rh in rand_head_sets]
    rand_acc = sum(r[0] for r in rand_results) / len(rand_results)
    rand_t = {t: sum(r[1].get(t, 0.0) for r in rand_results) / len(rand_results)
              for t in baseline_t}

    specificity = (baseline - top_acc) - (baseline - rand_acc)
    per_term = {
        t: {
            "baseline":     round(baseline_t[t], 4),
            "top_ablated":  round(top_t.get(t, 0.0), 4),
            "rand_ablated": round(rand_t.get(t, 0.0), 4),
            "bot_ablated":  round(bot_t.get(t, 0.0), 4),
            "specificity":  round(
                (baseline_t[t] - top_t.get(t, 0.0)) -
                (baseline_t[t] - rand_t.get(t, 0.0)), 4),
        }
        for t in sorted(baseline_t)
    }
    return {
        "model": model_name, "checkpoint": ck,
        "n_prompts": len(prompts),
        "baseline_rec": round(baseline, 4),
        "top_ablated_rec": round(top_acc, 4),
        "rand_ablated_rec": round(rand_acc, 4),
        "bot_ablated_rec": round(bot_acc, 4),
        "rec_drop_top": round(baseline - top_acc, 4),
        "rec_drop_rand": round(baseline - rand_acc, 4),
        "specificity": round(specificity, 4),
        "top_heads": top_heads,
        "per_term": per_term,
    }


def _get_bsi_ranking_hf(model, tokenizer, prompts, n_layers, n_heads):
    """Compute mean BSI per (layer, head) from output_attentions."""
    head_bsi = defaultdict(list)
    for p in tqdm(prompts[:30], desc="BSI ranking"):  # sample 30 for speed
        inputs = tokenizer(p["template"], return_tensors="pt").to(DEVICE)
        term = p["term"]
        ids = inputs["input_ids"][0].tolist()
        # find term span
        term_toks = tokenizer.encode(" " + term, add_special_tokens=False)
        span = None
        for i in range(len(ids) - len(term_toks) + 1):
            if ids[i:i + len(term_toks)] == term_toks:
                span = list(range(i, i + len(term_toks)))
                break
        if not span or len(span) < 2:
            continue
        with torch.no_grad():
            out = model(**inputs, output_attentions=True, return_dict=True)
        for li, la in enumerate(out.attentions):
            la = la.float()
            nh = la.shape[1]
            for hi in range(min(nh, n_heads)):
                ha = la[0, hi]
                pairs = [ha[d, s].item() for d in span for s in span if d > s]
                if pairs:
                    head_bsi[(li, hi)].append(sum(pairs) / len(pairs))

    mean_bsi = {k: sum(v) / len(v) for k, v in head_bsi.items() if v}
    ranked = sorted(mean_bsi.items(), key=lambda x: -x[1])
    return ranked


# ── CRFM C5 (TransformerLens) ─────────────────────────────────────────────────
def _crfm_rec_perterm_fn(model, prompts, ablate_heads=None):
    """Per-term recognition accuracy for CRFM (TransformerLens)."""
    from run_causal_c5 import _make_hooks
    from scoring import score_recognition_logprob
    hooks = _make_hooks(ablate_heads)
    per_term = defaultdict(list)
    for p in prompts:
        if hooks:
            with model.hooks(fwd_hooks=hooks):
                r = score_recognition_logprob(model, p["template"], p["choices"], p["answer_idx"])
        else:
            r = score_recognition_logprob(model, p["template"], p["choices"], p["answer_idx"])
        per_term[p["term"]].append(float(r["is_correct"]))
    return {t: sum(v) / len(v) for t, v in per_term.items()}


def run_crfm_c5(seed=1):
    from extract_binding_crfm import load_crfm
    from run_causal_c5 import compute_per_head_bsi, eval_recognition

    ck = "checkpoint-400000"
    out_file = OUTPUT_DIR / f"crfm_seed{seed}_{ck}_c5_canonical41.json"
    perterm_file = OUTPUT_DIR / f"crfm_seed{seed}_{ck}_c5_canonical41_perterm.json"
    if out_file.exists() and perterm_file.exists():
        print(f"⏭  {out_file.name} + perterm — skipping"); return

    device = DEVICE
    model = load_crfm(seed, ck, device)
    prompts = load_rec_prompts()

    # Compute BSI per head
    head_bsi = defaultdict(list)
    for p in tqdm(prompts[:30], desc=f"BSI crfm-x{seed}"):
        bsi = compute_per_head_bsi(model, p["template"], p["term"])
        for (li, hi), v in bsi.items():
            head_bsi[(li, hi)].append(v)
    ranked = sorted({k: sum(v)/len(v) for k, v in head_bsi.items() if v}.items(),
                    key=lambda x: -x[1])
    top_heads = [list(k) for k, _ in ranked[:N_HEADS]]
    bot_heads = [list(k) for k, _ in ranked[-N_HEADS:]]
    all_heads = list(ranked)
    rand_sets = [[list(k) for k, _ in random.sample(all_heads, N_HEADS)] for _ in range(N_RAND)]

    def rec_acc(ablate):
        acc, _, _ = eval_recognition(model, prompts, ablate_heads=ablate if ablate else None)
        return acc

    baseline = rec_acc([])
    top_acc = rec_acc(top_heads)
    bot_acc = rec_acc(bot_heads)
    rand_acc = sum(rec_acc(rh) for rh in rand_sets) / N_RAND

    if not out_file.exists():
        result = {
            "model": f"crfm-gpt2-small-x{seed}", "checkpoint": ck,
            "n_prompts": len(prompts),
            "baseline_rec": round(baseline, 4),
            "top_ablated_rec": round(top_acc, 4),
            "rand_ablated_rec": round(rand_acc, 4),
            "bot_ablated_rec": round(bot_acc, 4),
            "rec_drop_top": round(baseline - top_acc, 4),
            "specificity": round((baseline - top_acc) - (baseline - rand_acc), 4),
            "top_heads": top_heads,
        }
        json.dump(result, open(out_file, "w"), indent=2)
        print(f"✅ CRFM C5 aggregate → {out_file}")
    print(f"   Baseline={baseline:.3f}  Top-ablated={top_acc:.3f}  Spec={round((baseline-top_acc)-(baseline-rand_acc),3):+.3f}")

    # Per-term computation
    baseline_t = _crfm_rec_perterm_fn(model, prompts)
    top_t      = _crfm_rec_perterm_fn(model, prompts, ablate_heads=top_heads)
    bot_t      = _crfm_rec_perterm_fn(model, prompts, ablate_heads=bot_heads)
    rand_t_list = [_crfm_rec_perterm_fn(model, prompts, ablate_heads=rh) for rh in rand_sets]
    rand_t = {t: sum(d.get(t, 0.0) for d in rand_t_list) / len(rand_t_list)
              for t in baseline_t}
    per_term = {
        t: {
            "baseline":     round(baseline_t[t], 4),
            "top_ablated":  round(top_t.get(t, 0.0), 4),
            "rand_ablated": round(rand_t.get(t, 0.0), 4),
            "bot_ablated":  round(bot_t.get(t, 0.0), 4),
            "specificity":  round(
                (baseline_t[t] - top_t.get(t, 0.0)) -
                (baseline_t[t] - rand_t.get(t, 0.0)), 4),
        }
        for t in sorted(baseline_t)
    }
    json.dump({"model": f"crfm-gpt2-small-x{seed}", "checkpoint": ck,
               "per_term": per_term}, open(perterm_file, "w"), indent=2)
    print(f"✅ CRFM C5 per-term → {perterm_file}")
    del model; torch.cuda.empty_cache()


# ── SmolLM3 C5 ───────────────────────────────────────────────────────────────
def run_smollm3_c5():
    from utils_model_smollm3 import load_smollm3_with_checkpoint

    ck = "step3440k"
    out_file = OUTPUT_DIR / f"smollm3_{ck}_c5_canonical41.json"
    perterm_file = OUTPUT_DIR / f"smollm3_{ck}_c5_canonical41_perterm.json"
    if perterm_file.exists():
        print(f"⏭  {perterm_file.name} — skipping"); return

    model, tokenizer = load_smollm3_with_checkpoint(ck, DEVICE)
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    prompts = load_rec_prompts()

    ranked = _get_bsi_ranking_hf(model, tokenizer, prompts, n_layers, n_heads)
    top_heads = [list(k) for k, _ in ranked[:N_HEADS]]
    bot_heads = [list(k) for k, _ in ranked[-N_HEADS:]]
    rand_sets = [[list(k) for k, _ in random.sample(ranked, N_HEADS)] for _ in range(N_RAND)]

    result = _hf_ablation_c5(model, tokenizer, prompts, n_layers, n_heads,
                              top_heads, rand_sets, bot_heads, "smollm3-3b", ck)
    per_term = result.pop("per_term")
    if not out_file.exists():
        json.dump(result, open(out_file, "w"), indent=2)
    json.dump({"model": result["model"], "checkpoint": ck, "per_term": per_term},
              open(perterm_file, "w"), indent=2)
    print(f"✅ SmolLM3 C5 → {out_file}  +  per-term → {perterm_file}")
    print(f"   Baseline={result['baseline_rec']:.3f}  Spec={result['specificity']:+.3f}")
    del model; torch.cuda.empty_cache()


# ── OLMo C5 ──────────────────────────────────────────────────────────────────
def run_olmo_c5():
    from utils_model_olmo import load_olmo_with_checkpoint

    ck = "step143k"
    out_file = OUTPUT_DIR / f"olmo_{ck}_c5_canonical41.json"
    perterm_file = OUTPUT_DIR / f"olmo_{ck}_c5_canonical41_perterm.json"
    if perterm_file.exists():
        print(f"⏭  {perterm_file.name} — skipping"); return

    model, tokenizer = load_olmo_with_checkpoint(ck, DEVICE)
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    prompts = load_rec_prompts()

    ranked = _get_bsi_ranking_hf(model, tokenizer, prompts, n_layers, n_heads)
    top_heads = [list(k) for k, _ in ranked[:N_HEADS]]
    bot_heads = [list(k) for k, _ in ranked[-N_HEADS:]]
    rand_sets = [[list(k) for k, _ in random.sample(ranked, N_HEADS)] for _ in range(N_RAND)]

    result = _hf_ablation_c5(model, tokenizer, prompts, n_layers, n_heads,
                              top_heads, rand_sets, bot_heads, "olmo-1b", ck)
    per_term = result.pop("per_term")
    if not out_file.exists():
        json.dump(result, open(out_file, "w"), indent=2)
    json.dump({"model": result["model"], "checkpoint": ck, "per_term": per_term},
              open(perterm_file, "w"), indent=2)
    print(f"✅ OLMo C5 → {out_file}  +  per-term → {perterm_file}")
    print(f"   Baseline={result['baseline_rec']:.3f}  Spec={result['specificity']:+.3f}")
    del model; torch.cuda.empty_cache()


# ── Qwen C5 ──────────────────────────────────────────────────────────────────
def run_qwen_c5():
    from utils_model_qwen import load_qwen

    ck = "final"
    out_file = OUTPUT_DIR / f"qwen_{ck}_c5_canonical41.json"
    perterm_file = OUTPUT_DIR / f"qwen_{ck}_c5_canonical41_perterm.json"
    if perterm_file.exists():
        print(f"⏭  {perterm_file.name} — skipping"); return

    model, tokenizer = load_qwen(DEVICE)
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    prompts = load_rec_prompts()

    ranked = _get_bsi_ranking_hf(model, tokenizer, prompts, n_layers, n_heads)
    top_heads = [list(k) for k, _ in ranked[:N_HEADS]]
    bot_heads = [list(k) for k, _ in ranked[-N_HEADS:]]
    rand_sets = [[list(k) for k, _ in random.sample(ranked, N_HEADS)] for _ in range(N_RAND)]

    result = _hf_ablation_c5(model, tokenizer, prompts, n_layers, n_heads,
                              top_heads, rand_sets, bot_heads, "qwen2.5-1.5b", ck)
    per_term = result.pop("per_term")
    if not out_file.exists():
        json.dump(result, open(out_file, "w"), indent=2)
    json.dump({"model": result["model"], "checkpoint": ck, "per_term": per_term},
              open(perterm_file, "w"), indent=2)
    print(f"✅ Qwen C5 → {out_file}  +  per-term → {perterm_file}")
    print(f"   Baseline={result['baseline_rec']:.3f}  Spec={result['specificity']:+.3f}")
    del model; torch.cuda.empty_cache()


def run_crfm_c5_all_seeds():
    for s in range(1, 6):
        run_crfm_c5(seed=s)


RUNNERS = {
    "crfm": run_crfm_c5,
    "smollm3": run_smollm3_c5,
    "olmo": run_olmo_c5,
    "qwen": run_qwen_c5,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(RUNNERS.keys()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed index for CRFM (1-5). Only used with --model crfm.")
    args = parser.parse_args()

    if args.all:
        for fn in RUNNERS.values():
            fn()
    elif args.model == "crfm":
        if args.seed is not None:
            run_crfm_c5(seed=args.seed)
        else:
            run_crfm_c5_all_seeds()
    elif args.model:
        RUNNERS[args.model]()
    else:
        parser.print_help()
