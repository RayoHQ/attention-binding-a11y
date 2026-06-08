"""Generalised C5 causal ablation — Pythia (TransformerLens).

Usage:
    python src/run_causal_c5.py --model 1b --checkpoint step120000
    python src/run_causal_c5.py --model 160m --checkpoint step120000 \
        --prompts data/prompts/expanded_terms_tier123.jsonl \
        --output data/results/causal/160m_step120k_c5_tier123.json
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_model import load_pythia_with_checkpoint
from scoring import score_recognition_logprob, score_generation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_HEADS_TO_ABLATE = 4
N_RANDOM_BASELINES = 5
SEED = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


# ── Per-head BSI ─────────────────────────────────────────────────────────

def compute_per_head_bsi(model, prompt_text, term):
    from extract_attention import TERM_ALIASES
    tokens = model.to_tokens(prompt_text, prepend_bos=True)
    seq_len = tokens.shape[1]
    tokenizer = model.tokenizer

    search_terms = [term] + TERM_ALIASES.get(term.lower(), [])
    variants = []
    for t in search_terms:
        for form in [t, t.capitalize(), t.title()]:
            variants.append(tokenizer.encode(form, add_special_tokens=False))
            variants.append(tokenizer.encode(" " + form, add_special_tokens=False))
    seen = set(); unique_variants = []
    for v in variants:
        if tuple(v) not in seen:
            seen.add(tuple(v)); unique_variants.append(v)

    full_ids = tokens[0].tolist()
    span_start = None
    term_tokens = unique_variants[0]
    for variant in unique_variants:
        for i in range(len(full_ids) - len(variant) + 1):
            if full_ids[i:i+len(variant)] == variant:
                span_start = i; term_tokens = variant; break
        if span_start is not None: break
    if span_start is None:
        decoded = [tokenizer.decode([t]) for t in full_ids]
        joined = "".join(decoded)
        cp = joined.lower().find(term.lower())
        if cp >= 0:
            cl = 0
            for idx, dt in enumerate(decoded):
                if cl >= cp: span_start = idx; break
                cl += len(dt)
        if span_start is None:
            span_start = max(0, seq_len - len(term_tokens) - 5)
    span_indices = [i for i in range(span_start, span_start + len(term_tokens)) if i < seq_len]

    head_scores = {}
    for layer_idx in range(model.cfg.n_layers):
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=[f"blocks.{layer_idx}.attn.hook_pattern"],
                stop_at_layer=layer_idx + 1,
            )
        attn = cache[f"blocks.{layer_idx}.attn.hook_pattern"]
        for head_idx in range(model.cfg.n_heads):
            ha = attn[0, head_idx]
            pairs = [ha[d, s].item() for d in span_indices for s in span_indices if d > s]
            head_scores[(layer_idx, head_idx)] = sum(pairs)/len(pairs) if pairs else 0.0
        del cache; torch.cuda.empty_cache()
    return head_scores


def find_top_binding_heads(model, prompts, n=N_HEADS_TO_ABLATE):
    agg = defaultdict(list)
    for p in tqdm(prompts, desc="Computing BSI per head"):
        for k, v in compute_per_head_bsi(model, p["template"], p["term"]).items():
            agg[k].append(v)
    avg = sorted([(l,h,float(np.mean(v))) for (l,h),v in agg.items()], key=lambda x:-x[2])
    return avg[:n], avg[-n:]


# ── Ablation evaluation ───────────────────────────────────────────────────

def _make_hooks(ablate_heads):
    if not ablate_heads: return []
    by_layer = defaultdict(list)
    for l, h in ablate_heads: by_layer[l].append(h)
    hooks = []
    for layer, heads in by_layer.items():
        def make_fn(hs):
            def fn(act, hook):
                for h in hs: act[:, h, :, :] = 0.0
                return act
            return fn
        hooks.append((f"blocks.{layer}.attn.hook_pattern", make_fn(heads)))
    return hooks


def eval_recognition(model, prompts, ablate_heads=None):
    hooks = _make_hooks(ablate_heads)
    correct = total = 0
    for p in prompts:
        if p["task"] != "recognition": continue
        choices, ans = p.get("choices"), p.get("answer_idx")
        if choices is None or ans is None: continue
        if hooks:
            with model.hooks(fwd_hooks=hooks):
                r = score_recognition_logprob(model, p["template"], choices, ans)
        else:
            r = score_recognition_logprob(model, p["template"], choices, ans)
        correct += int(r["is_correct"]); total += 1
    return correct/total if total else 0.0, correct, total


def eval_generation(model, prompts, ablate_heads=None):
    hooks = _make_hooks(ablate_heads)
    scores = []
    for p in prompts:
        if p["task"] != "generation": continue
        tokens = model.to_tokens(p["template"])
        max_tok = p.get("max_tokens", 25)
        with torch.no_grad():
            if hooks:
                with model.hooks(fwd_hooks=hooks):
                    out = model.generate(tokens, max_new_tokens=max_tok,
                                         temperature=0.0, do_sample=False)
            else:
                out = model.generate(tokens, max_new_tokens=max_tok,
                                     temperature=0.0, do_sample=False)
        text = model.tokenizer.decode(out[0], skip_special_tokens=True)
        comp = text[len(p["template"]):].strip()
        scores.append(score_generation(comp, p["term"]))
    return float(np.mean(scores)) if scores else 0.0


# ── Main ──────────────────────────────────────────────────────────────────

def run(model_size, checkpoint, prompts_file, output_file):
    print(f"\n{'='*65}")
    print(f"  C5 Causal Ablation  |  pythia-{model_size}  |  {checkpoint}")
    print(f"  Prompts: {prompts_file}")
    print(f"{'='*65}\n")

    model = load_pythia_with_checkpoint(model_size, checkpoint, DEVICE)
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    print(f"  {n_layers}L × {n_heads}H = {n_layers*n_heads} total heads\n")

    prompts = [json.loads(l) for l in open(prompts_file)]
    rec_p = [p for p in prompts if p["task"] == "recognition"]
    gen_p = [p for p in prompts if p["task"] == "generation"]
    print(f"  {len(rec_p)} recognition  |  {len(gen_p)} generation prompts\n")

    # Step 1: identify top/bottom heads
    top_data, bot_data = find_top_binding_heads(model, prompts)
    top_heads = [(h[0],h[1]) for h in top_data]
    bot_heads = [(h[0],h[1]) for h in bot_data]
    print(f"\nTop-{N_HEADS_TO_ABLATE} heads (layer, head, BSI):")
    for l,h,b in top_data: print(f"  L{l:2d} H{h:2d}  BSI={b:.4f}")

    # Step 2: build conditions
    all_heads = [(l,h) for l in range(n_layers) for h in range(n_heads)]
    conditions = [
        ("BASELINE", None),
        (f"TOP-{N_HEADS_TO_ABLATE} ablated", top_heads),
    ]
    for i in range(N_RANDOM_BASELINES):
        rh = random.sample([x for x in all_heads if x not in top_heads], N_HEADS_TO_ABLATE)
        conditions.append((f"RANDOM trial {i+1}", rh))
    conditions.append((f"BOTTOM-{N_HEADS_TO_ABLATE} ablated", bot_heads))

    # Step 3: evaluate
    print(f"\n{'Condition':<35} {'RecAcc':>8} {'GenScore':>9}")
    print("─"*55)
    rec_res, gen_res = {}, {}
    for name, heads in conditions:
        ra, rc, rt = eval_recognition(model, rec_p, heads)
        gs = eval_generation(model, gen_p, heads)
        rec_res[name] = {"accuracy": ra, "correct": rc, "total": rt}
        gen_res[name] = {"mean_score": gs}
        print(f"  {name:<33} {ra:>8.3f} {gs:>9.3f}")

    # Step 4: compute drops & specificity
    bl_rec = rec_res["BASELINE"]["accuracy"]
    bl_gen = gen_res["BASELINE"]["mean_score"]
    top_rd = bl_rec - rec_res[conditions[1][0]]["accuracy"]
    top_gd = bl_gen - gen_res[conditions[1][0]]["mean_score"]
    rnd_rd = np.mean([bl_rec - rec_res[n]["accuracy"] for n,_ in conditions[2:-1]])
    rnd_gd = np.mean([bl_gen - gen_res[n]["mean_score"] for n,_ in conditions[2:-1]])
    bot_rd = bl_rec - rec_res[conditions[-1][0]]["accuracy"]
    bot_gd = bl_gen - gen_res[conditions[-1][0]]["mean_score"]
    spec = ((top_rd + top_gd)/2) - ((rnd_rd + rnd_gd)/2)

    print(f"\n  Top ablation drop:    Rec {top_rd:+.3f}  Gen {top_gd:+.3f}")
    print(f"  Random drop (mean):   Rec {rnd_rd:+.3f}  Gen {rnd_gd:+.3f}")
    print(f"  Bottom drop:          Rec {bot_rd:+.3f}  Gen {bot_gd:+.3f}")
    print(f"  Specificity:          {spec:+.4f}")
    if spec > 0.10:   print("  → ✅ C5 SUPPORTED")
    elif spec > 0.0:  print("  → ⚠ C5 WEAKLY SUPPORTED")
    else:             print("  → ❌ or ↑ DECOUPLED (ablation helps)")

    # Save
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    save = {
        "model": f"pythia-{model_size}-deduped",
        "checkpoint": checkpoint,
        "prompts_file": str(prompts_file),
        "n_rec_prompts": len(rec_p),
        "n_gen_prompts": len(gen_p),
        "n_heads_ablated": N_HEADS_TO_ABLATE,
        "n_random_trials": N_RANDOM_BASELINES,
        "top_heads": [{"layer":l,"head":h,"avg_bsi":b} for l,h,b in top_data],
        "bottom_heads": [{"layer":l,"head":h,"avg_bsi":b} for l,h,b in bot_data],
        "recognition": {n: rec_res[n] for n in rec_res},
        "generation":  {n: gen_res[n]  for n in gen_res},
        "drops": {
            "top_rec_drop": top_rd, "top_gen_drop": top_gd,
            "mean_random_rec_drop": float(rnd_rd), "mean_random_gen_drop": float(rnd_gd),
            "bottom_rec_drop": bot_rd, "bottom_gen_drop": bot_gd,
            "specificity": spec,
        },
    }
    with open(output_file, "w") as f: json.dump(save, f, indent=2)
    print(f"\n  Saved → {output_file}")
    del model; torch.cuda.empty_cache()
    return spec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True, choices=["160m","1b","2.8b"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts",    default="data/prompts/expanded_terms_100.jsonl")
    parser.add_argument("--output",     default=None)
    args = parser.parse_args()

    if args.output is None:
        stem = Path(args.prompts).stem.replace("expanded_terms_","").replace("_","")
        args.output = f"data/results/causal/{args.model}_{args.checkpoint}_c5_{stem}.json"

    run(args.model, args.checkpoint, args.prompts, args.output)
