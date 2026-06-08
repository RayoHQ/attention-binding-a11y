"""C5 causal ablation for OLMo-1B using HuggingFace forward hooks.

Usage:
    python src/run_causal_c5_olmo.py --checkpoint step143k
    python src/run_causal_c5_olmo.py --checkpoint step143k \
        --prompts data/prompts/expanded_terms_100.jsonl
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
from utils_model_olmo import load_olmo_with_checkpoint, OLMO_1B_CHECKPOINTS
from scoring import score_generation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_HEADS_TO_ABLATE = 4
N_RANDOM_BASELINES = 5
SEED = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


# ── BSI extraction via attention outputs ──────────────────────────────────

def get_span_indices(tokenizer, prompt_text, term):
    ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    variants = []
    for form in [term, term.capitalize(), term.title()]:
        variants.append(tokenizer.encode(form, add_special_tokens=False))
        variants.append(tokenizer.encode(" " + form, add_special_tokens=False))
    seen = set(); unique = []
    for v in variants:
        if tuple(v) not in seen: seen.add(tuple(v)); unique.append(v)

    for variant in unique:
        for i in range(len(ids) - len(variant) + 1):
            if ids[i:i+len(variant)] == variant:
                return list(range(i, i+len(variant)))

    # Fallback: character-level
    decoded = tokenizer.decode(ids)
    cp = decoded.lower().find(term.lower())
    if cp >= 0:
        cl = 0
        for idx, tid in enumerate(ids):
            tok_str = tokenizer.decode([tid])
            if cl >= cp:
                n = len(tokenizer.encode(term, add_special_tokens=False))
                return [i for i in range(idx, idx+max(1,n)) if i < len(ids)]
            cl += len(tok_str)
    return list(range(max(0, len(ids)-4), len(ids)))


def compute_bsi_per_head_olmo(model, tokenizer, prompt_text, term):
    """Extract BSI per (layer, head) using output_attentions=True."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
    span = get_span_indices(tokenizer, prompt_text, term)

    with torch.no_grad():
        out = model(**inputs, output_attentions=True)

    head_scores = {}
    for layer_idx, layer_attn in enumerate(out.attentions):
        # layer_attn: [1, n_heads, seq, seq]
        attn = layer_attn[0].float()
        n_heads = attn.shape[0]
        seq_len = attn.shape[1]
        safe_span = [i for i in span if i < seq_len]
        for head_idx in range(n_heads):
            ha = attn[head_idx]
            pairs = [ha[d,s].item() for d in safe_span for s in safe_span if d > s]
            head_scores[(layer_idx, head_idx)] = sum(pairs)/len(pairs) if pairs else 0.0
    return head_scores


def find_top_binding_heads(model, tokenizer, prompts, n=N_HEADS_TO_ABLATE):
    agg = defaultdict(list)
    for p in tqdm(prompts, desc="BSI per head (OLMo)"):
        for k, v in compute_bsi_per_head_olmo(model, tokenizer, p["template"], p["term"]).items():
            agg[k].append(v)
    avg = sorted([(l,h,float(np.mean(v))) for (l,h),v in agg.items()], key=lambda x:-x[2])
    return avg[:n], avg[-n:]


# ── Ablation via attention mask hooks ─────────────────────────────────────

def make_ablation_hooks(ablate_heads, n_heads):
    """Register forward hooks that zero out specific attention heads."""
    if not ablate_heads:
        return []
    by_layer = defaultdict(list)
    for l, h in ablate_heads: by_layer[l].append(h)

    handles = []
    return by_layer   # returned as config; hooks registered in eval functions


def eval_recognition_hf(model, tokenizer, prompts, ablate_by_layer=None):
    handles = []
    if ablate_by_layer:
        # Hook into OLMo attention output projections at each layer
        for layer_idx, heads in ablate_by_layer.items():
            def make_hook(layer_i, head_list):
                def hook_fn(module, input, output):
                    # output[0]: [batch, seq, n_heads*head_dim] — need to split
                    # Use output_attentions path instead (see below)
                    return output
                return hook_fn
            # We'll use a simpler approach: hook on attn output and zero head slices
        # Actually use the attention weight hook approach via output_attentions
        pass

    correct = total = 0
    for p in prompts:
        if p["task"] != "recognition": continue
        choices, ans = p.get("choices"), p.get("answer_idx")
        if choices is None or ans is None: continue

        prompt_ids = tokenizer.encode(p["template"], add_special_tokens=True, return_tensors="pt").to(DEVICE)
        prompt_len = prompt_ids.shape[1]

        choice_scores = []
        for choice in choices:
            choice_tok = tokenizer.encode(" " + choice, add_special_tokens=False)
            choice_t = torch.tensor([choice_tok], device=DEVICE)
            full_ids = torch.cat([prompt_ids, choice_t], dim=1)

            with torch.no_grad():
                logits = _forward_with_ablation(model, full_ids, ablate_by_layer)
                lp = torch.log_softmax(logits.float(), dim=-1)

            total_lp = sum(lp[0, prompt_len-1+i, tid].item() for i,tid in enumerate(choice_tok))
            choice_scores.append(total_lp / max(1, len(choice_tok)))

        pred = int(torch.tensor(choice_scores).argmax())
        correct += int(pred == ans); total += 1

    return correct/total if total else 0.0, correct, total


def _forward_with_ablation(model, input_ids, ablate_by_layer):
    """Forward pass with attention head ablation via hooks on attn output."""
    handles = []
    if ablate_by_layer:
        for layer_idx, heads in ablate_by_layer.items():
            block = model.model.layers[layer_idx]
            # Get head dimension
            def make_hook(layer_i, head_list):
                def hook_fn(module, args, kwargs, output):
                    # output is a tuple; first element is [batch, seq, hidden]
                    attn_out = output[0]  # [B, S, H]
                    hidden = attn_out.shape[-1]
                    n_heads = model.config.num_attention_heads
                    head_dim = hidden // n_heads
                    for h in head_list:
                        attn_out[:, :, h*head_dim:(h+1)*head_dim] = 0.0
                    return (attn_out,) + output[1:]
                return hook_fn
            h = block.self_attn.register_forward_hook(
                make_hook(layer_idx, heads), with_kwargs=True
            )
            handles.append(h)

    with torch.no_grad():
        logits = model(input_ids).logits

    for h in handles: h.remove()
    return logits


def eval_generation_hf(model, tokenizer, prompts, ablate_by_layer=None):
    handles = []
    if ablate_by_layer:
        for layer_idx, heads in ablate_by_layer.items():
            block = model.model.layers[layer_idx]
            def make_hook(layer_i, head_list):
                def hook_fn(module, args, kwargs, output):
                    attn_out = output[0]
                    hidden = attn_out.shape[-1]
                    n_heads = model.config.num_attention_heads
                    head_dim = hidden // n_heads
                    for h in head_list:
                        attn_out[:, :, h*head_dim:(h+1)*head_dim] = 0.0
                    return (attn_out,) + output[1:]
                return hook_fn
            h = block.self_attn.register_forward_hook(
                make_hook(layer_idx, heads), with_kwargs=True
            )
            handles.append(h)

    scores = []
    for p in prompts:
        if p["task"] != "generation": continue
        inputs = tokenizer(p["template"], return_tensors="pt").to(DEVICE)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=p.get("max_tokens", 25),
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        comp = tokenizer.decode(out_ids[0, prompt_len:], skip_special_tokens=True).strip()
        scores.append(score_generation(comp, p["term"]))

    for h in handles: h.remove()
    return float(np.mean(scores)) if scores else 0.0


# ── Main ──────────────────────────────────────────────────────────────────

def run(checkpoint, prompts_file, output_file):
    print(f"\n{'='*65}")
    print(f"  C5 Causal Ablation (OLMo-1B)  |  {checkpoint}")
    print(f"  Prompts: {prompts_file}")
    print(f"{'='*65}\n")

    model, tokenizer = load_olmo_with_checkpoint(checkpoint, DEVICE)
    n_layers = model.config.num_hidden_layers
    n_heads  = model.config.num_attention_heads
    print(f"  {n_layers}L × {n_heads}H = {n_layers*n_heads} total heads\n")

    prompts = [json.loads(l) for l in open(prompts_file)]
    rec_p = [p for p in prompts if p["task"] == "recognition"]
    gen_p = [p for p in prompts if p["task"] == "generation"]
    print(f"  {len(rec_p)} recognition  |  {len(gen_p)} generation prompts\n")

    # Top/bottom heads
    top_data, bot_data = find_top_binding_heads(model, tokenizer, prompts)
    top_heads = [(h[0],h[1]) for h in top_data]
    bot_heads = [(h[0],h[1]) for h in bot_data]
    print(f"\nTop-{N_HEADS_TO_ABLATE} heads:")
    for l,h,b in top_data: print(f"  L{l:2d} H{h:2d}  BSI={b:.4f}")

    all_heads = [(l,hh) for l in range(n_layers) for hh in range(n_heads)]
    conditions = [
        ("BASELINE", None),
        (f"TOP-{N_HEADS_TO_ABLATE} ablated", top_heads),
    ]
    for i in range(N_RANDOM_BASELINES):
        rh = random.sample([x for x in all_heads if x not in top_heads], N_HEADS_TO_ABLATE)
        conditions.append((f"RANDOM trial {i+1}", rh))
    conditions.append((f"BOTTOM-{N_HEADS_TO_ABLATE} ablated", bot_heads))

    print(f"\n{'Condition':<35} {'RecAcc':>8} {'GenScore':>9}")
    print("─"*55)
    rec_res, gen_res = {}, {}
    for name, heads in conditions:
        abl = defaultdict(list)
        if heads:
            for l, h in heads: abl[l].append(h)
        abl_dict = dict(abl) if heads else None

        ra, rc, rt = eval_recognition_hf(model, tokenizer, rec_p, abl_dict)
        gs = eval_generation_hf(model, tokenizer, gen_p, abl_dict)
        rec_res[name] = {"accuracy": ra, "correct": rc, "total": rt}
        gen_res[name] = {"mean_score": gs}
        print(f"  {name:<33} {ra:>8.3f} {gs:>9.3f}")

    bl_rec = rec_res["BASELINE"]["accuracy"]
    bl_gen = gen_res["BASELINE"]["mean_score"]
    top_rd = bl_rec - rec_res[conditions[1][0]]["accuracy"]
    top_gd = bl_gen - gen_res[conditions[1][0]]["mean_score"]
    rnd_rd = float(np.mean([bl_rec - rec_res[n]["accuracy"] for n,_ in conditions[2:-1]]))
    rnd_gd = float(np.mean([bl_gen - gen_res[n]["mean_score"] for n,_ in conditions[2:-1]]))
    bot_rd = bl_rec - rec_res[conditions[-1][0]]["accuracy"]
    bot_gd = bl_gen - gen_res[conditions[-1][0]]["mean_score"]
    spec = ((top_rd+top_gd)/2) - ((rnd_rd+rnd_gd)/2)

    print(f"\n  Top drop:   Rec {top_rd:+.3f}  Gen {top_gd:+.3f}")
    print(f"  Random:     Rec {rnd_rd:+.3f}  Gen {rnd_gd:+.3f}")
    print(f"  Specificity: {spec:+.4f}")
    if spec > 0.10:   print("  → ✅ C5 SUPPORTED")
    elif spec > 0.0:  print("  → ⚠ WEAKLY SUPPORTED")
    else:             print("  → ❌ or ↑ DECOUPLED")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    save = {
        "model": "olmo-1b",
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
            "mean_random_rec_drop": rnd_rd, "mean_random_gen_drop": rnd_gd,
            "bottom_rec_drop": bot_rd, "bottom_gen_drop": bot_gd,
            "specificity": spec,
        },
    }
    with open(output_file, "w") as f: json.dump(save, f, indent=2)
    print(f"\n  Saved → {output_file}")
    del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, choices=list(OLMO_1B_CHECKPOINTS.keys()))
    parser.add_argument("--prompts", default="data/prompts/expanded_terms_100.jsonl")
    parser.add_argument("--output",  default=None)
    args = parser.parse_args()

    if args.output is None:
        stem = Path(args.prompts).stem.replace("expanded_terms_","")
        args.output = f"data/results/causal/olmo_1b_{args.checkpoint}_c5_{stem}.json"

    run(args.checkpoint, args.prompts, args.output)
