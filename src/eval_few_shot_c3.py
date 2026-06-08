"""Generalised C3 few-shot unlockability evaluation.

Handles Pythia (TransformerLens) and OLMo (HuggingFace).

Usage:
    python src/eval_few_shot_c3.py --model 2.8b  --checkpoint step15000
    python src/eval_few_shot_c3.py --model olmo  --checkpoint step15k
    python src/eval_few_shot_c3.py --model 2.8b  --checkpoint step143000
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from scoring import score_generation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPTS_FILE = Path("data/prompts/expanded_terms_100.jsonl")
OUTPUT_DIR   = Path("data/results/few_shot_c3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# One-shot examples for all 9 terms
FEW_SHOT_EXAMPLES = {
    "screen reader": (
        "Example: A screen reader is assistive software that reads "
        "digital text aloud for blind or visually impaired users.\n\n"
    ),
    "skip link": (
        "Example: A skip link is a keyboard-accessible link that allows "
        "users to bypass navigation and jump directly to the main content "
        "of a webpage.\n\n"
    ),
    "alt text": (
        "Example: Alt text is a written description of an image that "
        "screen readers read aloud to blind users, conveying the content "
        "and function of the image.\n\n"
    ),
    "color contrast": (
        "Example: Color contrast refers to the difference in brightness "
        "between text and its background, ensuring readability for users "
        "with low vision or color blindness.\n\n"
    ),
    "focus indicator": (
        "Example: A focus indicator is a visual highlight that shows which "
        "element on a webpage is currently selected for keyboard interaction, "
        "helping keyboard-only users navigate.\n\n"
    ),
    "heading structure": (
        "Example: Heading structure is the hierarchical organization of headings "
        "(H1, H2, H3) that allows screen reader users to understand page "
        "organization and navigate efficiently.\n\n"
    ),
    "aria attribute": (
        "Example: An ARIA attribute is an HTML attribute that provides additional "
        "accessibility information to assistive technologies, describing the role "
        "or state of interactive elements.\n\n"
    ),
    "keyboard navigation": (
        "Example: Keyboard navigation is the ability to operate all interactive "
        "elements of a website using only keyboard input, essential for users "
        "with motor disabilities.\n\n"
    ),
    "landmark region": (
        "Example: A landmark region is a semantically meaningful section of a "
        "webpage (navigation, main content, footer) that helps screen reader "
        "users quickly jump between page areas.\n\n"
    ),
}


# ── Model-agnostic generation ─────────────────────────────────────────────

def generate_pythia(model, template, max_new_tokens):
    tokens = model.to_tokens(template)
    with torch.no_grad():
        out = model.generate(tokens, max_new_tokens=max_new_tokens,
                             temperature=0.0, do_sample=False)
    text = model.tokenizer.decode(out[0], skip_special_tokens=True)
    return text[len(template):].strip()


def generate_olmo(model, tokenizer, template, max_new_tokens):
    inputs = tokenizer(template, return_tensors="pt").to(DEVICE)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True).strip()


# ── Core evaluation ───────────────────────────────────────────────────────

def evaluate_checkpoint(model_size, checkpoint):
    out_file = OUTPUT_DIR / f"{model_size}_{checkpoint}_c3_fewshot.json"
    if out_file.exists():
        print(f"  ⏭  {model_size} {checkpoint} already done — skipping")
        return json.load(open(out_file))

    print(f"\n{'='*65}")
    print(f"  C3 Few-shot  |  {model_size}  |  {checkpoint}")
    print(f"{'='*65}")

    # Load model
    is_olmo = model_size == "olmo"
    if is_olmo:
        from utils_model_olmo import load_olmo_with_checkpoint
        model, tokenizer = load_olmo_with_checkpoint(checkpoint, DEVICE)
        gen_fn = lambda tmpl, mxt: generate_olmo(model, tokenizer, tmpl, mxt)
    else:
        from utils_model import load_pythia_with_checkpoint
        model = load_pythia_with_checkpoint(model_size, checkpoint, DEVICE)
        gen_fn = lambda tmpl, mxt: generate_pythia(model, tmpl, mxt)

    # Load generation prompts only
    gen_prompts = [json.loads(l) for l in open(PROMPTS_FILE) if json.loads(l)["task"] == "generation"]
    n_terms = len(set(p["term"] for p in gen_prompts))
    print(f"  {len(gen_prompts)} generation prompts / {n_terms} terms")

    zs_scores, fs_scores = {}, {}
    zs_results, fs_results = [], []

    for p in tqdm(gen_prompts, desc=f"{model_size}/{checkpoint}"):
        term   = p["term"]
        mxt    = p.get("max_tokens", 25)
        prefix = FEW_SHOT_EXAMPLES.get(term, "")

        zs_comp = gen_fn(p["template"], mxt)
        fs_comp = gen_fn(prefix + p["template"], mxt)

        zs_s = score_generation(zs_comp, term)
        fs_s = score_generation(fs_comp, term)

        zs_scores.setdefault(term, []).append(zs_s)
        fs_scores.setdefault(term, []).append(fs_s)
        zs_results.append({"term": term, "prompt_id": p["prompt_id"], "score": zs_s, "completion": zs_comp})
        fs_results.append({"term": term, "prompt_id": p["prompt_id"], "score": fs_s, "completion": fs_comp})

    zs_mean = sum(r["score"] for r in zs_results) / len(zs_results)
    fs_mean = sum(r["score"] for r in fs_results) / len(fs_results)
    imp_pp  = (fs_mean - zs_mean) * 100
    rel_pct = imp_pp / zs_mean if zs_mean > 0 else float("inf")

    print(f"\n  Zero-shot mean: {zs_mean:.4f}")
    print(f"  Few-shot mean:  {fs_mean:.4f}")
    print(f"  Improvement:    {imp_pp:+.1f} pp  ({rel_pct:+.1f}% relative)")
    rep = "✅ C3 SUPPORTED" if imp_pp > 20 else ("⚠ WEAK" if imp_pp > 5 else "❌")
    print(f"  → {rep}")

    result = {
        "model": model_size,
        "checkpoint": checkpoint,
        "n_prompts": len(gen_prompts),
        "n_terms": n_terms,
        "zero_shot_mean": round(zs_mean, 4),
        "few_shot_mean":  round(fs_mean,  4),
        "improvement_pp": round(imp_pp, 1),
        "improvement_relative_pct": round(rel_pct, 1),
        "zero_shot_per_term": {t: round(sum(v)/len(v), 4) for t,v in zs_scores.items()},
        "few_shot_per_term":  {t: round(sum(v)/len(v), 4) for t,v in fs_scores.items()},
    }
    with open(out_file, "w") as f: json.dump(result, f, indent=2)
    print(f"  Saved → {out_file}")

    del model
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    evaluate_checkpoint(args.model, args.checkpoint)
