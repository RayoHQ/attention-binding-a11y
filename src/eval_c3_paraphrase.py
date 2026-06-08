"""C3 paraphrase-exemplar control experiment.

Tests whether the few-shot gains from eval_few_shot_c3.py are driven by
copying keywords from the exemplar prefix, or by genuine knowledge retrieval.

Design:
  - ZS:            zero-shot baseline (no prefix)
  - FS_original:   standard exemplar (from eval_few_shot_c3.py, uses rubric vocabulary)
  - FS_paraphrase: paraphrased exemplar (same concept, different vocabulary)

Interpretation:
  - FS_orig ≈ FS_para >> ZS  →  genuine knowledge retrieval (copying unlikely)
  - FS_orig >> FS_para > ZS  →  partial copying (exemplar phrasing drives score)
  - FS_orig ≈ FS_para ≈ ZS  →  exemplars don't help (no latent knowledge)

Focuses on Pythia early checkpoints where copying effect would be largest
(EB* high, ZS low, maximum few-shot leverage).

Usage:
    python src/eval_c3_paraphrase.py --model 160m --checkpoint step15000
    python src/eval_c3_paraphrase.py --model 1b   --checkpoint step15000
    python src/eval_c3_paraphrase.py --model 2.8b --checkpoint step15000
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
OUTPUT_DIR   = Path("data/results/few_shot_c3_paraphrase")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Original exemplars (from eval_few_shot_c3.py) ─────────────────────────────
FEW_SHOT_ORIGINAL = {
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

# ── Paraphrase exemplars ───────────────────────────────────────────────────────
# Same concept, noticeably different vocabulary.
# Deliberately avoid the highest-frequency scoring keywords so that
# a model copying the prefix verbatim would produce different (lower-scoring)
# completions than one drawing on internal knowledge.
FEW_SHOT_PARAPHRASE = {
    "screen reader": (
        "Example: A screen reader is a utility that translates on-screen "
        "content into synthesized speech or refreshable braille, enabling "
        "people without functional vision to operate computers independently.\n\n"
    ),
    "skip link": (
        "Example: A skip link is a hidden anchor element placed before "
        "repetitive page furniture that lets non-mouse visitors jump directly "
        "to the primary body of the page, bypassing menus.\n\n"
    ),
    "alt text": (
        "Example: Alt text is a brief verbal equivalent of a graphic embedded "
        "in HTML markup that lets people relying on audio output understand "
        "images they cannot see.\n\n"
    ),
    "color contrast": (
        "Example: Color contrast is the measurable luminance gap between "
        "foreground and background, which WCAG quantifies as a ratio to verify "
        "legibility for people with low vision or colour-perception differences.\n\n"
    ),
    "focus indicator": (
        "Example: A focus indicator is the visible outline or ring surrounding "
        "an interactive widget when activated via keyboard, revealing the current "
        "position in the document for non-mouse operators.\n\n"
    ),
    "heading structure": (
        "Example: Heading structure is the nested system of HTML heading tags "
        "from h1 through h6 that creates a navigable document outline for people "
        "using assistive tools to move between sections.\n\n"
    ),
    "aria attribute": (
        "Example: An ARIA attribute is a WAI-ARIA property added to markup that "
        "exposes semantic meaning—such as role, state, or label—to assistive "
        "programs that cannot infer intent from visual layout alone.\n\n"
    ),
    "keyboard navigation": (
        "Example: Keyboard navigation is the capacity to interact with every "
        "operable element on a site through key presses alone, without a pointing "
        "device, critical for people with limited hand dexterity.\n\n"
    ),
    "landmark region": (
        "Example: A landmark region is a named page area defined by ARIA roles "
        "or HTML5 sectioning elements that allows assistive-tool users to "
        "teleport directly to major sections of a document.\n\n"
    ),
}


def generate_pythia(model, template, max_new_tokens):
    tokens = model.to_tokens(template)
    with torch.no_grad():
        out = model.generate(tokens, max_new_tokens=max_new_tokens,
                             temperature=0.0, do_sample=False)
    text = model.tokenizer.decode(out[0], skip_special_tokens=True)
    return text[len(template):].strip()


def evaluate_checkpoint(model_size, checkpoint):
    out_file = OUTPUT_DIR / f"{model_size}_{checkpoint}_c3_paraphrase.json"
    if out_file.exists():
        print(f"  ⏭  {model_size} {checkpoint} already done — skipping")
        return json.load(open(out_file))

    print(f"\n{'='*65}")
    print(f"  C3 Paraphrase Control  |  {model_size}  |  {checkpoint}")
    print(f"{'='*65}")

    from utils_model import load_pythia_with_checkpoint
    model = load_pythia_with_checkpoint(model_size, checkpoint, DEVICE)
    gen_fn = lambda tmpl, mxt: generate_pythia(model, tmpl, mxt)

    gen_prompts = [json.loads(l) for l in open(PROMPTS_FILE)
                   if json.loads(l)["task"] == "generation"]
    n_terms = len(set(p["term"] for p in gen_prompts))
    print(f"  {len(gen_prompts)} prompts / {n_terms} terms")

    zs_s, orig_s, para_s = {}, {}, {}
    zs_r, orig_r, para_r = [], [], []

    for p in tqdm(gen_prompts, desc=f"{model_size}/{checkpoint}"):
        term  = p["term"]
        mxt   = p.get("max_tokens", 25)
        orig_prefix = FEW_SHOT_ORIGINAL.get(term, "")
        para_prefix = FEW_SHOT_PARAPHRASE.get(term, "")

        zs_comp   = gen_fn(p["template"], mxt)
        orig_comp = gen_fn(orig_prefix + p["template"], mxt)
        para_comp = gen_fn(para_prefix + p["template"], mxt) if para_prefix else orig_comp

        sv_zs   = score_generation(zs_comp,   term)
        sv_orig = score_generation(orig_comp, term)
        sv_para = score_generation(para_comp, term)

        zs_s.setdefault(term, []).append(sv_zs)
        orig_s.setdefault(term, []).append(sv_orig)
        para_s.setdefault(term, []).append(sv_para)

        zs_r.append({"term": term, "prompt_id": p["prompt_id"], "score": sv_zs,
                     "completion": zs_comp})
        orig_r.append({"term": term, "prompt_id": p["prompt_id"], "score": sv_orig,
                       "completion": orig_comp})
        para_r.append({"term": term, "prompt_id": p["prompt_id"], "score": sv_para,
                       "completion": para_comp})

    zs_mean   = sum(r["score"] for r in zs_r)   / len(zs_r)
    orig_mean = sum(r["score"] for r in orig_r) / len(orig_r)
    para_mean = sum(r["score"] for r in para_r) / len(para_r)

    delta_orig = (orig_mean - zs_mean) * 100
    delta_para = (para_mean - zs_mean) * 100
    delta_diff = (orig_mean - para_mean) * 100

    print(f"\n  Zero-shot mean:      {zs_mean:.4f}")
    print(f"  FS-original mean:    {orig_mean:.4f}  (Δ={delta_orig:+.1f} pp vs ZS)")
    print(f"  FS-paraphrase mean:  {para_mean:.4f}  (Δ={delta_para:+.1f} pp vs ZS)")
    print(f"  Original − Para:     {delta_diff:+.1f} pp")

    if abs(delta_diff) <= 3.0:
        verdict = "✅ KNOWLEDGE: orig ≈ para — copying unlikely"
    elif delta_diff > 10.0:
        verdict = "⚠  COPYING: orig >> para — exemplar phrasing drives score"
    else:
        verdict = "~  MIXED: moderate copying signal"
    print(f"  → {verdict}")

    print(f"\n  Per-term breakdown:")
    print(f"  {'Term':<22}  {'ZS':>6}  {'FS-orig':>8}  {'FS-para':>8}  {'Δorig':>7}  {'Δpara':>7}  {'orig-para':>10}")
    all_terms = sorted(zs_s.keys())
    for t in all_terms:
        m_zs   = sum(zs_s[t])   / len(zs_s[t])
        m_orig = sum(orig_s[t]) / len(orig_s[t])
        m_para = sum(para_s[t]) / len(para_s[t])
        print(f"  {t:<22}  {m_zs:>6.3f}  {m_orig:>8.3f}  {m_para:>8.3f}  "
              f"{(m_orig-m_zs)*100:>+6.1f}  {(m_para-m_zs)*100:>+6.1f}  "
              f"{(m_orig-m_para)*100:>+9.1f}")

    result = {
        "model": model_size, "checkpoint": checkpoint,
        "n_prompts": len(gen_prompts), "n_terms": n_terms,
        "zero_shot_mean":      round(zs_mean,   4),
        "fs_original_mean":    round(orig_mean, 4),
        "fs_paraphrase_mean":  round(para_mean, 4),
        "delta_orig_pp":       round(delta_orig, 2),
        "delta_para_pp":       round(delta_para, 2),
        "orig_minus_para_pp":  round(delta_diff, 2),
        "verdict": verdict,
        "per_term": {
            t: {
                "zs":   round(sum(zs_s[t])   / len(zs_s[t]),   4),
                "orig": round(sum(orig_s[t]) / len(orig_s[t]), 4),
                "para": round(sum(para_s[t]) / len(para_s[t]), 4),
            }
            for t in all_terms
        },
    }
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved → {out_file}")

    del model
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True,
                        help="160m | 1b | 2.8b")
    parser.add_argument("--checkpoint", required=True,
                        help="e.g. step15000 | step143000")
    args = parser.parse_args()
    evaluate_checkpoint(args.model, args.checkpoint)
