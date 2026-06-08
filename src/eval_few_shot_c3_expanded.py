"""C3 few-shot unlockability — expanded to all 41 canonical terms.

Extends eval_few_shot_c3.py with hand-authored exemplars for all T-V3 (Tier 1/2/3)
and T-V4 (Wave-2) terms. Uses generation prompts from data/prompts/canonical_45terms.jsonl.

Usage:
    python src/eval_few_shot_c3_expanded.py --model 160m     --checkpoint step15000
    python src/eval_few_shot_c3_expanded.py --model 1b       --checkpoint step143000
    python src/eval_few_shot_c3_expanded.py --model 2.8b     --checkpoint step15000
    python src/eval_few_shot_c3_expanded.py --model olmo     --checkpoint step15k
    python src/eval_few_shot_c3_expanded.py --model crfm1    --checkpoint checkpoint-400000
    python src/eval_few_shot_c3_expanded.py --model crfm1    --checkpoint checkpoint-1000
    python src/eval_few_shot_c3_expanded.py --model smollm3  --checkpoint step40k
    python src/eval_few_shot_c3_expanded.py --model smollm3  --checkpoint step3440k
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
PROMPTS_FILE = Path("data/prompts/canonical_45terms.jsonl")
OUTPUT_DIR   = Path("data/results/few_shot_c3_expanded")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── CRFM GPT-2 Small checkpoint/seed map ─────────────────────────────────────
CRFM_SEED_MODEL_IDS = {
    1: "stanford-crfm/alias-gpt2-small-x21",
    2: "stanford-crfm/battlestar-gpt2-small-x49",
    3: "stanford-crfm/caprica-gpt2-small-x81",
    4: "stanford-crfm/darkmatter-gpt2-small-x343",
    5: "stanford-crfm/expanse-gpt2-small-x777",
}

# ── Few-shot exemplars for all 41 canonical terms ─────────────────────────────
# T-V2 Set B (9 terms) — identical to eval_few_shot_c3.py
# T-V3 Tier 1/2/3 (20 new terms)
# T-V4 Wave-2 (12 terms)
FEW_SHOT_EXAMPLES = {
    # ── T-V2 Set B (original 9) ───────────────────────────────────────────────
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

    # ── T-V3 Tier 1: AT hardware / core concepts (10 new) ────────────────────
    "braille display": (
        "Example: A braille display is a hardware device that converts digital "
        "text into tactile patterns of raised dots, allowing blind or deaf-blind "
        "users to read by touch.\n\n"
    ),
    "screen magnifier": (
        "Example: A screen magnifier is software that enlarges portions of a "
        "computer display, enabling users with low vision to perceive text and "
        "images that would otherwise be too small.\n\n"
    ),
    "voice control": (
        "Example: Voice control is a hands-free input method that lets users "
        "operate software by speaking commands aloud, essential for people with "
        "motor impairments who cannot use a keyboard or mouse.\n\n"
    ),
    "switch access": (
        "Example: Switch access is an assistive input method where users operate "
        "a device with one or more large buttons that scan through on-screen options, "
        "enabling interaction for people with severe motor disabilities.\n\n"
    ),
    "audio description": (
        "Example: An audio description is a narrated track added to video content "
        "that verbally describes visual scenes and actions for viewers who are "
        "blind or have low vision.\n\n"
    ),
    "captions closed": (
        "Example: Closed captions are time-synchronized text transcripts of spoken "
        "dialogue and relevant sounds in a video, toggled on or off by the viewer, "
        "providing access for deaf or hard-of-hearing users.\n\n"
    ),
    "cognitive load": (
        "Example: Cognitive load in accessibility refers to the mental effort "
        "required to understand an interface, which designers minimize through "
        "clear language, simple layouts, and reduced complexity for users with "
        "cognitive disabilities.\n\n"
    ),
    "high contrast": (
        "Example: High contrast mode is an accessibility display setting that "
        "intensifies the visual difference between text and background colors, "
        "improving legibility for users with low vision or light sensitivity.\n\n"
    ),
    "keyboard shortcut": (
        "Example: A keyboard shortcut is a combination of key presses that "
        "triggers an action without navigating through menus, reducing the "
        "interaction burden for users with motor disabilities.\n\n"
    ),
    "text resize": (
        "Example: Text resize is the ability to increase font size via browser "
        "zoom or user stylesheets without breaking page layout, ensuring "
        "readability for users with low vision.\n\n"
    ),

    # ── T-V3 Tier 2: WCAG 2.2 Success Criteria (7 new — keyboard nav already above) ──
    "focus management": (
        "Example: Focus management is the practice of programmatically controlling "
        "where keyboard focus lands during dynamic page changes, such as moving "
        "focus into a modal dialog when it opens.\n\n"
    ),
    "skip navigation": (
        "Example: Skip navigation is a technique providing links at the top of a "
        "page so keyboard users can jump past repetitive navigation menus directly "
        "to the main content.\n\n"
    ),
    "reflow content": (
        "Example: Content reflow is the ability of a web page to reorganize its "
        "layout into a single scrolling column when zoomed to 400%, preventing "
        "horizontal scrolling for users with low vision.\n\n"
    ),
    "non-text content": (
        "Example: Non-text content refers to images, charts, and icons that require "
        "a text alternative so assistive technologies can communicate their meaning "
        "to users who cannot see them.\n\n"
    ),
    "error identification": (
        "Example: Error identification is the practice of clearly labeling form "
        "input errors with descriptive text messages so all users, including those "
        "using screen readers, can understand what went wrong.\n\n"
    ),
    "input purpose": (
        "Example: Input purpose is the programmatic identification of form field "
        "semantics using autocomplete attributes, helping users with cognitive "
        "disabilities by enabling browsers to auto-populate familiar information.\n\n"
    ),
    "text spacing": (
        "Example: Text spacing refers to the ability to override letter, word, and "
        "line spacing properties without loss of content or functionality, "
        "accommodating users with dyslexia or low vision who need custom spacing.\n\n"
    ),

    # ── T-V3 Tier 3: WAI-ARIA roles (3) ─────────────────────────────────────
    "live region": (
        "Example: A live region is an ARIA-marked container whose content updates "
        "dynamically and is automatically announced by screen readers, keeping "
        "users informed of real-time changes without requiring focus movement.\n\n"
    ),
    "alert dialog": (
        "Example: An alert dialog is a modal overlay that interrupts the user with "
        "critical information requiring immediate response, with focus trapped "
        "inside so screen reader users cannot accidentally miss it.\n\n"
    ),
    "tree grid": (
        "Example: A tree grid is an interactive widget combining hierarchical tree "
        "navigation with tabular data display, using ARIA keyboard conventions to "
        "make complex nested data accessible.\n\n"
    ),

    # ── T-V4 Wave-2 (12 terms) ────────────────────────────────────────────────
    "contrast ratio": (
        "Example: A contrast ratio is a mathematical value expressing the luminance "
        "difference between two colors, calculated by WCAG formulas to verify that "
        "text meets the 4.5:1 minimum for normal-size text.\n\n"
    ),
    "eye tracking": (
        "Example: Eye tracking is an assistive input technology that allows users "
        "to control a computer cursor with their gaze, providing access for people "
        "with severe motor disabilities who cannot use their hands.\n\n"
    ),
    "time limits": (
        "Example: Time limits on interactive content must provide options to extend "
        "or disable session timeouts to accommodate users with cognitive, motor, or "
        "visual disabilities who may need more time.\n\n"
    ),
    "reduced motion": (
        "Example: Reduced motion is a CSS media query (prefers-reduced-motion) that "
        "detects when a user has requested minimal animation, allowing developers to "
        "disable vestibular-triggering effects like parallax or spinning.\n\n"
    ),
    "focus trap": (
        "Example: A focus trap intentionally confines keyboard focus within a modal "
        "dialog or overlay, preventing users from accidentally interacting with "
        "background content while the dialog is open.\n\n"
    ),
    "sign language": (
        "Example: Sign language accessibility involves providing video-based signed "
        "language versions of audio content so that deaf users who prefer sign "
        "language over written captions can access multimedia.\n\n"
    ),
    "touch target size": (
        "Example: Touch target size refers to the minimum clickable area recommended "
        "for interactive elements on touchscreens, with WCAG 2.5.5 specifying at "
        "least 44 by 44 CSS pixels to accommodate users with motor impairments.\n\n"
    ),
    "haptic feedback": (
        "Example: Haptic feedback is tactile vibration or physical response from a "
        "device that confirms user actions, providing an accessible non-visual and "
        "non-auditory notification channel for users with sensory differences.\n\n"
    ),
    "plain language": (
        "Example: Plain language is a writing approach that prioritizes clear, simple "
        "vocabulary over technical jargon, improving comprehension for users with "
        "cognitive disabilities, low literacy, or limited language proficiency.\n\n"
    ),
    "motion sensitivity": (
        "Example: Motion sensitivity in accessibility refers to the risk that moving "
        "or flashing content can trigger adverse physical reactions such as dizziness "
        "or seizures in users with vestibular disorders or photosensitivity.\n\n"
    ),
    "semantic html": (
        "Example: Semantic HTML is the practice of using elements that convey "
        "meaning about their content, such as nav, main, and article, providing "
        "inherent structure that assistive technologies can interpret.\n\n"
    ),
    "orientation support": (
        "Example: Orientation support is the requirement that web content remains "
        "functional in both portrait and landscape orientations, avoiding locks to "
        "a single direction that disadvantage users with fixed device mounts.\n\n"
    ),
}


# ── Model-agnostic generation ─────────────────────────────────────────────────

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


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate_checkpoint(model_size, checkpoint):
    out_file = OUTPUT_DIR / f"{model_size}_{checkpoint}_c3_expanded.json"
    if out_file.exists():
        print(f"  ⏭  {model_size} {checkpoint} already done — skipping")
        return json.load(open(out_file))

    print(f"\n{'='*65}")
    print(f"  C3 Expanded (41 terms)  |  {model_size}  |  {checkpoint}")
    print(f"{'='*65}")

    is_olmo = model_size == "olmo"
    is_crfm = model_size.startswith("crfm")
    is_smollm3 = model_size == "smollm3"
    is_qwen = model_size == "qwen"
    if is_olmo:
        from utils_model_olmo import load_olmo_with_checkpoint
        model, tokenizer = load_olmo_with_checkpoint(checkpoint, DEVICE)
        gen_fn = lambda tmpl, mxt: generate_olmo(model, tokenizer, tmpl, mxt)
    elif is_crfm:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformer_lens import HookedTransformer
        seed = int(model_size[4:]) if len(model_size) > 4 else 1
        model_id = CRFM_SEED_MODEL_IDS[seed]
        print(f"  Loading {model_id} @ {checkpoint} on {DEVICE} ...")
        _tok = AutoTokenizer.from_pretrained("gpt2")
        _tok.pad_token = _tok.eos_token
        _hf = AutoModelForCausalLM.from_pretrained(
            model_id, revision=checkpoint, dtype=torch.float32,
        )
        model = HookedTransformer.from_pretrained(
            "gpt2", hf_model=_hf, tokenizer=_tok, device=DEVICE,
        )
        gen_fn = lambda tmpl, mxt: generate_pythia(model, tmpl, mxt)
    elif is_smollm3:
        from utils_model_smollm3 import load_smollm3_with_checkpoint
        model, tokenizer = load_smollm3_with_checkpoint(checkpoint, DEVICE)
        gen_fn = lambda tmpl, mxt: generate_olmo(model, tokenizer, tmpl, mxt)
    elif is_qwen:
        from utils_model_qwen import load_qwen
        model, tokenizer = load_qwen(DEVICE)
        gen_fn = lambda tmpl, mxt: generate_olmo(model, tokenizer, tmpl, mxt)
    else:
        from utils_model import load_pythia_with_checkpoint
        model = load_pythia_with_checkpoint(model_size, checkpoint, DEVICE)
        gen_fn = lambda tmpl, mxt: generate_pythia(model, tmpl, mxt)

    gen_prompts = [json.loads(l) for l in open(PROMPTS_FILE)
                   if json.loads(l)["task"] == "generation"]
    n_terms = len(set(p["term"] for p in gen_prompts))
    print(f"  {len(gen_prompts)} generation prompts / {n_terms} terms")

    # Warn about terms without exemplars
    terms_no_exemplar = set(p["term"] for p in gen_prompts) - set(FEW_SHOT_EXAMPLES)
    if terms_no_exemplar:
        print(f"  ⚠  No exemplar for: {sorted(terms_no_exemplar)} — will run ZS only")

    zs_scores, fs_scores = {}, {}
    zs_results, fs_results = [], []

    for p in tqdm(gen_prompts, desc=f"{model_size}/{checkpoint}"):
        term   = p["term"]
        mxt    = p.get("max_tokens", 25)
        prefix = FEW_SHOT_EXAMPLES.get(term, "")

        zs_comp = gen_fn(p["template"], mxt)
        fs_comp = gen_fn(prefix + p["template"], mxt) if prefix else zs_comp

        zs_s = score_generation(zs_comp, term)
        fs_s = score_generation(fs_comp, term)

        zs_scores.setdefault(term, []).append(zs_s)
        fs_scores.setdefault(term, []).append(fs_s)
        zs_results.append({"term": term, "prompt_id": p["prompt_id"],
                           "score": zs_s, "completion": zs_comp})
        fs_results.append({"term": term, "prompt_id": p["prompt_id"],
                           "score": fs_s, "completion": fs_comp})

    zs_mean = sum(r["score"] for r in zs_results) / len(zs_results)
    fs_mean = sum(r["score"] for r in fs_results) / len(fs_results)
    imp_pp  = (fs_mean - zs_mean) * 100
    rel_pct = imp_pp / zs_mean * 100 if zs_mean > 0 else float("inf")

    print(f"\n  Zero-shot mean:  {zs_mean:.4f}")
    print(f"  Few-shot mean:   {fs_mean:.4f}")
    print(f"  Improvement:     {imp_pp:+.1f} pp  ({rel_pct:+.0f}% relative)")
    rep = "✅ C3 SUPPORTED" if imp_pp > 20 else ("⚠ WEAK" if imp_pp > 5 else "❌")
    print(f"  → {rep}")

    print(f"\n  Per-term:")
    print(f"  {'Term':<28}  {'ZS':>6}  {'FS':>6}  {'Δpp':>7}  Exemplar?")
    all_terms = sorted(zs_scores.keys())
    for t in all_terms:
        m_zs = sum(zs_scores[t]) / len(zs_scores[t])
        m_fs = sum(fs_scores[t]) / len(fs_scores[t])
        has_ex = "✓" if t in FEW_SHOT_EXAMPLES else "—"
        print(f"  {t:<28}  {m_zs:>6.3f}  {m_fs:>6.3f}  {(m_fs-m_zs)*100:>+6.1f}  {has_ex}")

    result = {
        "model": model_size, "checkpoint": checkpoint,
        "n_prompts": len(gen_prompts), "n_terms": n_terms,
        "zero_shot_mean": round(zs_mean, 4),
        "few_shot_mean":  round(fs_mean,  4),
        "improvement_pp": round(imp_pp, 2),
        "improvement_relative_pct": round(rel_pct, 1),
        "zero_shot_per_term": {t: round(sum(v)/len(v), 4) for t, v in zs_scores.items()},
        "few_shot_per_term":  {t: round(sum(v)/len(v), 4) for t, v in fs_scores.items()},
    }
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved → {out_file}")

    del model
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    evaluate_checkpoint(args.model, args.checkpoint)
