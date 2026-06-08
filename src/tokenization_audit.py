"""Tokenization audit: Verify spans are well-defined across all models.

Covers all 45 terms (9 Set-B + 21 tier123 + 12 wave-2).
Models: Pythia, OLMo, CRFM GPT-2 Small, SmolLM3-3B, Qwen2.5-1.5B.
"""

import csv
import json
from pathlib import Path

from transformers import AutoTokenizer

# Config — Pythia
MODEL_SIZES = ["160m", "1b", "2.8b"]
CHECKPOINT = "step143000"  # Use final checkpoint (tokenization stable across steps)
OUTPUT_DIR = Path("data/tokenization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Current 9 terms (pilot + expanded)
TERMS_CURRENT = [
    "screen reader", "skip link", "alt text",
    "aria attribute", "color contrast", "focus indicator",
    "form validation", "heading structure", "tab order",
]

# 21 new terms — Tier 1 (AT hardware/concepts)
TERMS_TIER1 = [
    "braille display", "screen magnifier", "voice control",
    "switch access", "audio description", "captions closed",
    "cognitive load", "high contrast", "keyboard shortcut", "text resize",
]

# 21 new terms — Tier 2 (WCAG 2.2 success criteria)
TERMS_TIER2 = [
    "keyboard navigation", "focus management", "skip navigation",
    "reflow content", "non-text content", "error identification",
    "input purpose", "text spacing",
]

# 21 new terms — Tier 3 (WAI-ARIA 1.2 roles/properties)
TERMS_TIER3 = [
    "live region", "alert dialog", "tree grid",
]

# Wave-2 terms (12 new accessibility terms)
TERMS_WAVE2 = [
    "contrast ratio", "eye tracking", "focus trap", "haptic feedback",
    "motion sensitivity", "orientation support", "plain language",
    "reduced motion", "semantic html", "sign language",
    "time limits", "touch target size",
]

ALL_TERMS = TERMS_CURRENT + TERMS_TIER1 + TERMS_TIER2 + TERMS_TIER3 + TERMS_WAVE2

# Model tokenizers for full cross-model audit
OLMO_MODEL = "allenai/OLMo-1B-hf"
CRFM_MODEL = "gpt2"  # CRFM GPT-2 Small uses identical tokenizer to OpenAI GPT-2
SMOLLM3_MODEL = "HuggingFaceTB/SmolLM3-3B"
QWEN25_MODEL = "Qwen/Qwen2.5-1.5B"

NEW_MODEL_LIST = [
    ("crfm-gpt2-sm", CRFM_MODEL),
    ("smollm3-3b",   SMOLLM3_MODEL),
    ("qwen2.5-1.5b", QWEN25_MODEL),
]


def get_span_indices(tokenizer, text: str) -> tuple[list[int], list[str]]:
    """Get token IDs and strings for a text span."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    token_strings = [tokenizer.decode([t]) for t in tokens]
    return tokens, token_strings


def _audit_one_model(model_name: str, tokenizer, terms: list, tag: str) -> list[dict]:
    """Run audit for one tokenizer over a list of terms. Returns result rows."""
    results = []
    print(f"\n=== {model_name} ===")
    for term in terms:
        tokens, token_strings = get_span_indices(tokenizer, term)
        is_clean = all(not s.startswith("##") for s in token_strings)
        result = {
            "model": tag,
            "term": term,
            "tier": (
                "current" if term in TERMS_CURRENT else
                "tier1" if term in TERMS_TIER1 else
                "tier2" if term in TERMS_TIER2 else
                "tier3"
            ),
            "n_tokens": len(tokens),
            "token_ids": json.dumps(tokens),
            "token_strings": json.dumps(token_strings),
            "is_clean": is_clean,
            "valid_for_binding": len(tokens) >= 2,
        }
        results.append(result)
        flag = "✅" if result["valid_for_binding"] and is_clean else "⚠️ "
        print(f"  {flag} {term}: {len(tokens)} tokens → {token_strings}")
    return results


def audit_tokenization() -> list[dict]:
    """Run tokenization audit for all 45 terms across Pythia 160m/1b/2.8b."""
    results = []
    for size in MODEL_SIZES:
        model_name = f"EleutherAI/pythia-{size}-deduped"
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=CHECKPOINT)
        results.extend(_audit_one_model(model_name, tokenizer, ALL_TERMS, tag=f"pythia-{size}"))

    output_file = OUTPUT_DIR / "tokenization_table_45terms.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✅ Saved to {output_file}")
    return results


def audit_olmo_tokenization() -> list[dict]:
    """Run tokenization audit for all 45 terms under OLMo tokenizer."""
    print(f"\nLoading OLMo tokenizer: {OLMO_MODEL}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(OLMO_MODEL)
    except Exception as e:
        print(f"⚠️  Could not load OLMo tokenizer: {e}")
        return []

    results = _audit_one_model(OLMO_MODEL, tokenizer, ALL_TERMS, tag="olmo-1b")

    output_file = OUTPUT_DIR / "tokenization_olmo_45terms.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✅ Saved to {output_file}")
    return results


def audit_new_model_tokenization() -> list[dict]:
    """Run tokenization audit for all 45 terms under CRFM GPT-2, SmolLM3, Qwen2.5."""
    all_results = []
    for tag, model_name in NEW_MODEL_LIST:
        print(f"\nLoading tokenizer: {model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"⚠️  Could not load {model_name}: {e}")
            continue
        results = _audit_one_model(model_name, tokenizer, ALL_TERMS, tag=tag)
        all_results.extend(results)

    if all_results:
        output_file = OUTPUT_DIR / "tokenization_new_models_45terms.csv"
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n✅ Saved to {output_file}")
    return all_results


def validate_for_binding(results: list[dict]) -> bool:
    """Check if all terms are valid for binding analysis."""
    issues = []
    for r in results:
        if not r["valid_for_binding"]:
            issues.append(f"{r['model']}/{r['term']}: only {r['n_tokens']} token(s) — exclude")
        if not r["is_clean"]:
            issues.append(f"{r['model']}/{r['term']}: subword boundary issue")
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    print("\n✅ All terms valid for binding analysis")
    return True


def summarize_results(results: list[dict]):
    """Print per-tier summary table."""
    print("\n=== Token count summary (first model only) ===")
    print(f"{'Tier':<10} {'Term':<25} {'n_tokens':<10} {'Valid'}")
    print("-" * 55)
    seen = set()
    for r in results:
        key = (r["tier"], r["term"])
        if key not in seen:
            seen.add(key)
            valid = "✅" if r["valid_for_binding"] and r["is_clean"] else "❌"
            print(f"{r['tier']:<10} {r['term']:<25} {r['n_tokens']:<10} {valid}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--olmo-only",      action="store_true", help="Run only OLMo audit")
    parser.add_argument("--pythia-only",    action="store_true", help="Run only Pythia audit")
    parser.add_argument("--new-models-only", action="store_true", help="Run CRFM/SmolLM3/Qwen2.5 audit")
    args = parser.parse_args()

    all_results = []

    if not args.olmo_only and not args.new_models_only:
        print("\n--- Pythia 45-term tokenization audit ---")
        pythia_results = audit_tokenization()
        all_results.extend(pythia_results)
        summarize_results(pythia_results)

    if not args.pythia_only and not args.new_models_only:
        print("\n--- OLMo 45-term tokenization audit ---")
        olmo_results = audit_olmo_tokenization()
        if olmo_results:
            all_results.extend(olmo_results)

    if not args.olmo_only and not args.pythia_only:
        print("\n--- New models (CRFM/SmolLM3/Qwen2.5) 45-term tokenization audit ---")
        new_results = audit_new_model_tokenization()
        all_results.extend(new_results)

    if all_results:
        is_valid = validate_for_binding([r for r in all_results if r["model"].startswith("pythia-160m")])
        print("\n=== INVALID TERMS PER MODEL ===")
        for r in all_results:
            if not r["valid_for_binding"]:
                print(f"  EXCLUDE from {r['model']}: '{r['term']}' → {r['n_tokens']} token(s)")
        exit(0 if is_valid else 1)
