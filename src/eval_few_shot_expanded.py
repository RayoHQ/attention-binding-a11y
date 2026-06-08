"""Evaluate few-shot unlockability on EXPANDED 99-prompt dataset."""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils_model import load_pythia_with_checkpoint
from eval_behavior import run_behavioral_probe

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("data/results/few_shot_expanded")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Few-shot examples for all 9 terms
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
        "(H1, H2, H3, etc.) that allows screen reader users to understand page "
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
        "webpage (like navigation, main content, or footer) that helps screen "
        "reader users quickly navigate to different page areas.\n\n"
    ),
}


def create_few_shot_prompt(original_prompt: dict) -> dict:
    """Prepend one-shot example to a generation prompt."""
    term = original_prompt["term"]
    prefix = FEW_SHOT_EXAMPLES.get(term, "")
    modified = original_prompt.copy()
    modified["template"] = prefix + original_prompt["template"]
    modified["prompt_id"] = original_prompt["prompt_id"] + "_fs"
    return modified


def evaluate_checkpoint(model_size: str, checkpoint: str):
    """Run zero-shot and few-shot generation evaluation on one checkpoint."""
    print(f"\n{'=' * 70}")
    print(f"Evaluating {model_size} {checkpoint} on EXPANDED dataset")
    print("=" * 70)

    model = load_pythia_with_checkpoint(model_size, checkpoint, DEVICE)

    # Load EXPANDED prompts
    with open("data/prompts/expanded_terms_100.jsonl") as f:
        all_prompts = [json.loads(line) for line in f]

    gen_prompts = [p for p in all_prompts if p["task"] == "generation"]
    print(f"Loaded {len(gen_prompts)} generation prompts from expanded dataset")
    print(f"Terms: {len(set(p['term'] for p in gen_prompts))}")

    # --- Zero-shot ---
    print("\n--- Zero-shot (baseline) ---")
    zero_shot_results = []
    for i, prompt in enumerate(gen_prompts):
        result = run_behavioral_probe(model, prompt, DEVICE)
        result["term"] = prompt["term"]
        result["prompt_id"] = prompt["prompt_id"]
        result["template"] = prompt["template"]
        zero_shot_results.append(result)
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{len(gen_prompts)} prompts...")

    zero_shot_scores = [r["score"] for r in zero_shot_results]
    zero_shot_mean = sum(zero_shot_scores) / len(zero_shot_scores)
    print(f"\nZero-shot mean: {zero_shot_mean:.4f}")

    # Per-term breakdown
    print("\nZero-shot per-term:")
    terms_dict = {}
    for r in zero_shot_results:
        term = r["term"]
        if term not in terms_dict:
            terms_dict[term] = []
        terms_dict[term].append(r["score"])
    
    for term in sorted(terms_dict.keys()):
        scores = terms_dict[term]
        mean_score = sum(scores) / len(scores)
        print(f"  {term:20s}: {mean_score:.4f} (n={len(scores)})")

    # --- Few-shot ---
    print("\n--- Few-shot (one-shot) ---")
    few_shot_results = []
    for i, prompt in enumerate(gen_prompts):
        fs_prompt = create_few_shot_prompt(prompt)
        result = run_behavioral_probe(model, fs_prompt, DEVICE)
        result["term"] = prompt["term"]
        result["prompt_id"] = fs_prompt["prompt_id"]
        result["template"] = fs_prompt["template"]
        few_shot_results.append(result)
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{len(gen_prompts)} prompts...")

    few_shot_scores = [r["score"] for r in few_shot_results]
    few_shot_mean = sum(few_shot_scores) / len(few_shot_scores)
    print(f"\nFew-shot mean:  {few_shot_mean:.4f}")

    # Per-term breakdown
    print("\nFew-shot per-term:")
    fs_terms_dict = {}
    for r in few_shot_results:
        term = r["term"]
        if term not in fs_terms_dict:
            fs_terms_dict[term] = []
        fs_terms_dict[term].append(r["score"])
    
    for term in sorted(fs_terms_dict.keys()):
        scores = fs_terms_dict[term]
        mean_score = sum(scores) / len(scores)
        print(f"  {term:20s}: {mean_score:.4f} (n={len(scores)})")

    # --- Summary ---
    improvement = few_shot_mean - zero_shot_mean
    relative = (improvement / zero_shot_mean * 100) if zero_shot_mean > 0 else float("inf")

    print(f"\n=== RESULTS ===")
    print(f"Zero-shot: {zero_shot_mean:.4f}")
    print(f"Few-shot:  {few_shot_mean:.4f}")
    print(f"Improvement: +{improvement:.4f} ({relative:.1f}% relative)")
    print(f"Improvement: +{improvement * 100:.1f} percentage points")

    # --- Save ---
    output = {
        "model": model_size,
        "checkpoint": checkpoint,
        "dataset": "expanded_99_prompts",
        "n_prompts": len(gen_prompts),
        "n_terms": len(set(p['term'] for p in gen_prompts)),
        "zero_shot_mean": round(zero_shot_mean, 4),
        "few_shot_mean": round(few_shot_mean, 4),
        "improvement_pp": round(improvement * 100, 1),
        "improvement_relative_pct": round(relative, 1),
        "zero_shot_per_term": {term: round(sum(scores)/len(scores), 4) 
                               for term, scores in terms_dict.items()},
        "few_shot_per_term": {term: round(sum(scores)/len(scores), 4) 
                             for term, scores in fs_terms_dict.items()},
        "zero_shot_details": zero_shot_results,
        "few_shot_details": few_shot_results,
    }

    output_file = OUTPUT_DIR / f"{model_size}_{checkpoint}_expanded_few_shot.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved to {output_file}")

    del model
    torch.cuda.empty_cache()

    return output


def main():
    # Same checkpoints as original few-shot experiment
    test_conditions = [
        ("160m", "step15000"),   # Original: EB*=0.644, gen=0.333
        ("160m", "step30000"),   # Original: EB*=0.642, gen=0.667
        ("1b", "step15000"),     # Original: EB*=0.646, gen=0.556
    ]

    all_results = []
    for model_size, checkpoint in test_conditions:
        try:
            result = evaluate_checkpoint(model_size, checkpoint)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR on {model_size} {checkpoint}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY - EXPANDED DATASET FEW-SHOT VALIDATION")
    print("=" * 70)
    print(f"{'Model':8s} {'Checkpoint':12s} {'Zero-shot':>10s} {'Few-shot':>10s} {'Δ (pp)':>8s} {'Relative':>10s}")
    print("-" * 70)
    for r in all_results:
        print(
            f"{r['model']:8s} {r['checkpoint']:12s} "
            f"{r['zero_shot_mean']:10.4f} {r['few_shot_mean']:10.4f} "
            f"{r['improvement_pp']:+8.1f} {r['improvement_relative_pct']:+10.1f}%"
        )

    print(f"\nDataset: {all_results[0]['n_prompts']} generation prompts across {all_results[0]['n_terms']} terms")


if __name__ == "__main__":
    main()
