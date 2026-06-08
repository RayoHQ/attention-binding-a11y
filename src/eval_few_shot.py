"""Evaluate few-shot unlockability for C3 with saved results."""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils_model import load_pythia_with_checkpoint
from eval_behavior import run_behavioral_probe

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("data/results/few_shot")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# One worked example per term (used as few-shot prefix)
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
    print(f"\n{'=' * 60}")
    print(f"Evaluating {model_size} {checkpoint}")
    print("=" * 60)

    model = load_pythia_with_checkpoint(model_size, checkpoint, DEVICE)

    with open("data/prompts/pilot_terms.jsonl") as f:
        all_prompts = [json.loads(line) for line in f]

    gen_prompts = [p for p in all_prompts if p["task"] == "generation"]
    print(f"Loaded {len(gen_prompts)} generation prompts\n")

    # --- Zero-shot ---
    print("--- Zero-shot (baseline) ---")
    zero_shot_results = []
    for prompt in gen_prompts:
        result = run_behavioral_probe(model, prompt, DEVICE)
        result["term"] = prompt["term"]
        result["prompt_id"] = prompt["prompt_id"]
        result["template"] = prompt["template"]
        zero_shot_results.append(result)
        print(f"  {prompt['term']:15s} {prompt['prompt_id']}: score={result['score']:.4f}")

    zero_shot_scores = [r["score"] for r in zero_shot_results]
    zero_shot_mean = sum(zero_shot_scores) / len(zero_shot_scores)
    print(f"Zero-shot mean: {zero_shot_mean:.4f}")

    # --- Few-shot ---
    print("\n--- Few-shot (one-shot) ---")
    few_shot_results = []
    for prompt in gen_prompts:
        fs_prompt = create_few_shot_prompt(prompt)
        result = run_behavioral_probe(model, fs_prompt, DEVICE)
        result["term"] = prompt["term"]
        result["prompt_id"] = fs_prompt["prompt_id"]
        result["template"] = fs_prompt["template"]
        few_shot_results.append(result)
        print(f"  {prompt['term']:15s} {fs_prompt['prompt_id']}: score={result['score']:.4f}")

    few_shot_scores = [r["score"] for r in few_shot_results]
    few_shot_mean = sum(few_shot_scores) / len(few_shot_scores)
    print(f"Few-shot mean:  {few_shot_mean:.4f}")

    # --- Summary ---
    improvement = few_shot_mean - zero_shot_mean
    relative = (improvement / zero_shot_mean * 100) if zero_shot_mean > 0 else float("inf")

    print(f"\n=== Results ===")
    print(f"Zero-shot: {zero_shot_mean:.4f}")
    print(f"Few-shot:  {few_shot_mean:.4f}")
    print(f"Improvement: +{improvement:.4f} ({relative:.1f}% relative)")
    print(f"Improvement: +{improvement * 100:.1f} percentage points")

    # --- Save ---
    output = {
        "model": model_size,
        "checkpoint": checkpoint,
        "zero_shot_mean": round(zero_shot_mean, 4),
        "few_shot_mean": round(few_shot_mean, 4),
        "improvement_pp": round(improvement * 100, 1),
        "improvement_relative_pct": round(relative, 1),
        "zero_shot_details": zero_shot_results,
        "few_shot_details": few_shot_results,
    }

    output_file = OUTPUT_DIR / f"{model_size}_{checkpoint}_few_shot.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"Saved to {output_file}")

    del model
    torch.cuda.empty_cache()

    return output


def main():
    # Checkpoints where EB* > 0.6 but behavior is mid/low
    test_conditions = [
        ("160m", "step15000"),   # EB*=0.644, gen=0.333
        ("160m", "step30000"),   # EB*=0.642, gen=0.667
        ("1b", "step15000"),     # EB*=0.646, gen=0.556
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
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':8s} {'Checkpoint':12s} {'Zero-shot':>10s} {'Few-shot':>10s} {'Δ (pp)':>8s}")
    print("-" * 50)
    for r in all_results:
        print(
            f"{r['model']:8s} {r['checkpoint']:12s} "
            f"{r['zero_shot_mean']:10.4f} {r['few_shot_mean']:10.4f} "
            f"{r['improvement_pp']:+8.1f}"
        )


if __name__ == "__main__":
    main()
