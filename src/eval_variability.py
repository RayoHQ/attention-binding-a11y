"""Evaluate generation score variability across temperature and seed settings.

Tests robustness of generation scores (C1, C3 findings) across decoding parameters.
Runs generation-only evaluation on a selected subset of checkpoints with:
- Temperatures: 0.0 (greedy), 0.3 (low sampling), 0.7 (moderate sampling)
- Seeds: 42, 123, 456, 789, 1024 (5 replicates per temperature)

Recognition scoring is deterministic (log-prob ranking) and unaffected by these parameters.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils_model import load_pythia_with_checkpoint
from eval_behavior import run_behavioral_probe

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("data/results/variability")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Experiment matrix
TEMPERATURES = [0.0, 0.3, 0.7]
SEEDS = [42, 123, 456, 789, 1024]

# Selected checkpoints for variability testing
# Choose representative points from each model's trajectory
TEST_CHECKPOINTS = [
    ("160m", "step15000"),   # Early high EB*, low behavior
    ("160m", "step120000"),  # Peak performance
    ("1b", "step15000"),     # Decoupling onset
    ("1b", "step143000"),    # Convergence
    ("2.8b", "step15000"),   # Early high performance
    ("2.8b", "step143000"),  # Convergence
]


def load_prompts(prompt_file: str):
    """Load prompts from JSONL."""
    prompts = []
    with open(prompt_file) as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


def run_variability_experiment(
    model_size: str,
    checkpoint: str,
    prompt_file: str = "data/prompts/pilot_terms.jsonl",
):
    """Run variability experiment for one model-checkpoint combination."""
    print(f"\n{'='*60}")
    print(f"Variability Experiment: {model_size} {checkpoint}")
    print(f"{'='*60}")
    
    # Load model
    model = load_pythia_with_checkpoint(model_size, checkpoint, DEVICE)
    
    # Load prompts (generation only)
    all_prompts = load_prompts(prompt_file)
    gen_prompts = [p for p in all_prompts if p["task"] == "generation"]
    print(f"Loaded {len(gen_prompts)} generation prompts")
    
    # Run experiments across temperature × seed matrix
    all_results = []
    
    for temp in TEMPERATURES:
        for seed in SEEDS:
            print(f"\n--- Temperature={temp}, Seed={seed} ---")
            
            for prompt in tqdm(gen_prompts, desc=f"T={temp} S={seed}"):
                result = run_behavioral_probe(
                    model, prompt, DEVICE, temperature=temp, seed=seed
                )
                
                record = {
                    "model": f"pythia-{model_size}-deduped",
                    "checkpoint": checkpoint,
                    "term": prompt["term"],
                    "prompt_id": prompt["prompt_id"],
                    "temperature": temp,
                    "seed": seed,
                    "score": result["score"],
                    "text_out": result["text_out"],
                    "is_correct": result["is_correct"],
                }
                all_results.append(record)
    
    # Save raw results
    output_file = OUTPUT_DIR / f"{model_size}_{checkpoint}_variability_raw.jsonl"
    with open(output_file, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"\n✅ Saved {len(all_results)} raw results to {output_file}")
    
    # Compute aggregated statistics
    stats = compute_statistics(all_results)
    
    # Save statistics
    stats_file = OUTPUT_DIR / f"{model_size}_{checkpoint}_variability_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Saved statistics to {stats_file}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return stats


def compute_statistics(results):
    """Compute mean, std, min, max per (term, temperature) and overall."""
    from collections import defaultdict
    
    # Group by (term, temperature)
    groups = defaultdict(list)
    for r in results:
        key = (r["term"], r["temperature"])
        groups[key].append(r["score"])
    
    # Per-group stats
    per_group = {}
    for (term, temp), scores in groups.items():
        per_group[f"{term}_T{temp}"] = {
            "mean": round(float(np.mean(scores)), 4),
            "std": round(float(np.std(scores)), 4),
            "min": round(float(np.min(scores)), 4),
            "max": round(float(np.max(scores)), 4),
            "n": len(scores),
        }
    
    # Overall stats per temperature
    temp_groups = defaultdict(list)
    for r in results:
        temp_groups[r["temperature"]].append(r["score"])
    
    per_temp = {}
    for temp, scores in temp_groups.items():
        per_temp[f"T{temp}"] = {
            "mean": round(float(np.mean(scores)), 4),
            "std": round(float(np.std(scores)), 4),
            "n": len(scores),
        }
    
    # Overall stats (all temps, all seeds)
    all_scores = [r["score"] for r in results]
    overall = {
        "mean": round(float(np.mean(all_scores)), 4),
        "std": round(float(np.std(all_scores)), 4),
        "min": round(float(np.min(all_scores)), 4),
        "max": round(float(np.max(all_scores)), 4),
        "n": len(all_scores),
    }
    
    return {
        "per_group": per_group,
        "per_temperature": per_temp,
        "overall": overall,
    }


def main():
    """Run variability experiments on selected checkpoints."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run temperature/seed variability experiments")
    parser.add_argument("--model", type=str, default=None, help="Specific model (e.g., 160m)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint (e.g., step120000)")
    parser.add_argument("--prompts", type=str, default="data/prompts/pilot_terms.jsonl", help="Prompts file")
    args = parser.parse_args()
    
    if args.model and args.checkpoint:
        # Single experiment
        run_variability_experiment(args.model, args.checkpoint, args.prompts)
    else:
        # Run all predefined test checkpoints
        print(f"Running variability experiments on {len(TEST_CHECKPOINTS)} checkpoints")
        print(f"Matrix: {len(TEMPERATURES)} temps × {len(SEEDS)} seeds = {len(TEMPERATURES) * len(SEEDS)} runs per prompt")
        
        all_stats = {}
        for model_size, checkpoint in TEST_CHECKPOINTS:
            try:
                stats = run_variability_experiment(model_size, checkpoint, args.prompts)
                all_stats[f"{model_size}_{checkpoint}"] = stats
            except Exception as e:
                print(f"❌ Error on {model_size} {checkpoint}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save combined summary
        summary_file = OUTPUT_DIR / "variability_summary.json"
        with open(summary_file, "w") as f:
            json.dump(all_stats, f, indent=2)
        print(f"\n✅ Saved combined summary to {summary_file}")
        
        print(f"\n{'='*60}")
        print("✅ Variability experiments complete!")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
