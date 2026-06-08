"""Extract binding scores for V3 domain-adjacent near-miss controls.

Tests whether EB* discriminates between:
- Real terms: "alt text", "screen reader"
- Domain-adjacent near-misses: "alt function", "screen editor"

Addresses reviewer concern: "alt function" vs "alt text" discrimination.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils_model import load_pythia_with_checkpoint
from extract_attention import extract_binding_for_prompt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("data/results/binding_v3_controls")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Test on representative checkpoints
TEST_CHECKPOINTS = [
    ("160m", "step15000"),   # Early high EB*
    ("160m", "step120000"),  # Trained
    ("1b", "step15000"),     # Early
    ("1b", "step143000"),    # Trained
]


def load_controls(control_file: str):
    """Load V3 control terms."""
    controls = []
    with open(control_file) as f:
        for line in f:
            controls.append(json.loads(line))
    return controls


def extract_v3_binding(model_size: str, checkpoint: str):
    """Extract binding metrics for V3 controls."""
    print(f"\n{'='*60}")
    print(f"V3 Controls: {model_size} {checkpoint}")
    print(f"{'='*60}")
    
    # Load model
    model = load_pythia_with_checkpoint(model_size, checkpoint, DEVICE)
    tokenizer = model.tokenizer
    
    # Load V3 controls
    controls = load_controls("data/prompts/control_terms_v3.jsonl")
    print(f"Loaded {len(controls)} V3 domain-adjacent controls")
    
    results = []
    
    for ctrl in tqdm(controls, desc="Extracting binding"):
        try:
            # Extract binding metrics
            binding = extract_binding_for_prompt(
                model=model,
                prompt_text=ctrl["template"],
                term=ctrl["term"],
                tokenizer=tokenizer,
            )
            
            record = {
                "model": f"pythia-{model_size}-deduped",
                "checkpoint": checkpoint,
                "term": ctrl["term"],
                "control_group": ctrl["control_group"],
                "source_term": ctrl["source_term"],
                "overlap_token": ctrl["overlap"],
                "prompt_id": ctrl["prompt_id"],
                **binding,
            }
            results.append(record)
            
        except Exception as e:
            print(f"Error on {ctrl['term']}: {e}")
    
    # Save results
    output_file = OUTPUT_DIR / f"{model_size}_{checkpoint}_v3_controls.jsonl"
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    print(f"\n✅ Saved {len(results)} results to {output_file}")
    
    # Compute summary stats
    eb_star_scores = [r["eb_star"] for r in results if r.get("eb_star") is not None]
    if eb_star_scores:
        print(f"\nV3 Controls Summary:")
        print(f"  Mean EB*: {np.mean(eb_star_scores):.3f}")
        print(f"  Std EB*:  {np.std(eb_star_scores):.3f}")
        print(f"  Min EB*:  {np.min(eb_star_scores):.3f}")
        print(f"  Max EB*:  {np.max(eb_star_scores):.3f}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return results


def main():
    """Run V3 control experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract binding for V3 controls")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    
    if args.model and args.checkpoint:
        # Single experiment
        extract_v3_binding(args.model, args.checkpoint)
    else:
        # Run all test checkpoints
        print(f"Running V3 control experiments on {len(TEST_CHECKPOINTS)} checkpoints")
        
        all_results = {}
        for model_size, checkpoint in TEST_CHECKPOINTS:
            try:
                results = extract_v3_binding(model_size, checkpoint)
                all_results[f"{model_size}_{checkpoint}"] = results
            except Exception as e:
                print(f"❌ Error on {model_size} {checkpoint}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*60}")
        print("✅ V3 control experiments complete!")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
