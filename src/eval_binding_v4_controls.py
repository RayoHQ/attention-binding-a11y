"""Extract binding scores for V4 wrong-domain semantic near-miss controls.

Tests whether EB* discriminates between:
- Real terms: "alt text" (accessibility concept)
- Wrong-domain near-misses: "alt function" (programming concept)

Directly addresses reviewer concern with exact example.
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
OUTPUT_DIR = Path("data/results/binding_v4_controls")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Test on trained checkpoints where we expect best discrimination
TEST_CHECKPOINTS = [
    ("160m", "step120000"),  # Trained
    ("1b", "step143000"),    # Trained
]


def load_controls(control_file: str):
    """Load V4 control terms."""
    controls = []
    with open(control_file) as f:
        for line in f:
            controls.append(json.loads(line))
    return controls


def extract_v4_binding(model_size: str, checkpoint: str):
    """Extract binding metrics for V4 wrong-domain controls."""
    print(f"\n{'='*60}")
    print(f"V4 Wrong-Domain Controls: {model_size} {checkpoint}")
    print(f"{'='*60}")
    
    # Load model
    model = load_pythia_with_checkpoint(model_size, checkpoint, DEVICE)
    tokenizer = model.tokenizer
    
    # Load V4 controls
    controls = load_controls("data/prompts/control_terms_v4.jsonl")
    print(f"Loaded {len(controls)} V4 wrong-domain controls")
    
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
                "wrong_domain": ctrl["wrong_domain"],
                "prompt_id": ctrl["prompt_id"],
                **binding,
            }
            results.append(record)
            
        except Exception as e:
            print(f"Error on {ctrl['term']}: {e}")
    
    # Save results
    output_file = OUTPUT_DIR / f"{model_size}_{checkpoint}_v4_controls.jsonl"
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    print(f"\n✅ Saved {len(results)} results to {output_file}")
    
    # Compute summary stats
    eb_star_scores = [r["eb_star"] for r in results if r.get("eb_star") is not None]
    if eb_star_scores:
        print(f"\nV4 Wrong-Domain Controls Summary:")
        print(f"  Mean EB*: {np.mean(eb_star_scores):.3f}")
        print(f"  Std EB*:  {np.std(eb_star_scores):.3f}")
        print(f"  Min EB*:  {np.min(eb_star_scores):.3f}")
        print(f"  Max EB*:  {np.max(eb_star_scores):.3f}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return results


def main():
    """Run V4 control experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract binding for V4 wrong-domain controls")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    
    if args.model and args.checkpoint:
        # Single experiment
        extract_v4_binding(args.model, args.checkpoint)
    else:
        # Run all test checkpoints
        print(f"Running V4 wrong-domain control experiments on {len(TEST_CHECKPOINTS)} checkpoints")
        
        all_results = {}
        for model_size, checkpoint in TEST_CHECKPOINTS:
            try:
                results = extract_v4_binding(model_size, checkpoint)
                all_results[f"{model_size}_{checkpoint}"] = results
            except Exception as e:
                print(f"❌ Error on {model_size} {checkpoint}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*60}")
        print("✅ V4 wrong-domain control experiments complete!")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
