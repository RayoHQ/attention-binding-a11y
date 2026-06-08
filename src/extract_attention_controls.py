"""Extract binding metrics for control terms across all checkpoints.

This script runs binding extraction (no behavioral evaluation) for control
terms to test whether EB* distinguishes real accessibility concepts from:
- Backwards shuffles (e.g., "reader screen")
- Cross-term swaps (e.g., "screen text")
- Semantic field controls (e.g., "keyboard mouse")
- Frequency-matched bigrams (e.g., "web page")
- Random unrelated bigrams (e.g., "blue carpet")
"""

import json
import os
import sys
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils_model import load_pythia_with_checkpoint
from extract_attention import extract_binding_for_prompt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONTROL_PROMPTS = Path("data/prompts/control_terms_v2.jsonl")
OUTPUT_DIR = Path("data/results/binding_controls_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model configurations
MODELS = [
    ("160m", ["step0", "step15000", "step30000", "step60000", "step90000", "step120000", "step140000", "step143000"]),
    ("1b", ["step0", "step15000", "step30000", "step60000", "step90000", "step120000", "step140000", "step143000"]),
    ("2.8b", ["step0", "step15000", "step30000", "step60000", "step90000", "step120000", "step140000", "step143000"]),
]


def load_control_prompts() -> List[dict]:
    """Load control term prompts from JSONL."""
    prompts = []
    with open(CONTROL_PROMPTS) as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


def extract_controls_for_checkpoint(model_size: str, checkpoint: str):
    """Extract binding for all control terms at one checkpoint."""
    print(f"\n{'='*60}")
    print(f"Processing {model_size} {checkpoint}")
    print(f"{'='*60}")
    
    # Load model
    model = load_pythia_with_checkpoint(model_size, checkpoint, DEVICE)
    tokenizer = model.tokenizer
    
    # Load control prompts
    prompts = load_control_prompts()
    
    # Process each control term
    results = []
    for prompt in tqdm(prompts, desc=f"Extracting {model_size}/{checkpoint}"):
        binding = extract_binding_for_prompt(
            model=model,
            prompt_text=prompt["template"],
            term=prompt["term"],
            tokenizer=tokenizer,
        )
        
        result = {
            "model": f"pythia-{model_size}-deduped",
            "checkpoint": checkpoint,
            "term": prompt["term"],
            "control_group": prompt["control_group"],
            "prompt_id": prompt["prompt_id"],
            "prompt_template": prompt["template"],
            **binding,
        }
        
        # Add source term metadata
        if "source_term" in prompt:
            result["source_term"] = prompt["source_term"]
        if "source_terms" in prompt:
            result["source_terms"] = prompt["source_terms"]
        if "source_domain" in prompt:
            result["source_domain"] = prompt["source_domain"]
        
        results.append(result)
    
    # Save results
    output_file = OUTPUT_DIR / f"{model_size}_{checkpoint}_controls.jsonl"
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    print(f"✅ Saved {len(results)} control results to {output_file}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return output_file


def main():
    """Run control binding extraction for all models and checkpoints."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract binding for control terms")
    parser.add_argument("--model", type=str, default=None, help="Specific model size (160m, 1b, 2.8b)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint (e.g., step120000)")
    args = parser.parse_args()
    
    if args.model and args.checkpoint:
        # Single run
        extract_controls_for_checkpoint(args.model, args.checkpoint)
    elif args.model:
        # All checkpoints for one model
        checkpoints = next((ckpts for size, ckpts in MODELS if size == args.model), None)
        if checkpoints is None:
            print(f"❌ Invalid model size: {args.model}")
            return
        for checkpoint in checkpoints:
            extract_controls_for_checkpoint(args.model, checkpoint)
    else:
        # Full sweep: all models × all checkpoints
        total = sum(len(ckpts) for _, ckpts in MODELS)
        print(f"Running full sweep: {total} model-checkpoint combinations")
        
        for model_size, checkpoints in MODELS:
            for checkpoint in checkpoints:
                try:
                    extract_controls_for_checkpoint(model_size, checkpoint)
                except Exception as e:
                    print(f"❌ Error on {model_size} {checkpoint}: {e}")
                    import traceback
                    traceback.print_exc()
        
        print(f"\n{'='*60}")
        print("✅ Control binding extraction complete!")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
