"""Extract binding metrics for expanded accessibility terms across all checkpoints."""

import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils_model import load_pythia_with_checkpoint
from extract_attention import extract_binding_for_prompt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPTS_FILE = Path("data/prompts/expanded_terms.jsonl")
OUTPUT_DIR = Path("data/results/binding_expanded")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model configurations
MODELS = [
    ("160m", ["step0", "step15000", "step30000", "step60000", "step90000", "step120000", "step140000", "step143000"]),
    ("1b", ["step0", "step15000", "step30000", "step60000", "step90000", "step120000", "step140000", "step143000"]),
    ("2.8b", ["step0", "step15000", "step30000", "step60000", "step90000", "step120000", "step140000", "step143000"]),
]


def load_prompts():
    """Load expanded term prompts from JSONL - generation only for binding."""
    prompts = []
    with open(PROMPTS_FILE) as f:
        for line in f:
            prompt = json.loads(line)
            # Only use generation prompts for binding extraction (simpler, shorter)
            if prompt["task"] == "generation":
                prompts.append(prompt)
    return prompts


def extract_for_checkpoint(model_size: str, checkpoint: str):
    """Extract binding for all expanded terms at one checkpoint."""
    print(f"\n{'='*60}")
    print(f"Processing {model_size} {checkpoint}")
    print(f"{'='*60}")
    
    # Load model
    model = load_pythia_with_checkpoint(model_size, checkpoint, DEVICE)
    tokenizer = model.tokenizer
    
    # Load prompts
    prompts = load_prompts()
    
    # Process each prompt
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
            "task": prompt["task"],
            "prompt_id": prompt["prompt_id"],
            "prompt_template": prompt["template"],
            **binding,
        }
        results.append(result)
    
    # Save results
    output_file = OUTPUT_DIR / f"{model_size}_{checkpoint}_binding.jsonl"
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    print(f"✅ Saved {len(results)} results to {output_file}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return output_file


def main():
    """Run binding extraction for all models and checkpoints."""
    total = sum(len(ckpts) for _, ckpts in MODELS)
    print(f"Running binding extraction on expanded terms: {total} model-checkpoint combinations")
    
    for model_size, checkpoints in MODELS:
        for checkpoint in checkpoints:
            try:
                extract_for_checkpoint(model_size, checkpoint)
            except Exception as e:
                print(f"❌ Error on {model_size} {checkpoint}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("✅ Expanded term binding extraction complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
