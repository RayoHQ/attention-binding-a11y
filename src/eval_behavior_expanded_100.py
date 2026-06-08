"""Evaluate behavioral performance on 100-prompt expanded dataset."""

import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils_model import load_pythia_with_checkpoint
from scoring import score_generation, score_recognition_logprob

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPTS_FILE = Path("data/prompts/expanded_terms_100.jsonl")
OUTPUT_DIR = Path("data/results/behavioral_expanded_100")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model configurations
MODELS = [
    ("160m", ["step0", "step15000", "step30000", "step60000", "step90000", "step120000", "step140000", "step143000"]),
    ("1b", ["step0", "step15000", "step30000", "step60000", "step90000", "step120000", "step140000", "step143000"]),
    ("2.8b", ["step0", "step15000", "step30000", "step60000", "step90000", "step120000", "step140000", "step143000"]),
]


def load_prompts() -> List[Dict]:
    """Load all prompts from expanded dataset."""
    prompts = []
    with open(PROMPTS_FILE) as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


def evaluate_prompt(model, prompt_data: Dict) -> Dict:
    """Evaluate model on a single prompt."""
    task = prompt_data["task"]
    template = prompt_data["template"]
    
    if task == "recognition":
        # Multiple choice - use log-probability scoring
        choices = prompt_data["choices"]
        answer_idx = prompt_data["answer_idx"]
        result = score_recognition_logprob(model, template, choices, answer_idx)
        return {
            "score": result["score"],
            "predicted_idx": result["predicted_idx"],
            "is_correct": result["is_correct"],
            "task": task
        }
    else:  # generation
        # Generate completion and score with keyword rubric
        term = prompt_data["term"]
        max_tokens = prompt_data.get("max_tokens", 25)
        
        tokens = model.to_tokens(template)
        with torch.no_grad():
            output = model.generate(
                tokens,
                max_new_tokens=max_tokens,
                temperature=0.0,
                do_sample=False,
            )
        
        generated_text = model.tokenizer.decode(output[0], skip_special_tokens=True)
        completion = generated_text[len(template):].strip()
        
        score_val = score_generation(completion, term)
        
        return {
            "score": score_val,
            "generated_text": completion,
            "task": task
        }


def evaluate_checkpoint(model_size: str, checkpoint: str):
    """Evaluate all prompts at one checkpoint."""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_size} {checkpoint}")
    print(f"{'='*60}")
    
    # Load model
    model = load_pythia_with_checkpoint(model_size, checkpoint, DEVICE)
    tokenizer = model.tokenizer
    
    # Load prompts
    prompts = load_prompts()
    print(f"Loaded {len(prompts)} prompts")
    
    # Evaluate each prompt
    results = []
    for prompt in tqdm(prompts, desc=f"Evaluating {model_size}/{checkpoint}"):
        eval_result = evaluate_prompt(model, prompt)
        
        result = {
            "model": f"pythia-{model_size}-deduped",
            "checkpoint": checkpoint,
            "term": prompt["term"],
            "task": prompt["task"],
            "prompt_id": prompt["prompt_id"],
            "prompt_template": prompt["template"],
            "behavioral_score": eval_result["score"],
        }
        results.append(result)
    
    # Save results
    output_file = OUTPUT_DIR / f"{model_size}_{checkpoint}_behavioral_100.jsonl"
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    # Summary stats
    recognition_scores = [r["behavioral_score"] for r in results if r["task"] == "recognition"]
    generation_scores = [r["behavioral_score"] for r in results if r["task"] == "generation"]
    
    print(f"✅ Saved {len(results)} results to {output_file}")
    print(f"   Recognition: {len(recognition_scores)} prompts, mean={sum(recognition_scores)/len(recognition_scores):.3f}")
    print(f"   Generation: {len(generation_scores)} prompts, mean={sum(generation_scores)/len(generation_scores):.3f}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return output_file


def main():
    """Run behavioral evaluation for all models and checkpoints."""
    total = sum(len(ckpts) for _, ckpts in MODELS)
    print(f"Running behavioral evaluation on 100-prompt dataset: {total} model-checkpoint combinations")
    print(f"Prompts file: {PROMPTS_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    for model_size, checkpoints in MODELS:
        for checkpoint in checkpoints:
            try:
                evaluate_checkpoint(model_size, checkpoint)
            except Exception as e:
                print(f"❌ Error on {model_size} {checkpoint}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("✅ 100-prompt behavioral evaluation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
