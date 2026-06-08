"""Behavioral evaluation for expanded accessibility terms."""

import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils_model import load_pythia_with_checkpoint
from scoring import score_generation, score_recognition_logprob

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPTS_FILE = Path("data/prompts/expanded_terms.jsonl")
OUTPUT_DIR = Path("data/results/behavioral_expanded")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model configurations
MODELS = [
    ("160m", ["step0", "step15000", "step30000", "step60000", "step90000", "step120000", "step140000", "step143000"]),
    ("1b", ["step0", "step15000", "step30000", "step60000", "step90000", "step120000", "step140000", "step143000"]),
    ("2.8b", ["step0", "step15000", "step30000", "step60000", "step90000", "step120000", "step140000", "step143000"]),
]


def load_prompts():
    """Load expanded term prompts from JSONL."""
    prompts = []
    with open(PROMPTS_FILE) as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


def run_recognition(model, prompt_data):
    """Run recognition task (multiple choice)."""
    template = prompt_data["template"]
    choices = prompt_data["choices"]
    answer_idx = prompt_data["answer_idx"]
    
    # Use the standard log-probability scoring
    result = score_recognition_logprob(
        model=model,
        prompt=template,
        choices=choices,
        answer_idx=answer_idx,
    )
    
    return {
        "accuracy": result["score"],
        "predicted_idx": result["predicted_idx"],
        "correct_idx": answer_idx,
    }


def run_generation(model, prompt_data, term):
    """Run generation task."""
    template = prompt_data["template"]
    max_tokens = prompt_data.get("max_tokens", 20)
    
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
    
    # Score using keyword rubric
    score = score_generation(completion, term)
    
    return {
        "score": score,
        "generated_text": completion,
    }


def evaluate_checkpoint(model_size: str, checkpoint: str):
    """Evaluate all expanded terms at one checkpoint."""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_size} {checkpoint}")
    print(f"{'='*60}")
    
    # Load model
    model = load_pythia_with_checkpoint(model_size, checkpoint, DEVICE)
    
    # Load prompts
    prompts = load_prompts()
    
    # Group prompts by term
    from collections import defaultdict
    prompts_by_term = defaultdict(list)
    for p in prompts:
        prompts_by_term[p["term"]].append(p)
    
    # Evaluate each term
    results = []
    for term, term_prompts in tqdm(prompts_by_term.items(), desc=f"Terms {model_size}/{checkpoint}"):
        rec_scores = []
        gen_scores = []
        
        for prompt in term_prompts:
            if prompt["task"] == "recognition":
                rec_result = run_recognition(model, prompt)
                rec_scores.append(rec_result["accuracy"])
            elif prompt["task"] == "generation":
                gen_result = run_generation(model, prompt, term)
                gen_scores.append(gen_result["score"])
        
        # Aggregate scores for this term
        result = {
            "model": f"pythia-{model_size}-deduped",
            "checkpoint": checkpoint,
            "term": term,
            "rec_acc": sum(rec_scores) / len(rec_scores) if rec_scores else 0.0,
            "gen_mean": sum(gen_scores) / len(gen_scores) if gen_scores else 0.0,
            "beh_avg": (sum(rec_scores) / len(rec_scores) + sum(gen_scores) / len(gen_scores)) / 2 if (rec_scores and gen_scores) else 0.0,
            "n_rec": len(rec_scores),
            "n_gen": len(gen_scores),
        }
        results.append(result)
    
    # Save results
    output_file = OUTPUT_DIR / f"{model_size}_{checkpoint}_behavioral.jsonl"
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    print(f"✅ Saved {len(results)} term results to {output_file}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return output_file


def main():
    """Run behavioral evaluation for all models and checkpoints."""
    total = sum(len(ckpts) for _, ckpts in MODELS)
    print(f"Running behavioral evaluation on expanded terms: {total} model-checkpoint combinations")
    
    for model_size, checkpoints in MODELS:
        for checkpoint in checkpoints:
            try:
                evaluate_checkpoint(model_size, checkpoint)
            except Exception as e:
                print(f"❌ Error on {model_size} {checkpoint}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("✅ Expanded term behavioral evaluation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
