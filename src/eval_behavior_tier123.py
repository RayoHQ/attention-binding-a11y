"""A5: Behavioral evaluation for 21 new Tier 1/2/3 accessibility terms.

Mirrors eval_behavior_expanded_100.py but targets expanded_terms_tier123.jsonl.
Outputs to data/results/behavioral_tier123/.

Usage:
    python src/eval_behavior_tier123.py                    # all models/checkpoints
    python src/eval_behavior_tier123.py --model 160m       # one model only
    python src/eval_behavior_tier123.py --model 2.8b --checkpoint step143000
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils_model import load_pythia_with_checkpoint
from scoring import score_generation, score_recognition_logprob

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPTS_FILE = Path("data/prompts/expanded_terms_tier123.jsonl")
OUTPUT_DIR = Path("data/results/behavioral_tier123")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_CHECKPOINTS = [
    "step0", "step15000", "step30000", "step60000",
    "step90000", "step120000", "step140000", "step143000",
]

ALL_MODELS = ["160m", "1b", "2.8b"]


def load_prompts() -> list[dict]:
    prompts = []
    with open(PROMPTS_FILE) as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


def evaluate_prompt(model, prompt_data: dict) -> dict:
    task = prompt_data["task"]
    template = prompt_data["template"]

    if task == "recognition":
        choices = prompt_data["choices"]
        answer_idx = prompt_data["answer_idx"]
        result = score_recognition_logprob(model, template, choices, answer_idx)
        return {
            "score": result["score"],
            "predicted_idx": result["predicted_idx"],
            "is_correct": result["is_correct"],
            "task": task,
        }
    else:
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
        return {
            "score": score_generation(completion, term),
            "generated_text": completion,
            "task": task,
        }


def evaluate_checkpoint(model_size: str, checkpoint: str):
    out_file = OUTPUT_DIR / f"{model_size}_{checkpoint}_behavioral_tier123.jsonl"
    if out_file.exists():
        print(f"  ⏭  Skipping {model_size} {checkpoint} (already exists)")
        return out_file

    print(f"\n{'='*60}")
    print(f"Evaluating pythia-{model_size} {checkpoint} | Tier1/2/3 terms")
    print(f"{'='*60}")

    model = load_pythia_with_checkpoint(model_size, checkpoint, DEVICE)
    prompts = load_prompts()
    print(f"Loaded {len(prompts)} prompts across 21 terms")

    results = []
    for prompt in tqdm(prompts, desc=f"{model_size}/{checkpoint}"):
        eval_result = evaluate_prompt(model, prompt)
        results.append({
            "model": f"pythia-{model_size}-deduped",
            "checkpoint": checkpoint,
            "term": prompt["term"],
            "task": prompt["task"],
            "prompt_id": prompt["prompt_id"],
            "prompt_template": prompt["template"],
            "behavioral_score": eval_result["score"],
        })

    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    rec = [r["behavioral_score"] for r in results if r["task"] == "recognition"]
    gen = [r["behavioral_score"] for r in results if r["task"] == "generation"]
    print(f"Saved {len(results)} results → {out_file}")
    print(f"  Recognition mean={sum(rec)/len(rec):.3f}  Generation mean={sum(gen)/len(gen):.3f}")

    del model
    torch.cuda.empty_cache()
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=ALL_MODELS, help="Run one model only")
    parser.add_argument("--checkpoint", choices=ALL_CHECKPOINTS, help="Run one checkpoint only")
    args = parser.parse_args()

    models = [args.model] if args.model else ALL_MODELS
    checkpoints = [args.checkpoint] if args.checkpoint else ALL_CHECKPOINTS

    total = len(models) * len(checkpoints)
    print(f"Running {total} model-checkpoint combos | 21 Tier1/2/3 terms | 231 prompts each")

    for ms in models:
        for ck in checkpoints:
            try:
                evaluate_checkpoint(ms, ck)
            except Exception as e:
                print(f"❌ Error {ms} {ck}: {e}")
                import traceback; traceback.print_exc()

    print("\n✅ Tier1/2/3 behavioral evaluation complete!")
