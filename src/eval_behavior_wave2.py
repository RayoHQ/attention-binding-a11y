"""C1-B Phase 1: Behavioral evaluation for 12 wave-2 terms across all lifecycle checkpoints.

Mirrors eval_behavior_tier123.py but targets expanded_terms_wave2.jsonl.
Outputs to data/results/behavioral_wave2/.

Usage:
    python src/eval_behavior_wave2.py --model 160m --checkpoint step143000
    python src/eval_behavior_wave2.py --model 160m --all
    python src/eval_behavior_wave2.py --all   # all models × all checkpoints
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
PROMPTS_FILE = Path("data/prompts/expanded_terms_wave2.jsonl")
OUTPUT_DIR = Path("data/results/behavioral_wave2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_CHECKPOINTS = [
    "step0", "step15000", "step30000", "step60000",
    "step90000", "step120000", "step140000", "step143000",
]
ALL_MODELS = ["160m", "1b", "2.8b"]


def load_prompts() -> list[dict]:
    with open(PROMPTS_FILE) as f:
        return [json.loads(line) for line in f]


def evaluate_prompt(model, prompt_data: dict) -> dict:
    task = prompt_data["task"]
    template = prompt_data["template"]

    if task == "recognition":
        choices = prompt_data["choices"]
        answer_idx = prompt_data["answer_idx"]
        result = score_recognition_logprob(model, template, choices, answer_idx)
        return {
            "behavioral_score": float(result["score"]),
            "predicted_idx": result["predicted_idx"],
            "is_correct": result["is_correct"],
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
        completion = model.tokenizer.decode(output[0], skip_special_tokens=True)
        completion = completion[len(template):].strip()
        return {
            "behavioral_score": float(score_generation(completion, term)),
            "generated_text": completion,
        }


def evaluate_checkpoint(model_size: str, checkpoint_step: str):
    out_file = OUTPUT_DIR / f"{model_size}_{checkpoint_step}_behavioral_wave2.jsonl"
    if out_file.exists():
        print(f"  ⏭  Already exists: {out_file.name} — skipping")
        return out_file

    print(f"\nLoading pythia-{model_size} {checkpoint_step}...")
    model = load_pythia_with_checkpoint(model_size, checkpoint_step, DEVICE)

    prompts = load_prompts()
    results = []

    for prompt in tqdm(prompts, desc=f"behavioral pythia-{model_size}/{checkpoint_step}"):
        scores = evaluate_prompt(model, prompt)
        results.append({
            "model": f"pythia-{model_size}-deduped",
            "checkpoint": checkpoint_step,
            "term": prompt["term"],
            "task": prompt["task"],
            "prompt_id": prompt["prompt_id"],
            "prompt_template": prompt["template"],
            **scores,
        })

    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"  ✅ Saved {len(results)} records → {out_file}")

    del model
    torch.cuda.empty_cache()
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, choices=ALL_MODELS + ["all"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all or args.model == "all":
        for model_size in ALL_MODELS:
            for ck in ALL_CHECKPOINTS:
                evaluate_checkpoint(model_size, ck)
    elif args.model and args.checkpoint:
        evaluate_checkpoint(args.model, args.checkpoint)
    elif args.model:
        for ck in ALL_CHECKPOINTS:
            evaluate_checkpoint(args.model, ck)
    else:
        parser.print_help()
