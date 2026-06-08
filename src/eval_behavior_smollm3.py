"""C1-B Phase 1c: Behavioral evaluation for SmolLM3-3B lifecycle checkpoints.

Model: HuggingFaceTB/SmolLM3-3B (intermediate training checkpoints)
Prompts: expanded_terms_100.jsonl (9 Set-B terms)
Output: data/results/behavioral_smollm3/

Usage:
    python src/eval_behavior_smollm3.py --checkpoint step40k
    python src/eval_behavior_smollm3.py --all
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils_model_smollm3 import (
    load_smollm3_with_checkpoint,
    CHECKPOINT_KEYS,
    SMOLLM3_MODEL_ID,
)
from scoring import score_generation, score_recognition_logprob

PROMPTS_FILE = Path("data/prompts/expanded_terms_100.jsonl")
OUTPUT_DIR = Path("data/results/behavioral_smollm3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_prompts(prompts_file=None) -> list[dict]:
    with open(prompts_file or PROMPTS_FILE) as f:
        return [json.loads(line) for line in f]


def evaluate_prompt(model, tokenizer, prompt_data: dict) -> dict:
    task = prompt_data["task"]
    template = prompt_data["template"]

    if task == "recognition":
        choices = prompt_data["choices"]
        answer_idx = prompt_data["answer_idx"]

        log_probs = []
        for choice in choices:
            full_text = template + " " + choice
            inputs = tokenizer(full_text, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
            log_probs.append(-outputs.loss.item())

        predicted_idx = int(torch.tensor(log_probs).argmax().item())
        is_correct = predicted_idx == answer_idx
        return {
            "behavioral_score": float(is_correct),
            "predicted_idx": predicted_idx,
            "is_correct": bool(is_correct),
        }
    else:
        term = prompt_data["term"]
        max_tokens = prompt_data.get("max_tokens", 25)
        inputs = tokenizer(template, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=1.0,
            )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        completion = generated_text[len(template):].strip()
        return {
            "behavioral_score": float(score_generation(completion, term)),
            "generated_text": completion,
        }


def evaluate_checkpoint(checkpoint_key: str, prompts_file=None, outdir=None):
    _outdir = Path(outdir) if outdir else OUTPUT_DIR
    _outdir.mkdir(parents=True, exist_ok=True)
    out_file = _outdir / f"smollm3_{checkpoint_key}_behavioral_smollm3.jsonl"
    if out_file.exists():
        print(f"  ⏭  Already exists: {out_file.name} — skipping")
        return out_file

    print(f"\nLoading SmolLM3-3B @ {checkpoint_key} ...")
    model, tokenizer = load_smollm3_with_checkpoint(checkpoint_key, DEVICE)

    prompts = load_prompts(prompts_file)
    results = []

    for prompt in tqdm(prompts, desc=f"behavioral smollm3/{checkpoint_key}"):
        scores = evaluate_prompt(model, tokenizer, prompt)
        results.append({
            "model": "smollm3-3b",
            "checkpoint": checkpoint_key,
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
    parser.add_argument("--checkpoint", default=None, choices=CHECKPOINT_KEYS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--prompts", default=None, help="JSONL prompts file")
    parser.add_argument("--outdir", default=None, help="Output directory")
    args = parser.parse_args()

    if args.all:
        for ck in CHECKPOINT_KEYS:
            evaluate_checkpoint(ck, args.prompts, args.outdir)
    elif args.checkpoint:
        evaluate_checkpoint(args.checkpoint, args.prompts, args.outdir)
    else:
        parser.print_help()
