"""C1-B Phase 1: Extract EB* binding scores for 12 wave-2 terms across all lifecycle checkpoints.

Mirrors extract_binding_30terms.py but targets expanded_terms_wave2.jsonl.
Outputs to data/results/binding_wave2/.

Usage:
    python src/extract_binding_wave2.py --model 160m --checkpoint step143000
    python src/extract_binding_wave2.py --model 160m --all
    python src/extract_binding_wave2.py --all   # all models × all checkpoints
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils_model import CHECKPOINT_STEPS, load_pythia_with_checkpoint

OUTPUT_DIR = Path("data/results/binding_wave2")
PROMPTS_FILE = Path("data/prompts/expanded_terms_wave2.jsonl")

ALL_MODELS = ["160m", "1b", "2.8b"]
ALL_CHECKPOINTS = [
    "step0", "step15000", "step30000", "step60000",
    "step90000", "step120000", "step140000", "step143000",
]


def load_prompts() -> list[dict]:
    with open(PROMPTS_FILE) as f:
        return [json.loads(line) for line in f]


def extract_for_checkpoint(model_size: str, checkpoint_step: str):
    from extract_attention import extract_binding_for_prompt

    out_file = OUTPUT_DIR / f"{model_size}_{checkpoint_step}_binding_wave2.jsonl"
    if out_file.exists():
        print(f"  ⏭  Already exists: {out_file.name} — skipping")
        return out_file

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading pythia-{model_size} {checkpoint_step}...")
    model = load_pythia_with_checkpoint(model_size, checkpoint_step, device)
    tokenizer = model.tokenizer

    prompts = load_prompts()
    results = []

    for prompt in tqdm(prompts, desc=f"pythia-{model_size}/{checkpoint_step}"):
        binding = extract_binding_for_prompt(
            model=model,
            prompt_text=prompt["template"],
            term=prompt["term"],
            tokenizer=tokenizer,
        )
        results.append({
            "model": f"pythia-{model_size}-deduped",
            "checkpoint": checkpoint_step,
            "term": prompt["term"],
            "task": prompt["task"],
            "prompt_id": prompt["prompt_id"],
            "prompt_template": prompt["template"],
            **binding,
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"  ✅ Saved {len(results)} records → {out_file}")

    del model
    torch.cuda.empty_cache()
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, choices=ALL_MODELS + ["all"],
                        help="Model size or 'all'")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint step (e.g. step15000) or omit for --all")
    parser.add_argument("--all", action="store_true",
                        help="Run all models × all checkpoints")
    args = parser.parse_args()

    if args.all or args.model == "all":
        for model_size in ALL_MODELS:
            for ck in ALL_CHECKPOINTS:
                extract_for_checkpoint(model_size, ck)
    elif args.model and args.checkpoint:
        extract_for_checkpoint(args.model, args.checkpoint)
    elif args.model:
        for ck in ALL_CHECKPOINTS:
            extract_for_checkpoint(args.model, ck)
    else:
        parser.print_help()
