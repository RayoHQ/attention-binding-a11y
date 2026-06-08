"""A4: Extract EB* binding scores for all 30 terms across Pythia 160m/1b/2.8b.

Extends the existing extract_attention.py pipeline to include the 21 new
Tier 1/2/3 terms from data/prompts/expanded_terms_tier123.jsonl.

Usage:
    python src/extract_binding_30terms.py --model 160m --checkpoint step143000
    python src/extract_binding_30terms.py --model 2.8b --checkpoint step0
    python src/extract_binding_30terms.py --model 1b --all
"""

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

from utils_model import CHECKPOINT_STEPS, load_pythia_with_checkpoint

OUTPUT_DIR = Path("data/results/binding_tier123")

# Tier 1/2/3 prompts (21 new terms × 11 prompts = 231 entries)
NEW_TERMS_FILE = "data/prompts/expanded_terms_tier123.jsonl"
# Original 9 terms × 11 prompts = 99 entries
ORIG_TERMS_FILE = "data/prompts/expanded_terms_100.jsonl"


def load_all_prompts(new_only: bool = False) -> list[dict]:
    prompts = []
    if not new_only:
        with open(ORIG_TERMS_FILE) as f:
            prompts.extend(json.loads(line) for line in f)
    with open(NEW_TERMS_FILE) as f:
        prompts.extend(json.loads(line) for line in f)
    return prompts


def extract_for_checkpoint(model_size: str, checkpoint_step: str, new_only: bool = False):
    """Run EB* extraction for one Pythia model/checkpoint."""
    from extract_attention import extract_binding_for_prompt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading pythia-{model_size} {checkpoint_step}...")
    model = load_pythia_with_checkpoint(model_size, checkpoint_step, device)
    tokenizer = model.tokenizer

    prompts = load_all_prompts(new_only=new_only)
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = OUTPUT_DIR / f"{model_size}_{checkpoint_step}_binding_tier123.jsonl"
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(results)} results to {out_file}")

    del model
    torch.cuda.empty_cache()
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["160m", "1b", "2.8b"], required=True)
    parser.add_argument(
        "--checkpoint",
        choices=[f"step{s}000" if s > 0 else "step0" for s in CHECKPOINT_STEPS]
        + ["step0", "step15000", "step30000", "step60000",
           "step90000", "step120000", "step140000", "step143000"],
        help="Checkpoint step string",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 8 checkpoints sequentially",
    )
    parser.add_argument(
        "--new-only",
        action="store_true",
        help="Extract only the 21 new terms (skip original 9)",
    )
    args = parser.parse_args()

    ALL_STEPS = [
        "step0", "step15000", "step30000", "step60000",
        "step90000", "step120000", "step140000", "step143000",
    ]

    if args.all:
        for step in ALL_STEPS:
            extract_for_checkpoint(args.model, step, new_only=args.new_only)
    elif args.checkpoint:
        extract_for_checkpoint(args.model, args.checkpoint, new_only=args.new_only)
    else:
        parser.print_help()
