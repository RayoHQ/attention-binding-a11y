"""B4: Extract EB* binding scores from OLMo-1B across 8 training checkpoints.

Replicates the Pythia extraction pipeline (extract_attention.py) on OLMo-1B-hf
using the HF forward-hook extractor in utils_model_olmo.py.

Usage:
    python src/extract_binding_olmo.py --checkpoint step0
    python src/extract_binding_olmo.py --checkpoint step143k
    python src/extract_binding_olmo.py --all
"""

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

from utils_model_olmo import (
    OLMO_1B_CHECKPOINTS,
    OLMoAttentionHookExtractor,
    load_olmo_with_checkpoint,
)

OUTPUT_DIR = Path("data/results/binding")
PROMPTS_FILE = "data/prompts/expanded_terms_100.jsonl"  # 9 terms × 11 prompts


def extract_for_checkpoint(checkpoint_key: str, prompts_file: str = PROMPTS_FILE,
                           outdir: str = None):
    """Run EB* extraction for one OLMo checkpoint."""
    print(f"\nLoading OLMo-1B {checkpoint_key} ({OLMO_1B_CHECKPOINTS[checkpoint_key]})...")
    model, tokenizer = load_olmo_with_checkpoint(checkpoint_key)
    extractor = OLMoAttentionHookExtractor(model, tokenizer)

    prompts = []
    with open(prompts_file) as f:
        for line in f:
            prompts.append(json.loads(line))

    results = []
    for prompt in tqdm(prompts, desc=f"OLMo-1B {checkpoint_key}"):
        binding = extractor.extract_binding_for_prompt(
            prompt_text=prompt["template"],
            term=prompt["term"],
        )
        results.append({
            "model": "olmo-1b",
            "checkpoint": checkpoint_key,
            "olmo_revision": OLMO_1B_CHECKPOINTS[checkpoint_key],
            "term": prompt["term"],
            "task": prompt["task"],
            "prompt_id": prompt["prompt_id"],
            "prompt_template": prompt["template"],
            **binding,
        })

    _outdir = Path(outdir) if outdir else OUTPUT_DIR
    os.makedirs(_outdir, exist_ok=True)
    out_file = _outdir / f"olmo_1b_{checkpoint_key}_binding.jsonl"
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Saved {len(results)} results to {out_file}")

    del model
    torch.cuda.empty_cache()
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        choices=list(OLMO_1B_CHECKPOINTS.keys()),
        help="Checkpoint key to run",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 8 checkpoints sequentially",
    )
    parser.add_argument(
        "--prompts",
        default=PROMPTS_FILE,
        help="JSONL prompts file",
    )
    parser.add_argument("--outdir", default=None, help="Output directory")
    args = parser.parse_args()

    if args.all:
        for ck in OLMO_1B_CHECKPOINTS:
            extract_for_checkpoint(ck, args.prompts, args.outdir)
    elif args.checkpoint:
        extract_for_checkpoint(args.checkpoint, args.prompts, args.outdir)
    else:
        parser.print_help()
