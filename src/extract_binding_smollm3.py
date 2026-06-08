"""C1-B Phase 1c: Extract EB* binding scores for SmolLM3-3B lifecycle checkpoints.

Model: HuggingFaceTB/SmolLM3-3B (intermediate training checkpoints)
Prompts: expanded_terms_100.jsonl (9 Set-B terms)
Output: data/results/binding_smollm3/

Usage:
    python src/extract_binding_smollm3.py --checkpoint step40k
    python src/extract_binding_smollm3.py --all
    python src/extract_binding_smollm3.py --probe   # list available checkpoints
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
    SmolLM3AttentionExtractor,
    CHECKPOINT_KEYS,
    probe_checkpoints,
)

PROMPTS_FILE = Path("data/prompts/expanded_terms_100.jsonl")
OUTPUT_DIR = Path("data/results/binding_smollm3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_prompts() -> list[dict]:
    with open(PROMPTS_FILE) as f:
        return [json.loads(line) for line in f]


def extract_for_checkpoint(checkpoint_key: str, prompts_file=None, outdir=None):
    _outdir = Path(outdir) if outdir else OUTPUT_DIR
    _outdir.mkdir(parents=True, exist_ok=True)
    out_file = _outdir / f"smollm3_{checkpoint_key}_binding_smollm3.jsonl"
    if out_file.exists():
        print(f"  ⏭  Already exists: {out_file.name} — skipping")
        return out_file

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading SmolLM3-3B @ {checkpoint_key} ...")
    model, tokenizer = load_smollm3_with_checkpoint(checkpoint_key, device)
    extractor = SmolLM3AttentionExtractor(model, tokenizer)

    pfile = prompts_file or PROMPTS_FILE
    with open(pfile) as fh:
        prompts = [json.loads(l) for l in fh]
    results = []

    for prompt in tqdm(prompts, desc=f"smollm3/{checkpoint_key}"):
        binding = extractor.extract_binding_for_prompt(
            prompt_text=prompt["template"],
            term=prompt["term"],
        )
        results.append({
            "model": "smollm3-3b",
            "checkpoint": checkpoint_key,
            "seed": "smollm3",
            "term": prompt["term"],
            "task": prompt["task"],
            "prompt_id": prompt["prompt_id"],
            "prompt_template": prompt["template"],
            **binding,
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
    parser.add_argument("--checkpoint", default=None, choices=CHECKPOINT_KEYS,
                        help="Checkpoint key e.g. step40k, step1m")
    parser.add_argument("--all", action="store_true",
                        help="Run all checkpoints")
    parser.add_argument("--probe", action="store_true",
                        help="List available HuggingFace revisions and exit")
    parser.add_argument("--prompts", default=None, help="JSONL prompts file")
    parser.add_argument("--outdir", default=None, help="Output directory")
    args = parser.parse_args()

    if args.probe:
        branches = probe_checkpoints()
        print("\nAvailable branches:", branches)
    elif args.all:
        for ck in CHECKPOINT_KEYS:
            extract_for_checkpoint(ck, args.prompts, args.outdir)
    elif args.checkpoint:
        extract_for_checkpoint(args.checkpoint, args.prompts, args.outdir)
    else:
        parser.print_help()
