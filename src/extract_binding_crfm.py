"""C1-B Phase 1b: Extract EB* binding scores for CRFM GPT-2 Small (5 seeds).

Model: stanford-crfm/gpt2-small-x{seed} (seeds 1–5)
Checkpoints: 8 stages spanning ~5.7B token training run
Prompts: expanded_terms_100.jsonl (9 Set-B terms)
Output: data/results/binding_crfm/{seed}_step{N}_binding_crfm.jsonl

Usage:
    python src/extract_binding_crfm.py --seed 1 --checkpoint checkpoint-2729
    python src/extract_binding_crfm.py --seed 1 --all
    python src/extract_binding_crfm.py --all   # all seeds × all checkpoints
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from extract_attention import extract_binding_for_prompt

PROMPTS_FILE = Path("data/prompts/expanded_terms_100.jsonl")
OUTPUT_DIR = Path("data/results/binding_crfm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 5 CRFM GPT-2 Small seeds — actual HuggingFace model IDs
# Seeds are indexed by the random seed used in training (21, 49, 81, 343, 777)
SEEDS = [1, 2, 3, 4, 5]
SEED_MODEL_IDS = {
    1: "stanford-crfm/alias-gpt2-small-x21",
    2: "stanford-crfm/battlestar-gpt2-small-x49",
    3: "stanford-crfm/caprica-gpt2-small-x81",
    4: "stanford-crfm/darkmatter-gpt2-small-x343",
    5: "stanford-crfm/expanse-gpt2-small-x777",
}

# 8 checkpoints spanning the 400k-step training run (stored as git tags)
# Total training ~300B tokens on The Pile; each step ≈ 750K tokens
# checkpoint-0≈0, 1k≈750M, 5k≈3.75B, 10k≈7.5B, 50k≈37.5B, 100k≈75B, 200k≈150B, 400k≈300B
CHECKPOINTS = [
    "checkpoint-0",
    "checkpoint-1000",
    "checkpoint-5000",
    "checkpoint-10000",
    "checkpoint-50000",
    "checkpoint-100000",
    "checkpoint-200000",
    "checkpoint-400000",
]


def load_crfm(seed: int, checkpoint: str, device: str) -> HookedTransformer:
    """Load CRFM GPT-2 Small checkpoint into HookedTransformer."""
    model_id = SEED_MODEL_IDS[seed]
    print(f"  Loading {model_id} @ {checkpoint} ...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=checkpoint,
        dtype=torch.float32,
    )

    model = HookedTransformer.from_pretrained(
        "gpt2",
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
    )
    return model


def load_prompts(prompts_file=None) -> list[dict]:
    with open(prompts_file or PROMPTS_FILE) as f:
        return [json.loads(line) for line in f]


def extract_for_checkpoint(seed: int, checkpoint: str,
                           prompts_file=None, outdir=None):
    _outdir = Path(outdir) if outdir else OUTPUT_DIR
    _outdir.mkdir(parents=True, exist_ok=True)
    out_file = _outdir / f"seed{seed}_{checkpoint}_binding_crfm.jsonl"
    if out_file.exists():
        print(f"  ⏭  Already exists: {out_file.name} — skipping")
        return out_file

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_crfm(seed, checkpoint, device)
    tokenizer = model.tokenizer

    prompts = load_prompts(prompts_file)
    results = []

    for prompt in tqdm(prompts, desc=f"crfm-x{seed}/{checkpoint}"):
        binding = extract_binding_for_prompt(
            model=model,
            prompt_text=prompt["template"],
            term=prompt["term"],
            tokenizer=tokenizer,
        )
        results.append({
            "model": f"crfm-gpt2-small-x{seed}",
            "checkpoint": checkpoint,
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
    parser.add_argument("--seed", type=int, default=None, choices=SEEDS,
                        help="Seed (1–5)")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint string e.g. checkpoint-2729")
    parser.add_argument("--all", action="store_true",
                        help="Run all seeds × all checkpoints")
    parser.add_argument("--prompts", default=None, help="JSONL prompts file (overrides default)")
    parser.add_argument("--outdir", default=None, help="Output directory (overrides default)")
    args = parser.parse_args()

    if args.all:
        for seed in SEEDS:
            for ck in CHECKPOINTS:
                extract_for_checkpoint(seed, ck, args.prompts, args.outdir)
    elif args.seed and args.checkpoint:
        extract_for_checkpoint(args.seed, args.checkpoint, args.prompts, args.outdir)
    elif args.seed:
        for ck in CHECKPOINTS:
            extract_for_checkpoint(args.seed, ck, args.prompts, args.outdir)
    else:
        parser.print_help()
