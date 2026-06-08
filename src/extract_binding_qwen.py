"""Extract EB* binding scores for Qwen2.5-1.5B (single final checkpoint).

Usage:
    python src/extract_binding_qwen.py
    python src/extract_binding_qwen.py --prompts data/prompts/canonical_45terms.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils_model_qwen import load_qwen, QwenAttentionExtractor

PROMPTS_FILE = Path("data/prompts/canonical_45terms.jsonl")
OUTPUT_DIR = Path("data/results/binding_qwen")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run(prompts_file=None, outdir=None):
    _prompts = Path(prompts_file) if prompts_file else PROMPTS_FILE
    _outdir = Path(outdir) if outdir else OUTPUT_DIR
    _outdir.mkdir(parents=True, exist_ok=True)
    out_file = _outdir / "qwen_final_binding_qwen.jsonl"

    if out_file.exists():
        print(f"⏭  Already exists: {out_file} — skipping")
        return out_file

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading Qwen2.5-1.5B ...")
    model, tokenizer = load_qwen(device)
    extractor = QwenAttentionExtractor(model, tokenizer)

    with open(_prompts) as f:
        prompts = [json.loads(l) for l in f]

    results = []
    for prompt in tqdm(prompts, desc="qwen2.5-1.5b/final"):
        binding = extractor.extract_binding_for_prompt(
            prompt_text=prompt["template"],
            term=prompt["term"],
        )
        results.append({
            "model": "qwen2.5-1.5b",
            "checkpoint": "final",
            "term": prompt["term"],
            "task": prompt["task"],
            "prompt_id": prompt["prompt_id"],
            "prompt_template": prompt["template"],
            **binding,
        })

    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"✅ Saved {len(results)} records → {out_file}")

    del model
    torch.cuda.empty_cache()
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default=None, help="JSONL prompts file")
    parser.add_argument("--outdir", default=None, help="Output directory")
    args = parser.parse_args()
    run(args.prompts, args.outdir)
