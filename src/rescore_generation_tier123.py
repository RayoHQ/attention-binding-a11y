"""Re-run generation-only behavioral eval for Tier1/2/3 terms.

Existing behavioral_tier123 files have correct recognition scores but
generation scores are all 0.0 (scoring.py lacked keyword lists).
This script re-runs ONLY generation prompts and merges corrected scores.

Usage:
    python src/rescore_generation_tier123.py --model 160m
    python src/rescore_generation_tier123.py --all
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils_model import load_pythia_with_checkpoint
from scoring import score_generation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPTS_FILE = Path("data/prompts/expanded_terms_tier123.jsonl")
OUTPUT_DIR = Path("data/results/behavioral_tier123")

ALL_CHECKPOINTS = [
    "step0", "step15000", "step30000", "step60000",
    "step90000", "step120000", "step140000", "step143000",
]
ALL_MODELS = ["160m", "1b", "2.8b"]


def load_gen_prompts() -> list[dict]:
    prompts = []
    with open(PROMPTS_FILE) as f:
        for line in f:
            p = json.loads(line)
            if p["task"] == "generation":
                prompts.append(p)
    return prompts


def rescore_checkpoint(model_size: str, checkpoint: str):
    out_file = OUTPUT_DIR / f"{model_size}_{checkpoint}_behavioral_tier123.jsonl"
    if not out_file.exists():
        print(f"  ⚠  Missing {out_file.name} — skipping")
        return

    existing = []
    with open(out_file) as f:
        for line in f:
            existing.append(json.loads(line))

    # Check if gen scores are already non-zero (already rescored)
    gen_scores = [r["behavioral_score"] for r in existing if r["task"] == "generation"]
    if gen_scores and sum(gen_scores) > 0:
        print(f"  ⏭  {model_size} {checkpoint} gen scores already non-zero — skipping")
        return

    print(f"\n{'='*60}")
    print(f"Rescoring generation: pythia-{model_size} {checkpoint}")
    print(f"{'='*60}")

    model = load_pythia_with_checkpoint(model_size, checkpoint, DEVICE)
    gen_prompts = load_gen_prompts()

    # Build index: (term, prompt_id) -> new score
    new_scores: dict[tuple, float] = {}
    for prompt in tqdm(gen_prompts, desc=f"{model_size}/{checkpoint} gen"):
        max_tokens = prompt.get("max_tokens", 25)
        tokens = model.to_tokens(prompt["template"])
        with torch.no_grad():
            output = model.generate(
                tokens,
                max_new_tokens=max_tokens,
                temperature=0.0,
                do_sample=False,
            )
        generated_text = model.tokenizer.decode(output[0], skip_special_tokens=True)
        completion = generated_text[len(prompt["template"]):].strip()
        score = score_generation(completion, prompt["term"])
        new_scores[(prompt["term"], prompt["prompt_id"])] = score

    # Merge: replace generation scores in existing records
    updated = []
    for r in existing:
        if r["task"] == "generation":
            key = (r["term"], r["prompt_id"])
            r = dict(r)
            r["behavioral_score"] = new_scores.get(key, 0.0)
        updated.append(r)

    with open(out_file, "w") as f:
        for r in updated:
            f.write(json.dumps(r) + "\n")

    rec = [r["behavioral_score"] for r in updated if r["task"] == "recognition"]
    gen = [r["behavioral_score"] for r in updated if r["task"] == "generation"]
    print(f"Updated {out_file.name}")
    print(f"  Recognition mean={sum(rec)/len(rec):.3f}  Generation mean={sum(gen)/len(gen):.3f}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=ALL_MODELS)
    parser.add_argument("--checkpoint", choices=ALL_CHECKPOINTS)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    models = [args.model] if args.model else ALL_MODELS
    checkpoints = [args.checkpoint] if args.checkpoint else ALL_CHECKPOINTS

    if args.all or (not args.model and not args.checkpoint):
        models = ALL_MODELS
        checkpoints = ALL_CHECKPOINTS

    for ms in models:
        for ck in checkpoints:
            try:
                rescore_checkpoint(ms, ck)
            except Exception as e:
                print(f"❌ Error {ms} {ck}: {e}")
                import traceback; traceback.print_exc()

    print("\n✅ Generation rescoring complete!")
