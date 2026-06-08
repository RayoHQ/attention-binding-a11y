"""Phase 4 C5: Run causal ablation on the full canonical 41-term set.

Wraps run_causal_c5.py for all lifecycle models at their final trained checkpoint.
Uses data/prompts/canonical_45terms.jsonl (41 unique terms, 451 prompts).

Usage:
    python src/run_c5_canonical.py --model 160m
    python src/run_c5_canonical.py --all      # all 3 Pythia models
"""

import argparse
import subprocess
import sys
from pathlib import Path

CANONICAL_PROMPTS = "data/prompts/canonical_45terms.jsonl"
OUTPUT_DIR = Path("data/results/causal")

# Final trained checkpoints per model
MODEL_CHECKPOINTS = {
    "160m": "step143000",
    "1b":   "step143000",
    "2.8b": "step143000",
}

ALL_MODELS = list(MODEL_CHECKPOINTS.keys())


def run_model(model_size: str):
    checkpoint = MODEL_CHECKPOINTS[model_size]
    out_file = OUTPUT_DIR / f"{model_size}_{checkpoint}_c5_canonical41.json"

    if out_file.exists():
        print(f"⏭  Already exists: {out_file.name} — skipping")
        return

    cmd = [
        sys.executable, "src/run_causal_c5.py",
        "--model",      model_size,
        "--checkpoint", checkpoint,
        "--prompts",    CANONICAL_PROMPTS,
        "--output",     str(out_file),
    ]
    print(f"\n▶  {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"❌ Failed for pythia-{model_size}")
    else:
        print(f"✅ Done: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, choices=ALL_MODELS + ["all"])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all or args.model == "all":
        for m in ALL_MODELS:
            run_model(m)
    elif args.model:
        run_model(args.model)
    else:
        parser.print_help()
