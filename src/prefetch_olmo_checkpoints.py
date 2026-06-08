"""Pre-fetch all 8 OLMo-1B-hf checkpoint weights in parallel.

Uses HF_HUB_ENABLE_HF_TRANSFER=1 + multiprocessing so all 8 revisions
download concurrently instead of sequentially.

Usage:
    HF_HUB_ENABLE_HF_TRANSFER=1 python src/prefetch_olmo_checkpoints.py
"""

import os
import sys
from multiprocessing import Pool

# Enable hf_transfer before any huggingface_hub imports
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download

MODEL_ID = "allenai/OLMo-1B-hf"

REVISIONS = [
    "step1000-tokens4B",
    "step80000-tokens335B",
    "step110000-tokens461B",
    "step330000-tokens1383B",
    "step460000-tokens1928B",
    "step620000-tokens2599B",
    "step720000-tokens3018B",
    "step738020-tokens3094B",
]

# Only need model weights + tokenizer/config; skip optimizer states if present
IGNORE = ["optimizer.pt", "*.pt", "training_config.yaml"]


def fetch(revision: str) -> str:
    try:
        path = snapshot_download(
            MODEL_ID,
            revision=revision,
            ignore_patterns=IGNORE,
        )
        print(f"✅  {revision}  →  {path}", flush=True)
        return path
    except Exception as e:
        print(f"❌  {revision}  →  {e}", flush=True)
        return ""


if __name__ == "__main__":
    workers = min(8, len(REVISIONS))
    print(f"Prefetching {len(REVISIONS)} OLMo-1B checkpoints with {workers} workers...")
    with Pool(workers) as pool:
        results = pool.map(fetch, REVISIONS)
    failed = [r for r in zip(REVISIONS, results) if not r[1]]
    if failed:
        print(f"\n⚠  {len(failed)} failed: {[f[0] for f in failed]}")
        sys.exit(1)
    print("\n✅ All checkpoints cached.")
