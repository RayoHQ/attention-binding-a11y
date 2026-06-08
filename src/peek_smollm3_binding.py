"""Quick peek at SmolLM3-3B EB* trajectory across 8 lifecycle checkpoints."""
import json
from pathlib import Path
import numpy as np

BASE = Path("data/results/binding_smollm3")
CHECKPOINTS = [
    "step40k", "step120k", "step400k", "step800k",
    "step1200k", "step1600k", "step2400k", "step3440k",
]

eb_by_ck = {}
for ck in CHECKPOINTS:
    f = BASE / f"smollm3_{ck}_binding_smollm3.jsonl"
    if not f.exists():
        continue
    records = [json.loads(l) for l in open(f)]
    eb_by_ck[ck] = {r["term"]: r["eb_star"] for r in records}

# Per-term EB* across checkpoints
terms = sorted(set(t for d in eb_by_ck.values() for t in d))
print(f"{'Term':<28} " + "  ".join(f"{ck:>9}" for ck in CHECKPOINTS))
print("-" * (28 + 12 * len(CHECKPOINTS)))
for term in terms:
    row = [eb_by_ck.get(ck, {}).get(term, float("nan")) for ck in CHECKPOINTS]
    vals = "  ".join(f"{v:9.3f}" if not np.isnan(v) else f"{'---':>9}" for v in row)
    trend = "↑" if (not np.isnan(row[0]) and not np.isnan(row[-1]) and row[-1] > row[0]) else "↓"
    print(f"{term:<28} {vals}  {trend}")

# Mean EB* per checkpoint
print()
print(f"{'MEAN':<28} ", end="")
for ck in CHECKPOINTS:
    vals = [v for v in eb_by_ck.get(ck, {}).values()]
    mean = np.mean(vals) if vals else float("nan")
    print(f"{mean:9.3f}  ", end="")
print()
