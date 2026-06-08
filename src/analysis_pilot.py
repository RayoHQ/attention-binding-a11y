"""1D: Go/No-Go analysis — correlate binding with behavior."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

# Config
BEHAVIORAL_DIR = Path("data/results/behavioral")
BINDING_DIR = Path("data/results/binding")
CHECKPOINTS = [
    "step0", "step15000", "step30000", "step60000",
    "step90000", "step120000", "step140000", "step143000",
]


def load_results(model_size: str):
    """Load behavioral and binding results for a model size."""
    behavioral = {}
    binding = {}

    # Load behavioral
    for ckpt in CHECKPOINTS:
        file = BEHAVIORAL_DIR / f"{model_size}_{ckpt}_behavioral.jsonl"
        if not file.exists():
            continue
        with open(file) as f:
            for line in f:
                r = json.loads(line)
                key = (ckpt, r["term"], r["prompt_id"])
                behavioral[key] = r["score"]

    # Load binding
    for ckpt in CHECKPOINTS:
        file = BINDING_DIR / f"{model_size}_{ckpt}_binding.jsonl"
        if not file.exists():
            continue
        with open(file) as f:
            for line in f:
                r = json.loads(line)
                key = (r["checkpoint"], r["term"], r["prompt_id"])
                binding[key] = r["eb_star"]

    return behavioral, binding


def compute_correlation(behavioral: dict, binding: dict):
    """Compute Spearman correlation EB* vs accuracy score."""
    # Join on common keys
    common_keys = set(behavioral.keys()) & set(binding.keys())

    if len(common_keys) < 10:
        print(f"⚠️  Only {len(common_keys)} common samples — need more data")
        return None

    beh_scores = [behavioral[k] for k in common_keys]
    eb_scores = [binding[k] for k in common_keys]

    # Spearman correlation
    corr, pvalue = spearmanr(eb_scores, beh_scores)

    print(f"\n=== Correlation Analysis ({len(common_keys)} samples) ===")
    print(f"Spearman r = {corr:.3f} (p = {pvalue:.4f})")

    return corr, pvalue, eb_scores, beh_scores


def check_quadrant(eb_scores: list, beh_scores: list):
    """Check for high-EB*/low-accuracy quadrant (unlockability candidates)."""
    eb_high_thresh = np.percentile(eb_scores, 75)
    beh_low_thresh = np.percentile(beh_scores, 25)

    high_eb_low_acc = [
        (eb, beh)
        for eb, beh in zip(eb_scores, beh_scores)
        if eb >= eb_high_thresh and beh <= beh_low_thresh
    ]

    print(f"\n=== Quadrant Check ===")
    print(f"High EB* threshold (75th %ile): {eb_high_thresh:.3f}")
    print(f"Low accuracy threshold (25th %ile): {beh_low_thresh:.3f}")
    print(f"High-EB*/Low-acc samples: {len(high_eb_low_acc)}")

    return len(high_eb_low_acc) > 0


def go_no_go_decision(corr: float, quadrant_non_empty: bool):
    """Make Go/No-Go decision."""
    print(f"\n{'=' * 50}")
    print("GO / NO-GO DECISION")
    print(f"{'=' * 50}")

    criteria = []

    # Criterion 1: Correlation
    corr_pass = corr is not None and corr > 0.3
    criteria.append(
        ("Spearman r > 0.3", corr_pass, f"r = {corr:.3f}" if corr else "N/A")
    )

    # Criterion 2: Directional consistency
    sign_pass = corr is not None and corr > 0
    criteria.append(
        ("Positive correlation", sign_pass, "positive" if sign_pass else "negative/zero")
    )

    # Criterion 3: Quadrant
    criteria.append(
        ("High-EB*/Low-acc exists", quadrant_non_empty, "yes" if quadrant_non_empty else "no")
    )

    for name, passed, detail in criteria:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name} ({detail})")

    all_pass = all(p for _, p, _ in criteria)

    print(f"\n{'=' * 50}")
    if all_pass:
        print("✅ GO: Proceed to full 1b/2.8b sweep")
        print("Next: Run 1C for all 160m checkpoints, then 1b, then 2.8b")
    elif corr is not None and corr > 0:
        print("⚠️  CONDITIONAL GO: Weak but positive signal")
        print("Consider: Run more checkpoints, or proceed with caveats")
    else:
        print("❌ NO-GO: No detectable signal")
        print("Consider: Debug binding computation, or pivot to reduced scope")
    print(f"{'=' * 50}")

    return all_pass


if __name__ == "__main__":
    import sys

    model_size = sys.argv[1] if len(sys.argv) > 1 else "160m"

    print(f"Analyzing {model_size}...")

    behavioral, binding = load_results(model_size)
    result = compute_correlation(behavioral, binding)

    if result:
        corr, pvalue, eb_scores, beh_scores = result
        quadrant_ok = check_quadrant(eb_scores, beh_scores)
        go_no_go_decision(corr, quadrant_ok)
    else:
        print("❌ Cannot compute correlation — missing data")
