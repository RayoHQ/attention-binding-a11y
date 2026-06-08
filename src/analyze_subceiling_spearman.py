"""Sub-ceiling Spearman analysis.

Tests whether the 2.8B rho_late = +0.270 is an artifact of ceiling effects
by restricting the late-window Spearman to terms below behavioral score thresholds.

Hypothesis: if positive rho_late is ceiling-driven, filtering to sub-ceiling
terms should reduce or reverse it (revealing decoupling that was masked).

Runs on all Pythia models for comparison, using the existing C4-B series data.
No new model inference — reuses binding + behavioral data from Phase 1.

Outputs:
    data/results/c4b/subceiling_spearman_results.json
"""

import json
import csv
from pathlib import Path
from scipy.stats import spearmanr
import numpy as np

BASE = Path("data/results")
OUT_DIR = Path("data/results/c4b")

CHECKPOINTS_PYTHIA = [
    "step0", "step15000", "step30000", "step60000",
    "step90000", "step120000", "step140000", "step143000",
]
LATE_START = 4   # checkpoints 4-7 = steps 90k-143k
THRESHOLDS = [1.00, 0.90, 0.80, 0.70, 0.60]  # successively tighter ceiling filters

PYTHIA_MODELS = {
    "pythia-2.8b": {
        "binding_dirs":    ["binding_expanded_100", "binding_tier123"],
        "behavioral_dirs": ["behavioral_expanded_100", "behavioral_tier123"],
        "binding_suffixes":    ["_binding_100.jsonl",   "_binding_tier123.jsonl"],
        "behavioral_suffixes": ["_behavioral_100.jsonl","_behavioral_tier123.jsonl"],
        "prefix": "2.8b",
    },
    "pythia-1b": {
        "binding_dirs":    ["binding_expanded_100", "binding_tier123"],
        "behavioral_dirs": ["behavioral_expanded_100", "behavioral_tier123"],
        "binding_suffixes":    ["_binding_100.jsonl",   "_binding_tier123.jsonl"],
        "behavioral_suffixes": ["_behavioral_100.jsonl","_behavioral_tier123.jsonl"],
        "prefix": "1b",
    },
    "pythia-160m": {
        "binding_dirs":    ["binding_expanded_100", "binding_tier123"],
        "behavioral_dirs": ["behavioral_expanded_100", "behavioral_tier123"],
        "binding_suffixes":    ["_binding_100.jsonl",   "_binding_tier123.jsonl"],
        "behavioral_suffixes": ["_behavioral_100.jsonl","_behavioral_tier123.jsonl"],
        "prefix": "160m",
    },
}


def load_mean_per_term(result_dir, prefix, ck, suffix, field):
    path = BASE / result_dir / f"{prefix}_{ck}{suffix}"
    if not path.exists():
        return {}
    records = [json.loads(l) for l in open(path)]
    vals = {}
    for r in records:
        vals.setdefault(r["term"], []).append(r[field])
    return {t: float(np.mean(v)) for t, v in vals.items()}


def build_term_series(cfg):
    eb_all = {}
    beh_all = {}
    for bd, bs, bvd, bvs in zip(
        cfg["binding_dirs"], cfg["binding_suffixes"],
        cfg["behavioral_dirs"], cfg["behavioral_suffixes"]
    ):
        for ck in CHECKPOINTS_PYTHIA:
            eb  = load_mean_per_term(bd,  cfg["prefix"], ck, bs,  "eb_star")
            beh = load_mean_per_term(bvd, cfg["prefix"], ck, bvs, "behavioral_score")
            for t in set(eb) & set(beh):
                if t not in eb_all:
                    eb_all[t]  = [None] * len(CHECKPOINTS_PYTHIA)
                    beh_all[t] = [None] * len(CHECKPOINTS_PYTHIA)
                idx = CHECKPOINTS_PYTHIA.index(ck)
                eb_all[t][idx]  = eb[t]
                beh_all[t][idx] = beh[t]
    return {
        t: {"eb": eb_all[t], "beh": beh_all[t]}
        for t in eb_all
        if all(v is not None for v in eb_all[t]) and all(v is not None for v in beh_all[t])
    }


def window_spearman(values_a, values_b, start, end):
    a = np.array(values_a[start:end], dtype=float)
    b = np.array(values_b[start:end], dtype=float)
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return None
    rho, _ = spearmanr(a, b)
    return float(rho) if not np.isnan(rho) else None


def analyze_model(model_key, cfg):
    print(f"\n{'='*65}")
    print(f"  {model_key}  —  Sub-ceiling Spearman")
    print(f"{'='*65}")

    series = build_term_series(cfg)
    if not series:
        print("  ⚠  No data found.")
        return {}

    # Compute per-term mean late-window behavioral score
    term_mean_beh_late = {}
    for t, s in series.items():
        late_beh = [s["beh"][i] for i in range(LATE_START, len(CHECKPOINTS_PYTHIA))]
        term_mean_beh_late[t] = float(np.mean(late_beh))

    results = {}
    print(f"\n  {'Threshold':<12}  {'N terms':<10}  {'rho_late':<12}  {'Δ vs full'}")
    print(f"  {'-'*55}")

    rho_full = None
    for thresh in THRESHOLDS:
        subset = {
            t: s for t, s in series.items()
            if term_mean_beh_late[t] < thresh
        }
        if len(subset) < 4:
            print(f"  <{thresh:.2f}        {'<4 terms':<10}  {'—'}")
            continue

        rhos = []
        for t, s in subset.items():
            rho = window_spearman(s["eb"], s["beh"], LATE_START, len(CHECKPOINTS_PYTHIA))
            if rho is not None:
                rhos.append((t, rho))

        if len(rhos) < 4:
            print(f"  <{thresh:.2f}        {len(subset):<10}  {'<4 valid'}")
            continue

        # Pooled late-window Spearman across all sub-ceiling terms
        all_eb_late, all_beh_late = [], []
        for t, s in subset.items():
            eb_w  = s["eb"][LATE_START:]
            beh_w = s["beh"][LATE_START:]
            all_eb_late.extend(eb_w)
            all_beh_late.extend(beh_w)

        rho_pooled, pval = spearmanr(all_eb_late, all_beh_late)
        rho_pooled = round(float(rho_pooled), 4)
        pval       = round(float(pval), 4)

        if thresh == 1.00:
            rho_full = rho_pooled
            delta_str = "(full set)"
        else:
            delta = rho_pooled - rho_full if rho_full is not None else 0
            delta_str = f"{delta:+.4f}"

        n_excl = sum(1 for v in term_mean_beh_late.values() if v >= thresh)
        print(f"  <{thresh:.2f}        {len(subset):<10}  {rho_pooled:+.4f} (p={pval:.3f})  {delta_str}  [{n_excl} ceiling terms excluded]")

        results[f"thresh_{thresh}"] = {
            "threshold": thresh,
            "n_terms": len(subset),
            "n_excluded_ceiling": sum(1 for v in term_mean_beh_late.values() if v >= thresh),
            "rho_late_pooled": rho_pooled,
            "pval": pval,
            "per_term": [{"term": t, "rho_late": r} for t, r in sorted(rhos, key=lambda x: x[1])],
        }

    # Report which terms are ceiling (mean late beh >= 0.80)
    ceiling_terms = sorted([t for t, v in term_mean_beh_late.items() if v >= 0.80])
    print(f"\n  Ceiling terms (mean late beh ≥ 0.80)  [{len(ceiling_terms)}]:")
    for t in ceiling_terms:
        print(f"    {t:<35}  beh={term_mean_beh_late[t]:.3f}")

    subceil_terms = sorted([t for t, v in term_mean_beh_late.items() if v < 0.80])
    print(f"\n  Sub-ceiling terms (< 0.80)  [{len(subceil_terms)}]:")
    for t in subceil_terms:
        print(f"    {t:<35}  beh={term_mean_beh_late[t]:.3f}")

    results["ceiling_terms"]    = ceiling_terms
    results["subceil_terms"]    = subceil_terms
    results["term_mean_beh_late"] = {t: round(v, 4) for t, v in term_mean_beh_late.items()}
    return results


if __name__ == "__main__":
    all_results = {}

    for model_key, cfg in PYTHIA_MODELS.items():
        all_results[model_key] = analyze_model(model_key, cfg)

    out_file = OUT_DIR / "subceiling_spearman_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\n✅  Results saved → {out_file}")
