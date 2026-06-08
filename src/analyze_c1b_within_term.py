"""C1-B: Within-term temporal precedence analysis.

Tests whether EB*(t, k) predicts Beh(t, k+1) better than Beh(t, k) predicts EB*(t, k+1)
for each term t independently. No between-term variance required.

Uses all available lifecycle data:
  - binding_expanded_100 + behavioral_expanded_100  (9 terms Set B)
  - binding_tier123 + behavioral_tier123            (21 terms tier123)
  - binding + behavioral                            (OLMo: 9 terms Set B)

Population test: binomial test over all available terms per model.
H1: fraction of terms where r_forward > r_backward > 0.5

Outputs:
  data/results/c1b/{model}_within_term_lead.csv
  data/results/c1b/{model}_population_test.json
"""

import json
import csv
from pathlib import Path
from scipy.stats import pearsonr, binomtest
import numpy as np

BASE = Path("data/results")
OUT_DIR = Path("data/results/c1b")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS_PYTHIA = [
    "step0", "step15000", "step30000", "step60000",
    "step90000", "step120000", "step140000", "step143000",
]
CHECKPOINTS_OLMO = [
    "step0", "step15k", "step30k", "step60k",
    "step90k", "step120k", "step140k", "step143k",
]
CHECKPOINTS_CRFM = [
    "checkpoint-0", "checkpoint-1000", "checkpoint-5000", "checkpoint-10000",
    "checkpoint-50000", "checkpoint-100000", "checkpoint-200000", "checkpoint-400000",
]
CHECKPOINTS_SMOLLM3 = [
    "step40k", "step120k", "step400k", "step800k",
    "step1200k", "step1600k", "step2400k", "step3440k",
]

PYTHIA_MODELS = {
    "pythia-160m": {
        "binding_dirs":    ["binding_expanded_100", "binding_tier123", "binding_wave2"],
        "behavioral_dirs": ["behavioral_expanded_100", "behavioral_tier123", "behavioral_wave2"],
        "binding_suffixes":    ["_binding_100.jsonl", "_binding_tier123.jsonl", "_binding_wave2.jsonl"],
        "behavioral_suffixes": ["_behavioral_100.jsonl", "_behavioral_tier123.jsonl", "_behavioral_wave2.jsonl"],
        "prefix": "160m",
        "checkpoints": CHECKPOINTS_PYTHIA,
    },
    "pythia-1b": {
        "binding_dirs":    ["binding_expanded_100", "binding_tier123", "binding_wave2"],
        "behavioral_dirs": ["behavioral_expanded_100", "behavioral_tier123", "behavioral_wave2"],
        "binding_suffixes":    ["_binding_100.jsonl", "_binding_tier123.jsonl", "_binding_wave2.jsonl"],
        "behavioral_suffixes": ["_behavioral_100.jsonl", "_behavioral_tier123.jsonl", "_behavioral_wave2.jsonl"],
        "prefix": "1b",
        "checkpoints": CHECKPOINTS_PYTHIA,
    },
    "pythia-2.8b": {
        "binding_dirs":    ["binding_expanded_100", "binding_tier123", "binding_wave2"],
        "behavioral_dirs": ["behavioral_expanded_100", "behavioral_tier123", "behavioral_wave2"],
        "binding_suffixes":    ["_binding_100.jsonl", "_binding_tier123.jsonl", "_binding_wave2.jsonl"],
        "behavioral_suffixes": ["_behavioral_100.jsonl", "_behavioral_tier123.jsonl", "_behavioral_wave2.jsonl"],
        "prefix": "2.8b",
        "checkpoints": CHECKPOINTS_PYTHIA,
    },
    "olmo-1b": {
        "binding_dirs":    ["binding"],
        "behavioral_dirs": ["behavioral_olmo"],
        "binding_suffixes":    ["_binding.jsonl"],
        "behavioral_suffixes": ["_behavioral.jsonl"],
        "prefix": "olmo_1b",
        "checkpoints": CHECKPOINTS_OLMO,
    },
}

# CRFM GPT-2 Small: 5 seeds — generated dynamically
CRFM_MODELS = {
    f"crfm-gpt2-sm-x{s}": {
        "binding_dirs":    ["binding_crfm"],
        "behavioral_dirs": ["behavioral_crfm"],
        "binding_suffixes":    ["_binding_crfm.jsonl"],
        "behavioral_suffixes": ["_behavioral_crfm.jsonl"],
        "prefix": f"seed{s}",
        "checkpoints": CHECKPOINTS_CRFM,
    }
    for s in range(1, 6)
}

SMOLLM3_MODELS = {
    "smollm3-3b": {
        "binding_dirs":    ["binding_smollm3"],
        "behavioral_dirs": ["behavioral_smollm3"],
        "binding_suffixes":    ["_binding_smollm3.jsonl"],
        "behavioral_suffixes": ["_behavioral_smollm3.jsonl"],
        "prefix": "smollm3",
        "checkpoints": CHECKPOINTS_SMOLLM3,
    },
}

# 45-term canonical variants (output to *_45 directories)
CRFM_MODELS_45 = {
    f"crfm-gpt2-sm-x{s}-45": {
        "binding_dirs":    ["binding_crfm_45"],
        "behavioral_dirs": ["behavioral_crfm_45"],
        "binding_suffixes":    ["_binding_crfm.jsonl"],
        "behavioral_suffixes": ["_behavioral_crfm.jsonl"],
        "prefix": f"seed{s}",
        "checkpoints": CHECKPOINTS_CRFM,
    }
    for s in range(1, 6)
}

SMOLLM3_MODELS_45 = {
    "smollm3-3b-45": {
        "binding_dirs":    ["binding_smollm3_45"],
        "behavioral_dirs": ["behavioral_smollm3_45"],
        "binding_suffixes":    ["_binding_smollm3.jsonl"],
        "behavioral_suffixes": ["_behavioral_smollm3.jsonl"],
        "prefix": "smollm3",
        "checkpoints": CHECKPOINTS_SMOLLM3,
    },
}

OLMO_MODELS_45 = {
    "olmo-1b-45": {
        "binding_dirs":    ["binding_olmo_45"],
        "behavioral_dirs": ["behavioral_olmo_45"],
        "binding_suffixes":    ["_binding.jsonl"],
        "behavioral_suffixes": ["_behavioral.jsonl"],
        "prefix": "olmo_1b",
        "checkpoints": CHECKPOINTS_OLMO,
    },
}

ALL_MODELS = {**PYTHIA_MODELS, **CRFM_MODELS, **SMOLLM3_MODELS}
ALL_MODELS_45 = {**CRFM_MODELS_45, **SMOLLM3_MODELS_45, **OLMO_MODELS_45}


def load_eb_star_per_term(binding_dir: str, prefix: str, ck: str, suffix: str) -> dict[str, float]:
    """Load mean EB* per term from a binding file."""
    path = BASE / binding_dir / f"{prefix}_{ck}{suffix}"
    if not path.exists():
        return {}
    records = [json.loads(l) for l in open(path)]
    term_vals: dict[str, list[float]] = {}
    for r in records:
        term_vals.setdefault(r["term"], []).append(r["eb_star"])
    return {t: float(np.mean(v)) for t, v in term_vals.items()}


def load_beh_per_term(behavioral_dir: str, prefix: str, ck: str, suffix: str) -> dict[str, float]:
    """Load mean behavioral score per term from a behavioral file."""
    path = BASE / behavioral_dir / f"{prefix}_{ck}{suffix}"
    if not path.exists():
        return {}
    records = [json.loads(l) for l in open(path)]
    term_vals: dict[str, list[float]] = {}
    for r in records:
        term_vals.setdefault(r["term"], []).append(r["behavioral_score"])
    return {t: float(np.mean(v)) for t, v in term_vals.items()}


def build_term_series(model_key: str, cfg: dict) -> dict[str, dict]:
    """Build {term: {eb_series: [...], beh_series: [...]}} across all checkpoints."""
    checkpoints = cfg["checkpoints"]
    eb_all: dict[str, list] = {}
    beh_all: dict[str, list] = {}

    for bd, bs, bvd, bvs in zip(
        cfg["binding_dirs"], cfg["binding_suffixes"],
        cfg["behavioral_dirs"], cfg["behavioral_suffixes"]
    ):
        for ck in checkpoints:
            eb = load_eb_star_per_term(bd, cfg["prefix"], ck, bs)
            beh = load_beh_per_term(bvd, cfg["prefix"], ck, bvs)
            shared_terms = set(eb) & set(beh)
            for t in shared_terms:
                if t not in eb_all:
                    eb_all[t] = [None] * len(checkpoints)
                    beh_all[t] = [None] * len(checkpoints)
                ck_idx = checkpoints.index(ck)
                eb_all[t][ck_idx] = eb[t]
                beh_all[t][ck_idx] = beh[t]

    # Only keep terms with complete series
    complete = {}
    for t in eb_all:
        if all(v is not None for v in eb_all[t]) and all(v is not None for v in beh_all[t]):
            complete[t] = {
                "eb_series": eb_all[t],
                "beh_series": beh_all[t],
            }
    return complete


def compute_lead_indicator(eb_series: list[float], beh_series: list[float]) -> dict | None:
    """Compute 1-step forward lag correlation for one term. Returns None if series is constant."""
    if np.std(eb_series) < 1e-9 or np.std(beh_series) < 1e-9:
        return None  # constant series — pearsonr undefined
    eb = np.array(eb_series, dtype=float)
    beh = np.array(beh_series, dtype=float)
    r_forward, _ = pearsonr(eb[:-1], beh[1:])
    r_backward, _ = pearsonr(beh[:-1], eb[1:])
    if np.isnan(r_forward) or np.isnan(r_backward):
        return None
    lead = int(r_forward > r_backward)
    return {
        "r_forward": round(float(r_forward), 4),
        "r_backward": round(float(r_backward), 4),
        "lead_diff": round(float(r_forward - r_backward), 4),
        "lead_indicator": lead,
    }


def run_model(model_key: str, cfg: dict):
    print(f"\n{'='*60}")
    print(f"C1-B: {model_key}")
    print(f"{'='*60}")

    series = build_term_series(model_key, cfg)
    if not series:
        print(f"  ⚠️  No complete term series found — skipping")
        return

    rows = []
    skipped = []
    for term in sorted(series):
        eb = series[term]["eb_series"]
        beh = series[term]["beh_series"]
        result = compute_lead_indicator(eb, beh)
        if result is None:
            skipped.append(term)
            print(f"  {'(skip constant)':30s}  {term}")
            continue
        row = {"model": model_key, "term": term, **result}
        rows.append(row)
        flag = "→ EB* leads" if result["lead_indicator"] else "← Beh leads"
        print(f"  {term:<30s}  r_fwd={result['r_forward']:+.3f}  r_bck={result['r_backward']:+.3f}  {flag}")

    # Population test
    n_terms = len(rows)
    n_lead = sum(r["lead_indicator"] for r in rows)
    frac = n_lead / n_terms
    btest = binomtest(n_lead, n_terms, p=0.5, alternative="greater")
    p_val = btest.pvalue
    mean_diff = float(np.mean([r["lead_diff"] for r in rows]))

    print(f"\n  Population: {n_lead}/{n_terms} terms where EB* leads ({frac:.1%})")
    print(f"  Binomial p = {p_val:.4f}  (H1: frac > 0.5)")
    print(f"  Mean r_forward - r_backward = {mean_diff:+.4f}")

    # Save per-term CSV
    csv_path = OUT_DIR / f"{model_key.replace('.','')}_within_term_lead.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Save population test JSON
    pop_path = OUT_DIR / f"{model_key.replace('.','')}_population_test.json"
    with open(pop_path, "w") as f:
        json.dump({
            "model": model_key,
            "n_terms": n_terms,
            "n_lead": n_lead,
            "lead_fraction": round(frac, 4),
            "binomial_p": round(p_val, 6),
            "mean_lead_diff": round(mean_diff, 4),
            "supported": bool(frac >= 0.6 and p_val < 0.05),
        }, f, indent=2)

    print(f"  ✅ Saved: {csv_path.name}, {pop_path.name}")
    return rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all",
                        help="Model key, group (pythia/crfm/smollm3/olmo), or 'all'")
    args = parser.parse_args()

    if args.model == "all":
        targets = ALL_MODELS
    elif args.model == "all45":
        targets = ALL_MODELS_45
    elif args.model == "pythia":
        targets = PYTHIA_MODELS
    elif args.model == "crfm":
        targets = CRFM_MODELS
    elif args.model == "crfm45":
        targets = CRFM_MODELS_45
    elif args.model == "smollm3":
        targets = SMOLLM3_MODELS
    elif args.model == "smollm345":
        targets = SMOLLM3_MODELS_45
    elif args.model == "olmo45":
        targets = OLMO_MODELS_45
    elif args.model in ALL_MODELS:
        targets = {args.model: ALL_MODELS[args.model]}
    elif args.model in ALL_MODELS_45:
        targets = {args.model: ALL_MODELS_45[args.model]}
    else:
        raise ValueError(f"Unknown model '{args.model}'. Valid: {list(ALL_MODELS.keys())}")


    print("\n=== C1-B: Within-Term Temporal Precedence Analysis ===")
    print("H1: EB* rise precedes behavioral emergence in >50% of terms\n")

    all_pop = []
    for model_key, cfg in targets.items():
        rows = run_model(model_key, cfg)
        if rows:
            n_lead = sum(r["lead_indicator"] for r in rows)
            all_pop.append(f"  {model_key}: {n_lead}/{len(rows)} lead ({n_lead/len(rows):.0%})")

    print("\n=== SUMMARY ===")
    for line in all_pop:
        print(line)
