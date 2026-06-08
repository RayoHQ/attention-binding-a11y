"""C4-B: Within-term decoupling analysis (redesigned, all available terms).

For each term t, splits the 8-checkpoint lifecycle into early (ck0..3) and late (ck4..7)
windows and computes within-term Spearman correlation in each window independently.

decouple(t) = 1 if rho_early(t) > 0 AND rho_late(t) <= 0

Prediction: decoupling fraction increases with model scale (160M < 1B < 2.8B).
Connects to C5: heads are causally necessary at 160M (coupled), interfering at 2.8B (decoupled).

No new data — reuses all binding + behavioral lifecycle data from Phase 1.

Outputs:
  data/results/c4b/{model}_within_term_decoupling.csv
  data/results/c4b/{model}_decoupling_summary.json
"""

import json
import csv
from pathlib import Path
from scipy.stats import spearmanr
import numpy as np

BASE = Path("data/results")
OUT_DIR = Path("data/results/c4b")
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

MODELS = {
    "pythia-160m": {
        "binding_dirs":    ["binding_expanded_100", "binding_tier123"],
        "behavioral_dirs": ["behavioral_expanded_100", "behavioral_tier123"],
        "binding_suffixes":    ["_binding_100.jsonl", "_binding_tier123.jsonl"],
        "behavioral_suffixes": ["_behavioral_100.jsonl", "_behavioral_tier123.jsonl"],
        "prefix": "160m",
        "checkpoints": CHECKPOINTS_PYTHIA,
    },
    "pythia-1b": {
        "binding_dirs":    ["binding_expanded_100", "binding_tier123"],
        "behavioral_dirs": ["behavioral_expanded_100", "behavioral_tier123"],
        "binding_suffixes":    ["_binding_100.jsonl", "_binding_tier123.jsonl"],
        "behavioral_suffixes": ["_behavioral_100.jsonl", "_behavioral_tier123.jsonl"],
        "prefix": "1b",
        "checkpoints": CHECKPOINTS_PYTHIA,
    },
    "pythia-2.8b": {
        "binding_dirs":    ["binding_expanded_100", "binding_tier123"],
        "behavioral_dirs": ["behavioral_expanded_100", "behavioral_tier123"],
        "binding_suffixes":    ["_binding_100.jsonl", "_binding_tier123.jsonl"],
        "behavioral_suffixes": ["_behavioral_100.jsonl", "_behavioral_tier123.jsonl"],
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

ALL_MODELS = {**MODELS, **CRFM_MODELS, **SMOLLM3_MODELS}

# 45-term canonical variants
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

ALL_MODELS_45 = {**CRFM_MODELS_45, **SMOLLM3_MODELS_45, **OLMO_MODELS_45}


def load_mean_per_term(result_dir: str, prefix: str, ck: str, suffix: str, field: str) -> dict[str, float]:
    path = BASE / result_dir / f"{prefix}_{ck}{suffix}"
    if not path.exists():
        return {}
    records = [json.loads(l) for l in open(path)]
    vals: dict[str, list[float]] = {}
    for r in records:
        vals.setdefault(r["term"], []).append(r[field])
    return {t: float(np.mean(v)) for t, v in vals.items()}


def build_term_series(cfg: dict) -> dict[str, dict]:
    checkpoints = cfg["checkpoints"]
    eb_all: dict[str, list] = {}
    beh_all: dict[str, list] = {}

    for bd, bs, bvd, bvs in zip(
        cfg["binding_dirs"], cfg["binding_suffixes"],
        cfg["behavioral_dirs"], cfg["behavioral_suffixes"]
    ):
        for ck in checkpoints:
            eb = load_mean_per_term(bd, cfg["prefix"], ck, bs, "eb_star")
            beh = load_mean_per_term(bvd, cfg["prefix"], ck, bvs, "behavioral_score")
            for t in set(eb) & set(beh):
                if t not in eb_all:
                    eb_all[t] = [None] * len(checkpoints)
                    beh_all[t] = [None] * len(checkpoints)
                idx = checkpoints.index(ck)
                eb_all[t][idx] = eb[t]
                beh_all[t][idx] = beh[t]

    return {
        t: {"eb": eb_all[t], "beh": beh_all[t]}
        for t in eb_all
        if all(v is not None for v in eb_all[t]) and all(v is not None for v in beh_all[t])
    }


def window_spearman(eb: list, beh: list, start: int, end: int) -> float | None:
    eb_w = np.array(eb[start:end], dtype=float)
    beh_w = np.array(beh[start:end], dtype=float)
    if np.std(eb_w) < 1e-9 or np.std(beh_w) < 1e-9:
        return None
    rho, _ = spearmanr(eb_w, beh_w)
    return float(rho) if not np.isnan(rho) else None


def run_model(model_key: str, cfg: dict):
    print(f"\n{'='*60}")
    print(f"C4-B: {model_key}")
    print(f"{'='*60}")
    print(f"  {'Term':<30s}  rho_early  rho_late   decouple")
    print(f"  {'-'*60}")

    series = build_term_series(cfg)
    if not series:
        print(f"  ⚠️  No complete series — skipping")
        return

    rows = []
    for term in sorted(series):
        eb = series[term]["eb"]
        beh = series[term]["beh"]
        rho_e = window_spearman(eb, beh, 0, 4)   # ck0..ck3 (early)
        rho_l = window_spearman(eb, beh, 4, 8)   # ck4..ck7 (late)
        if rho_e is None or rho_l is None:
            print(f"  {'(skip constant)':30s}  {term}")
            continue
        decouple_strict = int(rho_e > 0 and rho_l <= 0)
        decouple_attenuate = int(rho_e > 0 and rho_l < 0.5 * rho_e)
        rows.append({
            "model": model_key,
            "term": term,
            "rho_early": round(rho_e, 4),
            "rho_late": round(rho_l, 4),
            "decouple_strict": decouple_strict,
            "decouple_attenuate": decouple_attenuate,
        })
        decouple = decouple_strict  # keep flag for display
        flag = "✓ DECOUPLE" if decouple else "  coupled " if rho_e > 0 else "  neg/zero"
        print(f"  {term:<30s}  {rho_e:+.3f}     {rho_l:+.3f}     {flag}")

    n = len(rows)
    n_coupled   = sum(1 for r in rows if r["rho_early"] > 0)
    n_strict    = sum(r["decouple_strict"] for r in rows)
    n_attenuate = sum(r["decouple_attenuate"] for r in rows)
    mean_rho_e  = float(np.mean([r["rho_early"] for r in rows]))
    mean_rho_l  = float(np.mean([r["rho_late"]  for r in rows]))

    print(f"\n  Early coupling     (rho_e > 0):          {n_coupled}/{n} ({n_coupled/n:.1%})")
    print(f"  Strict decouple    (e>0, l≤0):           {n_strict}/{n} ({n_strict/n:.1%})")
    print(f"  Attenuation        (e>0, l<0.5×e):       {n_attenuate}/{n} ({n_attenuate/n:.1%})")
    print(f"  Mean rho_early={mean_rho_e:+.3f}   mean rho_late={mean_rho_l:+.3f}")

    csv_path = OUT_DIR / f"{model_key.replace('.','')}_within_term_decoupling.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary_path = OUT_DIR / f"{model_key.replace('.','')}_decoupling_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "model": model_key,
            "n_terms": n,
            "n_early_coupled": n_coupled,
            "n_decouple_strict": n_strict,
            "n_decouple_attenuate": n_attenuate,
            "frac_decouple_strict": round(n_strict / n, 4),
            "frac_decouple_attenuate": round(n_attenuate / n, 4),
            "mean_rho_early": round(mean_rho_e, 4),
            "mean_rho_late": round(mean_rho_l, 4),
        }, f, indent=2)

    print(f"  ✅ Saved: {csv_path.name}")
    return rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all")
    args = parser.parse_args()

    if args.model == "all":
        targets = MODELS
    elif args.model == "all-models":
        targets = ALL_MODELS
    elif args.model == "all45":
        targets = ALL_MODELS_45
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

    print("\n=== C4-B: Within-Term Decoupling Analysis ===")
    print("H: decoupling fraction increases with model scale\n")

    summary = []
    for model_key, cfg in targets.items():
        rows = run_model(model_key, cfg)
        if rows:
            n_dec = sum(r["decouple_strict"] for r in rows)
            summary.append((model_key, n_dec, len(rows)))

    print("\n=== DECOUPLING SUMMARY ===")
    for m, n_dec, n in summary:
        print(f"  {m:<20s}  {n_dec}/{n} decouple ({n_dec/n:.0%})")
