"""Full cross-model result matrix summary.

Reads all C1-B population_test.json and C4-B decoupling CSV files,
plus causal C5 JSON files, and prints a unified table.

Usage:
    python src/summarize_full_matrix.py
    python src/summarize_full_matrix.py --set 45   # only 45-term results
    python src/summarize_full_matrix.py --set 9    # only 9-term results
"""

import argparse
import csv
import json
from pathlib import Path

BASE = Path("data/results")
C1B_DIR  = BASE / "c1b"
C4B_DIR  = BASE / "c4b"
C5_DIR   = BASE / "causal"
FS_DIR   = BASE / "few_shot_c3"


def load_c1b(model_key: str) -> dict | None:
    p = C1B_DIR / f"{model_key}_population_test.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_c4b(model_key: str) -> dict | None:
    p = C4B_DIR / f"{model_key}_within_term_decoupling.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    strict = [r for r in rows if r.get("decouple_strict", "").strip() in ("True", "1", "true")]
    early_pos = [r for r in rows if float(r.get("rho_early", 0) or 0) > 0]
    rho_lates = [float(r["rho_late"]) for r in rows if r.get("rho_late", "").strip() not in ("", "None")]
    return {
        "n_terms": len(rows),
        "n_decouple": len(strict),
        "frac_decouple": len(strict) / len(rows) if rows else 0,
        "n_early_pos": len(early_pos),
        "mean_rho_late": sum(rho_lates) / len(rho_lates) if rho_lates else 0,
    }


def load_c5(filename: str) -> dict | None:
    p = C5_DIR / filename
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    # Normalise: old format has nested 'drops' dict; new flat format has top-level keys
    if "drops" in d:
        drops = d["drops"]
        d["rec_drop_top"]  = drops.get("top_rec_drop", 0)
        d["specificity"]   = drops.get("specificity", 0)
    return d


def load_c3(filename: str) -> dict | None:
    p = FS_DIR / filename
    if not p.exists():
        return None
    rows = [json.loads(l) for l in open(p)]
    deltas = [r["delta"] for r in rows]
    pos = sum(1 for d in deltas if d > 0.0)
    return {
        "n": len(rows),
        "mean_delta": sum(deltas) / len(deltas) if deltas else 0,
        "frac_improved": pos / len(rows) if rows else 0,
    }


MODELS = [
    # (label, params, steps, c1b_key, c4b_key, c5_file, c3_file, prompt_set)
    ("Pythia-160M",      "160M",  "143k", "pythia-160m",           "pythia-160m",           "160m_step143000_c5_canonical41.json",   None,                      "41 terms"),
    ("Pythia-1B",        "1B",    "143k", "pythia-1b",             "pythia-1b",             "1b_step143000_c5_canonical41.json",     None,                      "41 terms"),
    ("Pythia-2.8B",      "2.8B",  "143k", "pythia-28b",            "pythia-28b",            "2.8b_step143000_c5_canonical41.json",   None,                      "41 terms"),
    ("OLMo-1B (9t)",     "1B",    "143k", "olmo-1b",               "olmo-1b",               None,                                    "olmo_c3_fewshot.jsonl",    "9 terms"),
    ("OLMo-1B (45t)",    "1B",    "143k", "olmo-1b-45",            "olmo-1b-45",            "olmo_step143k_c5_canonical41.json",     "olmo_c3_fewshot.jsonl",    "41 terms"),
    ("CRFM GPT-2 (9t)",  "117M",  "400k", "crfm-gpt2-sm-x1",       "crfm-gpt2-sm-x1",       None,                                    "crfm_seed1_c3_fewshot.jsonl", "9 terms"),
    ("CRFM GPT-2 (45t)", "117M",  "400k", "crfm-gpt2-sm-x1-45",    "crfm-gpt2-sm-x1-45",    "crfm_seed1_checkpoint-400000_c5_canonical41.json", "crfm_seed1_c3_fewshot.jsonl", "41 terms"),
    ("SmolLM3-3B (9t)",  "3B",    "3440k","smollm3-3b",            "smollm3-3b",            None,                                    "smollm3_c3_fewshot.jsonl", "9 terms"),
    ("SmolLM3-3B (45t)", "3B",    "3440k","smollm3-3b-45",         "smollm3-3b-45",         "smollm3_step3440k_c5_canonical41.json", "smollm3_c3_fewshot.jsonl", "41 terms"),
    ("Qwen2.5-1.5B",     "1.5B",  "final","",                       "",                      "qwen_final_c5_canonical41.json",        "qwen_final_c3_fewshot.jsonl", "41 terms (final ck)"),
]


def fmt_c1b(d):
    if d is None: return "—", "—", "—"
    pct = f"{d['lead_fraction']:.0%}"
    p   = f"{d['binomial_p']:.4f}" if d['binomial_p'] >= 0.0001 else "<0.0001"
    mld = f"{d.get('mean_lead_diff', 0):+.3f}"
    return pct, p, mld


def fmt_c4b(d):
    if d is None: return "—", "—", "—"
    pct = f"{d['frac_decouple']:.0%}"
    rho = f"{d['mean_rho_late']:+.3f}"
    n   = str(d['n_terms'])
    return pct, rho, n


def fmt_c5(d):
    if d is None: return "—", "—"
    drop = f"{d.get('rec_drop_top', 0):+.3f}"
    spec = f"{d.get('specificity', 0):+.3f}"
    return drop, spec


def fmt_c3(d):
    if d is None: return "—", "—"
    delta = f"{d['mean_delta']:+.3f}"
    frac  = f"{d['frac_improved']:.0%}"
    return delta, frac


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="all", choices=["all", "9", "45"])
    args = parser.parse_args()

    print("\n" + "="*130)
    print("FULL ANALYSIS MATRIX — Attention Binding × Accessibility Concepts")
    print("="*130)

    # ── C1-B + C4-B table ────────────────────────────────────────────────────
    print(f"\n{'Model':<22} {'Params':<7} {'Steps':<7} {'Prompts':<18}"
          f"{'C1-B%':>7} {'p':>8} {'ΔLead':>7}"
          f"{'C4-B%':>8} {'ρ_late':>8} {'N_c4':>6}"
          f"{'C5_drop':>9} {'Spec':>7}"
          f"{'C3_Δ':>8} {'C3_imp':>7}")
    print("-"*130)

    for (label, params, steps, c1b_key, c4b_key, c5_file, c3_file, pset) in MODELS:
        if args.set == "9"  and "45t" in label: continue
        if args.set == "45" and "9t"  in label: continue

        c1b = load_c1b(c1b_key) if c1b_key else None
        c4b = load_c4b(c4b_key) if c4b_key else None
        c5  = load_c5(c5_file)  if c5_file  else None
        c3  = load_c3(c3_file)  if c3_file  else None

        c1_pct, c1_p, c1_mld = fmt_c1b(c1b)
        c4_pct, c4_rho, c4_n  = fmt_c4b(c4b)
        c5_drop, c5_spec       = fmt_c5(c5)
        c3_d, c3_frac          = fmt_c3(c3)

        print(f"{label:<22} {params:<7} {steps:<7} {pset:<18}"
              f"{c1_pct:>7} {c1_p:>8} {c1_mld:>7}"
              f"{c4_pct:>8} {c4_rho:>8} {c4_n:>6}"
              f"{c5_drop:>9} {c5_spec:>7}"
              f"{c3_d:>8} {c3_frac:>7}")

    print("\nLegend:")
    print("  C1-B%   = % terms where EB* rise precedes behavioral rise")
    print("  ΔLead   = mean(r_forward - r_backward), positive = EB* leads")
    print("  C4-B%   = % terms showing strict decoupling (rho_early>0, rho_late≤0)")
    print("  ρ_late  = mean late-window Spearman(EB*, Beh), negative = decoupled")
    print("  C5_drop = RecAcc drop when top-binding heads ablated (negative = disrupted)")
    print("  Spec    = specificity = top_drop - rand_drop (positive = head-specific)")
    print("  C3_Δ    = mean few-shot delta score vs zero-shot")
    print("  C3_imp% = fraction of prompts improved by few-shot prefix")
    print()
