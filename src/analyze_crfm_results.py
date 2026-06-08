"""Quick analysis of CRFM C1-B/C4-B results vs Pythia baselines."""
import csv
import json
from pathlib import Path

SET_B = {
    "alt text", "aria attribute", "color contrast", "focus indicator",
    "heading structure", "keyboard navigation", "landmark region",
    "screen reader", "skip link",
}

C1B_DIR = Path("data/results/c1b")
C4B_DIR = Path("data/results/c4b")

# ── C1-B: Pythia-160M on Set-B terms only ───────────────────────────────────
print("=" * 60)
print("C1-B: Pythia-160M (Set-B terms only, for fair comparison)")
print("=" * 60)
p160_file = C1B_DIR / "pythia-160m_within_term_lead.csv"
if p160_file.exists():
    rows = list(csv.DictReader(open(p160_file)))
    setb = [r for r in rows if r["term"] in SET_B]
    n_lead = sum(1 for r in setb if r["lead_indicator"] in ("1", "True", "true"))
    print(f"  {n_lead}/{len(setb)} EB* lead ({n_lead/max(len(setb),1)*100:.0f}%)")
    for r in setb:
        arrow = "→ EB* leads" if r["lead_indicator"] in ("1", "True", "true") else "← Beh leads"
        print(f"    {r['term']:<28} {arrow}  r_fwd={float(r['r_forward']):+.3f}")
else:
    print("  File not found:", p160_file)

# ── C1-B: CRFM aggregate across 5 seeds ─────────────────────────────────────
print()
print("=" * 60)
print("C1-B: CRFM GPT-2 Small — aggregate across 5 seeds")
print("=" * 60)

seed_results = {}   # term -> list of lead indicators
for seed in range(1, 6):
    f = C1B_DIR / f"crfm-gpt2-sm-x{seed}_within_term_lead.csv"
    if not f.exists():
        print(f"  Missing: {f.name}")
        continue
    for r in csv.DictReader(open(f)):
        term = r["term"]
        lead = r["lead_indicator"] in ("1", "True", "true")
        seed_results.setdefault(term, []).append(lead)

# Aggregate: majority vote per term
print(f"  {'Term':<30} {'Seeds EB* leading':<20} Majority")
print(f"  {'-'*60}")
agg_leads = 0
for term in sorted(seed_results):
    leads = seed_results[term]
    n = sum(leads)
    majority = n > len(leads) / 2
    if majority:
        agg_leads += 1
    print(f"  {term:<30} {n}/{len(leads)} seeds          {'→ EB*' if majority else '← Beh'}")

total_terms = len(seed_results)
print(f"\n  Aggregate: {agg_leads}/{total_terms} terms EB* leads ({agg_leads/max(total_terms,1)*100:.0f}%)")

# Per-seed summary
print()
for seed in range(1, 6):
    f = C1B_DIR / f"crfm-gpt2-sm-x{seed}_population_test.json"
    if f.exists():
        d = json.load(open(f))
        print(f"  seed{seed}: {d['n_lead']}/{d['n_terms']} ({d['lead_fraction']*100:.0f}%)  "
              f"p={d['binomial_p']:.4f}  mean_delta={d['mean_lead_diff']:+.3f}")

# ── C4-B: CRFM aggregate ─────────────────────────────────────────────────────
print()
print("=" * 60)
print("C4-B: CRFM GPT-2 Small — aggregate across 5 seeds")
print("=" * 60)

rho_lates = []
decouple_fracs = []
for seed in range(1, 6):
    f = C4B_DIR / f"crfm-gpt2-sm-x{seed}_within_term_decoupling.csv"
    if not f.exists():
        continue
    rows = list(csv.DictReader(open(f)))
    n_dec = sum(1 for r in rows if r.get("decouple_strict") in ("1", "True", "true"))
    rho_l = [float(r["rho_late"]) for r in rows if r.get("rho_late")]
    avg_late = sum(rho_l) / len(rho_l) if rho_l else 0.0
    decouple_fracs.append(n_dec / len(rows))
    rho_lates.append(avg_late)
    print(f"  seed{seed}: {n_dec}/{len(rows)} decouple ({n_dec/len(rows)*100:.0f}%)  "
          f"mean_rho_late={avg_late:+.3f}")

if rho_lates:
    print(f"\n  Mean across seeds: decouple={sum(decouple_fracs)/len(decouple_fracs)*100:.0f}%  "
          f"rho_late={sum(rho_lates)/len(rho_lates):+.3f}")

# ── Cross-model comparison table ─────────────────────────────────────────────
print()
print("=" * 60)
print("CROSS-MODEL SUMMARY")
print("=" * 60)
print(f"  {'Model':<22} {'Params':<10} {'C1-B lead%':<14} {'C4-B decouple%':<16} {'C4-B rho_late'}")
print(f"  {'-'*75}")

comparisons = [
    ("Pythia-160M",    "160M", "143k",   "7%",      "46%", "+0.044", "41 terms"),
    ("CRFM GPT-2 Sm", "117M", "400k",   "~74%†",   "~33%†","+0.261†","9 terms, 5-seed maj."),
    ("Pythia-1B",      "1B",  "143k",   "73%",     "54%", "-0.054", "41 terms"),
    ("OLMo-1B",        "1B",  "143k",   "78%",     "62%", "-0.348", "9 terms"),
    ("Pythia-2.8B",    "2.8B","143k",   "79%",     "43%", "+0.270", "34 terms"),
    ("SmolLM3-3B",     "3B",  "3440k",  "33%‡",    "67%", "-0.189", "9 terms; ck starts step40k"),
]
hdr = f"  {'Model':<20} {'Params':<7} {'Steps':<8} {'C1-B%':<10} {'C4-B%':<8} {'rho_late':<10} Notes"
print(hdr)
print("  " + "-"*85)
for row in comparisons:
    print(f"  {row[0]:<20} {row[1]:<7} {row[2]:<8} {row[3]:<10} {row[4]:<8} {row[5]:<10} {row[6]}")
print()
print("  † CRFM: 5-seed majority vote across 5 random-seed GPT-2 Small models")
print("  ‡ SmolLM3 C1-B=33% is a CENSORING ARTIFACT: earliest available ck is step40k")
print("    (EB* already peaked by then; the binding→behavior transition precedes our window)")
print("  Key: C4-B rho_late < 0 = full decoupling; > 0 = partial coupling")
