import csv, json

rows = list(csv.DictReader(open("data/results/c4b/pythia-28b_within_term_decoupling.csv")))
res = json.load(open("data/results/c4b/subceiling_spearman_results.json"))
mean_beh_late = res["pythia-2.8b"]["term_mean_beh_late"]
ceiling_terms = set(res["pythia-2.8b"]["ceiling_terms"])

full_rhos = [float(r["rho_late"]) for r in rows]
sub_rhos  = [float(r["rho_late"]) for r in rows if r["term"] not in ceiling_terms]

print(f"2.8B per-term avg rho_late — FULL     ({len(full_rhos)} terms): {sum(full_rhos)/len(full_rhos):+.4f}")
print(f"2.8B per-term avg rho_late — sub-0.80 ({len(sub_rhos)} terms): {sum(sub_rhos)/len(sub_rhos):+.4f}")
print(f"Ceiling terms excluded: {sorted(ceiling_terms)}")
print("\nCeiling term rho_lates:")
for r in rows:
    if r["term"] in ceiling_terms:
        print(f"  {r['term']:<25}  rho_late={float(r['rho_late']):+.4f}  beh={mean_beh_late.get(r['term'],0):.3f}")
print("\nTop-6 sub-ceiling rho_lates:")
sub_rows = sorted([r for r in rows if r["term"] not in ceiling_terms],
                  key=lambda x: float(x["rho_late"]), reverse=True)
for r in sub_rows[:6]:
    print(f"  {r['term']:<25}  rho_late={float(r['rho_late']):+.4f}  beh={mean_beh_late.get(r['term'],0):.3f}")
