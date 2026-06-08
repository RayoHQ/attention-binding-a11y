"""Quick sanity check on 45-term extraction outputs."""
import json
from pathlib import Path

checks = [
    ("OLMo 45",  "data/results/binding_olmo_45",  "*.jsonl", "eb_star"),
    ("Qwen 45",  "data/results/binding_qwen",      "*.jsonl", "eb_star"),
]

for label, dirpath, glob, field in checks:
    d = Path(dirpath)
    files = sorted(d.glob(glob))
    print(f"\n{label}: {len(files)} files in {dirpath}")
    for f in files:
        recs = [json.loads(l) for l in open(f)]
        terms = set(r["term"] for r in recs)
        vals = [r[field] for r in recs if field in r]
        mean_val = sum(vals) / len(vals) if vals else 0
        zero_span = sum(1 for r in recs if r.get("n_term_tokens", 1) == 0)
        flag = " ⚠️ ZERO_SPAN" if zero_span else ""
        print(f"  {f.name:<55} {len(recs):4d} recs  {len(terms):2d} terms  "
              f"EB*_mean={mean_val:.3f}{flag}")
