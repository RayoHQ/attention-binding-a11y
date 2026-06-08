"""Phase 0b: Build canonical_45terms.jsonl — single deduplicated prompt register.

Sources (in priority order, last wins for duplicate prompt_ids per term):
  1. pilot_terms.jsonl            (3 terms)
  2. expanded_terms_100.jsonl     (9 terms Set B — replaces expanded_terms.jsonl)
  3. expanded_terms_tier123.jsonl (21 terms)
  4. expanded_terms_wave2.jsonl   (12 terms)

Output: data/prompts/canonical_45terms.jsonl
"""

import json
from pathlib import Path

BASE = Path("data/prompts")
OUTPUT = BASE / "canonical_45terms.jsonl"

SOURCES = [
    BASE / "pilot_terms.jsonl",
    BASE / "expanded_terms_100.jsonl",
    BASE / "expanded_terms_tier123.jsonl",
    BASE / "expanded_terms_wave2.jsonl",
]

EXPECTED_TERMS = 41  # 9 Set-B + 20 tier123-new + 12 wave-2 (Set-A-only: form validation, tab order excluded)


def load_source(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_canonical():
    seen_keys: dict[tuple, dict] = {}  # (term, task, prompt_id) -> record
    term_sources: dict[str, str] = {}

    for src in SOURCES:
        if not src.exists():
            print(f"⚠️  Missing: {src} — skipping")
            continue
        records = load_source(src)
        for rec in records:
            key = (rec["term"], rec["task"], rec.get("prompt_id", ""))
            seen_keys[key] = rec  # later source wins on collision
            term_sources[rec["term"]] = src.name

    all_records = list(seen_keys.values())

    # Sort: by term alphabetically, then task (recognition before generation), then prompt_id
    task_order = {"recognition": 0, "generation": 1}
    all_records.sort(key=lambda r: (r["term"], task_order.get(r["task"], 2), r.get("prompt_id", "")))

    unique_terms = sorted(set(r["term"] for r in all_records))
    print(f"\n✅ Canonical register: {len(unique_terms)} unique terms, {len(all_records)} total prompts")
    print(f"\nTerm list:")
    for i, t in enumerate(unique_terms, 1):
        src = term_sources.get(t, "?")
        n_rec = sum(1 for r in all_records if r["term"] == t and r["task"] == "recognition")
        n_gen = sum(1 for r in all_records if r["term"] == t and r["task"] == "generation")
        print(f"  {i:2d}. {t:<30s}  rec={n_rec}  gen={n_gen}  [{src}]")

    if len(unique_terms) != EXPECTED_TERMS:
        print(f"\n⚠️  Expected {EXPECTED_TERMS} unique terms, got {len(unique_terms)}")
    else:
        print(f"\n✅ Exactly {EXPECTED_TERMS} unique terms confirmed")

    with open(OUTPUT, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    print(f"✅ Written to {OUTPUT}")
    return unique_terms


if __name__ == "__main__":
    build_canonical()
