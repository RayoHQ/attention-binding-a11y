"""Watchdog: waits for CRFM behavioral 45-term to finish, then runs full pipeline.

Usage:
    python src/watchdog_and_run.py &
    # or:
    nohup python src/watchdog_and_run.py > logs/watchdog.log 2>&1 &
"""

import subprocess
import sys
import time
from pathlib import Path

WATCH_DIR = Path("data/results/behavioral_crfm_45")
EXPECTED_FILES = 40  # 5 seeds × 8 checkpoints
POLL_INTERVAL = 120  # check every 2 minutes
LOG_FILE = Path("logs/pipeline_post_crfm.log")


def count_done():
    return len(list(WATCH_DIR.glob("*.jsonl"))) if WATCH_DIR.exists() else 0


def run_step(label: str, cmd: list[str], log_fh):
    print(f"\n{'='*60}", flush=True)
    print(f"STEP: {label}", flush=True)
    print(f"CMD:  {' '.join(cmd)}", flush=True)
    print(f"{'='*60}", flush=True)
    log_fh.write(f"\n{'='*60}\nSTEP: {label}\n{'='*60}\n")
    log_fh.flush()
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"⚠️  Step '{label}' exited with code {result.returncode}", flush=True)
    return result.returncode == 0


def main():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(LOG_FILE, "a")

    print(f"Watchdog started. Waiting for {EXPECTED_FILES} files in {WATCH_DIR} ...", flush=True)

    while True:
        n = count_done()
        print(f"  [{time.strftime('%H:%M:%S')}] {n}/{EXPECTED_FILES} behavioral_crfm_45 files done",
              flush=True)
        if n >= EXPECTED_FILES:
            break
        time.sleep(POLL_INTERVAL)

    print(f"\n✅ CRFM behavioral complete ({n} files). Starting pipeline ...", flush=True)

    steps = [
        # C1-B + C4-B for CRFM 45-term
        ("C1-B crfm45",       ["python", "src/analyze_c1b_within_term.py", "--model", "crfm45"]),
        ("C4-B crfm45",       ["python", "src/analyze_c4b_decoupling.py",  "--model", "crfm45"]),
        # SmolLM3 45-term binding + behavioral
        ("SmolLM3 bind 45",   ["python", "src/extract_binding_smollm3.py", "--all",
                                "--prompts", "data/prompts/canonical_45terms.jsonl",
                                "--outdir",  "data/results/binding_smollm3_45"]),
        ("SmolLM3 beh 45",    ["python", "src/eval_behavior_smollm3.py", "--all",
                                "--prompts", "data/prompts/canonical_45terms.jsonl",
                                "--outdir",  "data/results/behavioral_smollm3_45"]),
        ("C1-B smollm345",    ["python", "src/analyze_c1b_within_term.py", "--model", "smollm345"]),
        ("C4-B smollm345",    ["python", "src/analyze_c4b_decoupling.py",  "--model", "smollm345"]),
        # C3 few-shot for all models
        ("C3 OLMo",           ["python", "src/run_c3_new_models.py", "--model", "olmo"]),
        ("C3 Qwen",           ["python", "src/run_c3_new_models.py", "--model", "qwen"]),
        ("C3 CRFM",           ["python", "src/run_c3_new_models.py", "--model", "crfm"]),
        ("C3 SmolLM3",        ["python", "src/run_c3_new_models.py", "--model", "smollm3"]),
        # C5 causal ablation for all models
        ("C5 OLMo",           ["python", "src/run_c5_new_models.py", "--model", "olmo"]),
        ("C5 Qwen",           ["python", "src/run_c5_new_models.py", "--model", "qwen"]),
        ("C5 CRFM",           ["python", "src/run_c5_new_models.py", "--model", "crfm"]),
        ("C5 SmolLM3",        ["python", "src/run_c5_new_models.py", "--model", "smollm3"]),
        # Full summary
        ("Summary matrix",    ["python", "src/summarize_full_matrix.py"]),
    ]

    results = {}
    for label, cmd in steps:
        ok = run_step(label, cmd, log_fh)
        results[label] = "✅" if ok else "❌"

    log_fh.close()
    print("\n\n==== PIPELINE RESULTS ====")
    for label, status in results.items():
        print(f"  {status}  {label}")
    print("==========================\n")


if __name__ == "__main__":
    main()
