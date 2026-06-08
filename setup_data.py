#!/usr/bin/env python3
"""Setup script for data directories and environment verification."""

import json
from pathlib import Path


def create_directories():
    dirs = [
        "data/prompts", "data/results/behavioral", "data/results/binding",
        "data/results/causal", "data/results/few_shot", "data/results/controls",
        "data/tokenization", "figures",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {d}")


def verify_prompts():
    p = Path("data/prompts/pilot_terms.jsonl")
    if p.exists():
        with open(p) as f:
            n = sum(1 for _ in f)
        print(f"  OK: {p} ({n} prompts)")
    else:
        print(f"  MISSING: {p}")


def check_dependencies():
    required = [
        ("torch", "torch"), ("transformers", "transformers"),
        ("transformer_lens", "transformer-lens"), ("numpy", "numpy"),
        ("scipy", "scipy"), ("pandas", "pandas"), ("tqdm", "tqdm"),
        ("matplotlib", "matplotlib"),
    ]
    for mod_name, pip_name in required:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "?")
            print(f"  OK: {pip_name} ({ver})")
        except ImportError:
            print(f"  MISSING: pip install {pip_name}")


def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU: {name} ({mem:.1f} GB)")
        else:
            print("  No CUDA GPU (CPU mode will be slow)")
    except Exception as e:
        print(f"  GPU check failed: {e}")


def main():
    print("=" * 60)
    print("Attention Binding - Reproducibility Setup")
    print("=" * 60)
    print("\n[1/4] Creating directories...")
    create_directories()
    print("\n[2/4] Checking dependencies...")
    check_dependencies()
    print("\n[3/4] Checking GPU...")
    check_gpu()
    print("\n[4/4] Verifying prompts...")
    verify_prompts()
    print("\nSetup complete! Next:")
    print("  python src/tokenization_audit.py")
    print("  python src/eval_behavior.py 160m step120000")


if __name__ == "__main__":
    main()
