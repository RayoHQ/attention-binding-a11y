"""C3: Few-shot unlockability test for CRFM, SmolLM3, OLMo, and Qwen2.5.

Tests whether models with high EB* but low behavioral performance show
improved generation scores with 2-sentence few-shot prompting prefix.

Runs at two checkpoints per lifecycle model (early + late) using
existing binding/behavioral data to select high-EB* candidates.

Usage:
    python src/run_c3_new_models.py --model crfm
    python src/run_c3_new_models.py --model smollm3
    python src/run_c3_new_models.py --model olmo
    python src/run_c3_new_models.py --model qwen
    python src/run_c3_new_models.py --all
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from scoring import score_generation

OUTPUT_DIR = Path("data/results/few_shot_c3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_FILE = Path("data/prompts/expanded_terms_100.jsonl")

FEW_SHOT_PREFIX = (
    "Here are two examples of accessibility concepts:\n"
    "A skip link is a navigation aid that lets keyboard users jump past repeated content.\n"
    "Alt text is a text description of an image used by screen readers.\n"
    "Now complete the following:\n"
)


def add_few_shot(template: str) -> str:
    return FEW_SHOT_PREFIX + template


def load_prompts(prompts_file=None):
    with open(prompts_file or PROMPTS_FILE) as f:
        return [json.loads(l) for l in f]


# ── CRFM ─────────────────────────────────────────────────────────────────────
def run_crfm_c3(seed=1):
    from extract_binding_crfm import load_crfm, SEED_MODEL_IDS
    from eval_behavior_crfm import evaluate_prompt as crfm_eval_prompt
    CHECKPOINTS_C3 = ["checkpoint-1000", "checkpoint-400000"]
    prompts = [p for p in load_prompts() if p["task"] == "generation"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_file = OUTPUT_DIR / f"crfm_seed{seed}_c3_fewshot.jsonl"
    if out_file.exists():
        print(f"⏭  {out_file.name} exists — skipping")
        return

    results = []
    for ck in CHECKPOINTS_C3:
        print(f"\n  Loading CRFM seed{seed} @ {ck} ...")
        model = load_crfm(seed, ck, device)
        for p in tqdm(prompts, desc=f"C3 crfm-x{seed}/{ck}"):
            # zero-shot
            zs = crfm_eval_prompt(model, p)
            # few-shot: inject prefix
            fs_prompt = {**p, "template": add_few_shot(p["template"])}
            fs = crfm_eval_prompt(model, fs_prompt)
            results.append({
                "model": f"crfm-gpt2-small-x{seed}",
                "checkpoint": ck,
                "term": p["term"],
                "prompt_id": p["prompt_id"],
                "zero_shot_score": zs["behavioral_score"],
                "few_shot_score": fs["behavioral_score"],
                "delta": fs["behavioral_score"] - zs["behavioral_score"],
            })
        del model
        torch.cuda.empty_cache()

    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"✅ CRFM C3 seed{seed} → {out_file}")


def run_crfm_c3_all_seeds():
    for s in range(1, 6):
        run_crfm_c3(seed=s)


# ── SmolLM3 ──────────────────────────────────────────────────────────────────
def run_smollm3_c3():
    from utils_model_smollm3 import load_smollm3_with_checkpoint
    from eval_behavior_smollm3 import evaluate_prompt as smollm3_eval_prompt
    CHECKPOINTS_C3 = ["step40k", "step3440k"]
    prompts = [p for p in load_prompts() if p["task"] == "generation"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_file = OUTPUT_DIR / "smollm3_c3_fewshot.jsonl"
    if out_file.exists():
        print(f"⏭  {out_file.name} exists — skipping")
        return

    results = []
    for ck in CHECKPOINTS_C3:
        print(f"\n  Loading SmolLM3 @ {ck} ...")
        model, tokenizer = load_smollm3_with_checkpoint(ck, device)
        for p in tqdm(prompts, desc=f"C3 smollm3/{ck}"):
            zs = smollm3_eval_prompt(model, tokenizer, p)
            fs_prompt = {**p, "template": add_few_shot(p["template"])}
            fs = smollm3_eval_prompt(model, tokenizer, fs_prompt)
            results.append({
                "model": "smollm3-3b",
                "checkpoint": ck,
                "term": p["term"],
                "prompt_id": p["prompt_id"],
                "zero_shot_score": zs["behavioral_score"],
                "few_shot_score": fs["behavioral_score"],
                "delta": fs["behavioral_score"] - zs["behavioral_score"],
            })
        del model
        torch.cuda.empty_cache()

    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"✅ SmolLM3 C3 → {out_file}")


# ── OLMo ─────────────────────────────────────────────────────────────────────
def run_olmo_c3():
    from utils_model_olmo import load_olmo_with_checkpoint
    from eval_behavior_olmo import score_generation_hf, score_recognition_hf
    CHECKPOINTS_C3 = ["step15k", "step143k"]
    prompts = [p for p in load_prompts() if p["task"] == "generation"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_file = OUTPUT_DIR / "olmo_c3_fewshot.jsonl"
    if out_file.exists():
        print(f"⏭  {out_file.name} exists — skipping")
        return

    results = []
    for ck in CHECKPOINTS_C3:
        print(f"\n  Loading OLMo @ {ck} ...")
        model, tokenizer = load_olmo_with_checkpoint(ck, device)
        for p in tqdm(prompts, desc=f"C3 olmo/{ck}"):
            zs = score_generation_hf(model, tokenizer, p["template"], p["term"])
            fs_text = add_few_shot(p["template"])
            fs = score_generation_hf(model, tokenizer, fs_text, p["term"])
            results.append({
                "model": "olmo-1b",
                "checkpoint": ck,
                "term": p["term"],
                "prompt_id": p.get("prompt_id", ""),
                "zero_shot_score": zs["score"],
                "few_shot_score": fs["score"],
                "delta": fs["score"] - zs["score"],
            })
        del model
        torch.cuda.empty_cache()

    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"✅ OLMo C3 → {out_file}")


# ── Qwen ─────────────────────────────────────────────────────────────────────
def run_qwen_c3():
    from utils_model_qwen import load_qwen
    from eval_behavior_qwen import score_generation_hf
    prompts = [p for p in load_prompts(Path("data/prompts/canonical_45terms.jsonl"))
               if p["task"] == "generation"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_file = OUTPUT_DIR / "qwen_final_c3_fewshot.jsonl"
    if out_file.exists():
        print(f"⏭  {out_file.name} exists — skipping")
        return

    print("Loading Qwen2.5-1.5B ...")
    model, tokenizer = load_qwen(device)
    results = []
    for p in tqdm(prompts, desc="C3 qwen/final"):
        zs = score_generation_hf(model, tokenizer, p["template"], p["term"])
        fs = score_generation_hf(model, tokenizer, add_few_shot(p["template"]), p["term"])
        results.append({
            "model": "qwen2.5-1.5b",
            "checkpoint": "final",
            "term": p["term"],
            "prompt_id": p.get("prompt_id", ""),
            "zero_shot_score": zs["score"],
            "few_shot_score": fs["score"],
            "delta": fs["score"] - zs["score"],
        })
    del model
    torch.cuda.empty_cache()

    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"✅ Qwen C3 → {out_file}")


# ── Pythia ────────────────────────────────────────────────────────────────────
def run_pythia_c3():
    """Run C3 few-shot for Pythia-160M, 1B, and 2.8B at early + late checkpoints.

    Delegates to eval_few_shot_c3.evaluate_checkpoint which handles Pythia via
    TransformerLens and saves to data/results/few_shot_c3/{size}_{ck}_c3_fewshot.json.
    """
    from eval_few_shot_c3 import evaluate_checkpoint
    PYTHIA_C3_RUNS = [
        ("160m",  "step15000"),
        ("160m",  "step143000"),
        ("1b",    "step15000"),
        ("1b",    "step143000"),
        ("2.8b",  "step15000"),
        ("2.8b",  "step143000"),
    ]
    for size, ck in PYTHIA_C3_RUNS:
        print(f"\n>>> Pythia-{size} @ {ck}")
        evaluate_checkpoint(size, ck)


RUNNERS = {
    "pythia": run_pythia_c3,
    "crfm": run_crfm_c3_all_seeds,
    "smollm3": run_smollm3_c3,
    "olmo": run_olmo_c3,
    "qwen": run_qwen_c3,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(RUNNERS.keys()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed index for CRFM (1-5). Only used with --model crfm.")
    args = parser.parse_args()

    if args.all:
        for fn in RUNNERS.values():
            fn()
    elif args.model == "crfm":
        if args.seed is not None:
            run_crfm_c3(seed=args.seed)
        else:
            run_crfm_c3_all_seeds()
    elif args.model:
        RUNNERS[args.model]()
    else:
        parser.print_help()
