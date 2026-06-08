"""B5: Behavioral evaluation for OLMo-1B on the 9 original terms.

Uses HuggingFace model directly (no TransformerLens) — compatible with
the eager-attention OLMo loaded in utils_model_olmo.py.

Usage:
    python src/eval_behavior_olmo.py                        # all checkpoints
    python src/eval_behavior_olmo.py --checkpoint step143k  # one checkpoint
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils_model_olmo import load_olmo_with_checkpoint, OLMO_1B_CHECKPOINTS
from scoring import score_generation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPTS_FILE = Path("data/prompts/expanded_terms_100.jsonl")
OUTPUT_DIR = Path("data/results/behavioral_olmo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_CHECKPOINTS = list(OLMO_1B_CHECKPOINTS.keys())


# ---------------------------------------------------------------------------
# HF-native recognition scorer (mirrors scoring.score_recognition_logprob)
# ---------------------------------------------------------------------------

def score_recognition_hf(model, tokenizer, prompt: str, choices: list, answer_idx: int) -> dict:
    """Score MCQ via full-sequence log-probs using a HuggingFace CausalLM."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    prompt_ids = prompt_ids.to(DEVICE)
    prompt_len = prompt_ids.shape[1]

    choice_scores = []
    for choice in choices:
        choice_token_ids = tokenizer.encode(
            " " + choice, add_special_tokens=False
        )
        choice_tensor = torch.tensor([choice_token_ids], device=DEVICE)
        full_ids = torch.cat([prompt_ids, choice_tensor], dim=1)

        with torch.no_grad():
            logits = model(full_ids).logits          # [1, seq_len, vocab]
            log_probs = torch.log_softmax(logits.float(), dim=-1)

        total_lp = 0.0
        for i, tok_id in enumerate(choice_token_ids):
            pos = prompt_len - 1 + i
            total_lp += log_probs[0, pos, tok_id].item()

        choice_scores.append(total_lp / max(1, len(choice_token_ids)))

    predicted_idx = int(torch.tensor(choice_scores).argmax())
    is_correct = predicted_idx == answer_idx
    return {
        "predicted_idx": predicted_idx,
        "is_correct": is_correct,
        "score": 1.0 if is_correct else 0.0,
        "log_probs": choice_scores,
    }


# ---------------------------------------------------------------------------
# Generation scorer
# ---------------------------------------------------------------------------

def score_generation_hf(model, tokenizer, prompt: str, term: str, max_new_tokens: int = 25) -> dict:
    """Generate a completion and score it with keyword rubric."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,        # ignored when do_sample=False
            pad_token_id=tokenizer.eos_token_id,
        )

    completion_ids = output_ids[0, prompt_len:]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    return {
        "score": score_generation(completion, term),
        "generated_text": completion,
    }


# ---------------------------------------------------------------------------
# Per-checkpoint evaluation
# ---------------------------------------------------------------------------

def evaluate_checkpoint(checkpoint_key: str, prompts_file=None, outdir=None):
    _outdir = Path(outdir) if outdir else OUTPUT_DIR
    _outdir.mkdir(parents=True, exist_ok=True)
    out_file = _outdir / f"olmo_1b_{checkpoint_key}_behavioral.jsonl"
    if out_file.exists():
        print(f"  ⏭  {checkpoint_key} already exists — skipping")
        return

    print(f"\n{'='*60}")
    print(f"OLMo-1B behavioral: {checkpoint_key}")
    print(f"{'='*60}")

    model, tokenizer = load_olmo_with_checkpoint(checkpoint_key, DEVICE)

    prompts = []
    with open(prompts_file or PROMPTS_FILE) as f:
        for line in f:
            prompts.append(json.loads(line))

    results = []
    for p in tqdm(prompts, desc=checkpoint_key):
        task = p["task"]
        template = p["template"]

        if task == "recognition":
            res = score_recognition_hf(
                model, tokenizer, template, p["choices"], p["answer_idx"]
            )
            bscore = res["score"]
        else:
            res = score_generation_hf(
                model, tokenizer, template, p["term"],
                max_new_tokens=p.get("max_tokens", 25),
            )
            bscore = res["score"]

        results.append({
            "model": "olmo-1b",
            "checkpoint": checkpoint_key,
            "term": p["term"],
            "task": task,
            "prompt_id": p.get("prompt_id", ""),
            "behavioral_score": bscore,
        })

    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    rec = [r["behavioral_score"] for r in results if r["task"] == "recognition"]
    gen = [r["behavioral_score"] for r in results if r["task"] == "generation"]
    print(f"Saved {len(results)} results → {out_file}")
    print(f"  Recognition mean={sum(rec)/len(rec):.3f}  Generation mean={sum(gen)/len(gen):.3f}")

    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=ALL_CHECKPOINTS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--prompts", default=None, help="JSONL prompts file")
    parser.add_argument("--outdir", default=None, help="Output directory")
    args = parser.parse_args()

    checkpoints = ALL_CHECKPOINTS if (args.all or not args.checkpoint) else [args.checkpoint]

    for ck in checkpoints:
        try:
            evaluate_checkpoint(ck, args.prompts, args.outdir)
        except Exception as e:
            print(f"❌ Error {ck}: {e}")
            import traceback; traceback.print_exc()

    print("\n✅ OLMo-1B behavioral evaluation complete!")
