"""Behavioral evaluation for Qwen2.5-1.5B (single final checkpoint).

Usage:
    python src/eval_behavior_qwen.py
    python src/eval_behavior_qwen.py --prompts data/prompts/canonical_45terms.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils_model_qwen import load_qwen, DEVICE
from scoring import score_generation

PROMPTS_FILE = Path("data/prompts/canonical_45terms.jsonl")
OUTPUT_DIR = Path("data/results/behavioral_qwen")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def score_recognition_hf(model, tokenizer, prompt: str, choices: list, answer_idx: int) -> dict:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    prompt_ids = prompt_ids.to(DEVICE)
    prompt_len = prompt_ids.shape[1]

    choice_scores = []
    for choice in choices:
        choice_token_ids = tokenizer.encode(" " + choice, add_special_tokens=False)
        choice_tensor = torch.tensor([choice_token_ids], device=DEVICE)
        full_ids = torch.cat([prompt_ids, choice_tensor], dim=1)

        with torch.no_grad():
            logits = model(full_ids).logits
            log_probs = torch.log_softmax(logits.float(), dim=-1)

        total_lp = 0.0
        for i, tok_id in enumerate(choice_token_ids):
            pos = prompt_len - 1 + i
            total_lp += log_probs[0, pos, tok_id].item()
        choice_scores.append(total_lp / max(1, len(choice_token_ids)))

    predicted_idx = int(torch.tensor(choice_scores).argmax())
    return {
        "predicted_idx": predicted_idx,
        "is_correct": predicted_idx == answer_idx,
        "score": 1.0 if predicted_idx == answer_idx else 0.0,
    }


def score_generation_hf(model, tokenizer, prompt: str, term: str, max_new_tokens: int = 25) -> dict:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True).strip()
    return {"score": score_generation(completion, term), "generated_text": completion}


def run(prompts_file=None, outdir=None):
    _prompts = Path(prompts_file) if prompts_file else PROMPTS_FILE
    _outdir = Path(outdir) if outdir else OUTPUT_DIR
    _outdir.mkdir(parents=True, exist_ok=True)
    out_file = _outdir / "qwen_final_behavioral_qwen.jsonl"

    if out_file.exists():
        print(f"⏭  Already exists: {out_file} — skipping")
        return out_file

    print("Loading Qwen2.5-1.5B ...")
    model, tokenizer = load_qwen(DEVICE)

    with open(_prompts) as f:
        prompts = [json.loads(l) for l in f]

    results = []
    for p in tqdm(prompts, desc="behavioral qwen2.5-1.5b"):
        task = p["task"]
        if task == "recognition":
            res = score_recognition_hf(model, tokenizer, p["template"],
                                       p["choices"], p["answer_idx"])
            bscore = res["score"]
        else:
            res = score_generation_hf(model, tokenizer, p["template"],
                                      p["term"], p.get("max_tokens", 25))
            bscore = res["score"]

        results.append({
            "model": "qwen2.5-1.5b",
            "checkpoint": "final",
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
    print(f"✅ Saved {len(results)} → {out_file}")
    print(f"  Recognition mean={sum(rec)/len(rec):.3f}  Generation mean={sum(gen)/len(gen):.3f}")

    del model
    torch.cuda.empty_cache()
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default=None)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()
    run(args.prompts, args.outdir)
