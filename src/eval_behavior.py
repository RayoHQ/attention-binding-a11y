"""Behavioral probe evaluation (1A)."""

import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from utils_model import load_pythia_with_checkpoint
from scoring import evaluate_output


def load_prompts(prompt_file: str) -> List[Dict]:
    """Load prompt templates from JSONL."""
    prompts = []
    with open(prompt_file, "r") as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


def run_behavioral_probe(
    model,
    prompt: Dict,
    device: str = "cuda",
    temperature: float = 0.0,
    seed: int = None,
) -> Dict:
    """
    Run single behavioral probe.

    Args:
        model: HookedTransformer
        prompt: Prompt dict with keys: template, task, term, etc.
        device: Device to run on

    Returns:
        Result dict with output and scores
    """
    template = prompt["template"]
    task = prompt["task"]

    if task == "recognition":
        # Log-prob scoring: no generation needed, just compare choice probs
        choices = prompt.get("choices")
        answer_idx = prompt.get("answer_idx")

        eval_result = evaluate_output(
            text_out="",
            task=task,
            term=prompt["term"],
            answer=prompt.get("answer"),
            model=model,
            prompt=template,
            choices=choices,
            answer_idx=answer_idx,
        )

        # Map predicted_idx back to letter for readability
        letters = ["A", "B", "C", "D"]
        predicted_letter = letters[eval_result.get("predicted_idx", 0)]

        return {
            "text_out": f"predicted={predicted_letter}",
            "full_output": template,
            "is_correct": eval_result["is_correct"],
            "score": eval_result["score"],
            "eval_method": eval_result["method"],
            "predicted_idx": eval_result.get("predicted_idx"),
            "log_probs": eval_result.get("log_probs"),
        }
    else:
        # Generation: produce text and score with keyword rubric
        tokens = model.to_tokens(template)

        with torch.no_grad():
            max_tokens = prompt.get("max_tokens", 20)
            if seed is not None:
                torch.manual_seed(seed)
            do_sample = temperature > 0.0
            output = model.generate(
                tokens,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )

        text_out = model.tokenizer.decode(output[0], skip_special_tokens=True)

        # Remove the prompt from output for cleaner evaluation
        if text_out.startswith(template):
            generated = text_out[len(template):].strip()
        else:
            generated = text_out.strip()

        eval_result = evaluate_output(
            text_out=generated,
            task=task,
            term=prompt["term"],
        )

        return {
            "text_out": generated,
            "full_output": text_out,
            "is_correct": eval_result["is_correct"],
            "score": eval_result["score"],
            "eval_method": eval_result["method"],
        }


def run_behavioral_evaluation(
    model_size: str,
    checkpoint_step: str,
    prompts_file: str = "data/prompts/pilot_terms.jsonl",
    output_dir: str = "data/results/behavioral",
    device: str = "cuda",
    temperature: float = 0.0,
    seed: int = None,
) -> str:
    """
    Run full behavioral evaluation for one model checkpoint.

    Args:
        model_size: "160m", "1b", or "2.8b"
        checkpoint_step: e.g., "step0", "step30000"
        prompts_file: Path to prompts JSONL
        output_dir: Where to save results
        device: Device to use

    Returns:
        Path to output file
    """
    print(f"Loading {model_size} {checkpoint_step}...")
    model = load_pythia_with_checkpoint(model_size, checkpoint_step, device)

    prompts = load_prompts(prompts_file)

    results = []
    for prompt in tqdm(prompts, desc="Evaluating"):
        result = run_behavioral_probe(model, prompt, device, temperature, seed)

        record = {
            "model": f"pythia-{model_size}-deduped",
            "checkpoint": checkpoint_step,
            "term": prompt["term"],
            "task": prompt["task"],
            "prompt_id": prompt["prompt_id"],
            "prompt_template": prompt["template"],
            **result,
        }
        results.append(record)

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        f"{model_size}_{checkpoint_step}_behavioral.jsonl",
    )

    with open(output_file, "w") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(results)} results to {output_file}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_file


def aggregate_behavioral_results(
    results_dir: str = "data/results/behavioral",
) -> Dict:
    """
    Aggregate results across all model checkpoints.
    Returns accuracy per (model, checkpoint, term, task).
    """
    from collections import defaultdict

    stats = defaultdict(lambda: {
        "correct": 0,
        "total": 0,
        "scores": [],
    })

    for filename in os.listdir(results_dir):
        if not filename.endswith(".jsonl"):
            continue

        filepath = os.path.join(results_dir, filename)
        with open(filepath, "r") as f:
            for line in f:
                record = json.loads(line)

                key = (
                    record["model"],
                    record["checkpoint"],
                    record["term"],
                    record["task"],
                )

                stats[key]["total"] += 1
                if record["is_correct"]:
                    stats[key]["correct"] += 1
                stats[key]["scores"].append(record["score"])

    aggregated = {}
    for key, data in stats.items():
        aggregated[key] = {
            "accuracy": data["correct"] / data["total"] if data["total"] > 0 else 0,
            "mean_score": sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0,
            "n": data["total"],
        }

    return aggregated


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        size = sys.argv[1]
        step = sys.argv[2]
    else:
        size = "160m"
        step = "step0"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_file = run_behavioral_evaluation(size, step, device=device)
    print(f"Complete: {output_file}")
