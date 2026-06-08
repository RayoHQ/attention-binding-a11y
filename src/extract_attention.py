"""Attention extraction with memory-efficient layer-by-layer processing."""

import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from utils_model import load_pythia_with_checkpoint

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("data/results/binding")

# Aliases for terms that may appear differently in prompts
TERM_ALIASES = {
    "alt text": ["alternative text"],
}


def extract_binding_for_prompt(
    model,
    prompt_text: str,
    term: str,
    tokenizer,
) -> Dict:
    """
    Compute binding metrics for a single prompt.

    Uses layer-by-layer extraction to manage memory.
    """
    # Tokenize
    tokens = model.to_tokens(prompt_text, prepend_bos=True)
    seq_len = tokens.shape[1]

    # Get term span indices (token positions).
    # GPT-NeoX BPE: "screen reader" → ['screen', ' reader'] (bare),
    # but in context " screen reader" → [' screen', ' reader'] (space-prefixed).
    # Also try capitalized form (e.g., "Alt text" at sentence start).
    # Build search variants: bare, space-prefixed, capitalized, title-cased
    search_terms = [term]
    search_terms.extend(TERM_ALIASES.get(term.lower(), []))

    term_variants = []
    for t in search_terms:
        for form in [t, t.capitalize(), t.title()]:
            term_variants.append(tokenizer.encode(form, add_special_tokens=False))
            term_variants.append(tokenizer.encode(" " + form, add_special_tokens=False))
    # Deduplicate while preserving order
    seen = set()
    unique_variants = []
    for v in term_variants:
        key = tuple(v)
        if key not in seen:
            seen.add(key)
            unique_variants.append(v)

    full_token_ids = tokens[0].tolist()

    # Find term span start (try each variant)
    span_start = None
    term_tokens = unique_variants[0]  # default
    for variant in unique_variants:
        for i in range(len(full_token_ids) - len(variant) + 1):
            if full_token_ids[i : i + len(variant)] == variant:
                span_start = i
                term_tokens = variant
                break
        if span_start is not None:
            break

    n_term_tokens = len(term_tokens)

    if span_start is None:
        # Fallback: character-level search on decoded tokens.
        # Handles cases like "alternative text" containing "alt text"
        # conceptually — we locate the term's character position and
        # map back to the covering token indices.
        decoded_tokens = [tokenizer.decode([t]) for t in full_token_ids]
        joined = "".join(decoded_tokens)
        char_pos = joined.lower().find(term.lower())
        if char_pos >= 0:
            # Map character position back to token index
            cum_len = 0
            for idx, dt in enumerate(decoded_tokens):
                if cum_len >= char_pos:
                    span_start = idx
                    # Use the standard n_term_tokens (bare encoding length)
                    n_term_tokens = len(unique_variants[0])
                    break
                cum_len += len(dt)

        if span_start is None:
            span_start = max(0, seq_len - n_term_tokens - 5)
            print(
                f"⚠️  Could not find exact span for '{term}', "
                f"using approximate position {span_start}"
            )

    # Ensure span doesn't exceed sequence length
    span_end = min(span_start + n_term_tokens, seq_len)
    span_indices = list(range(span_start, span_end))

    # Layer-by-layer extraction
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    eb_per_layer = []
    bsi_per_layer_per_head = []

    for layer_idx in range(n_layers):
        # Extract only this layer's attention
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=[f"blocks.{layer_idx}.attn.hook_pattern"],
                stop_at_layer=layer_idx + 1,  # Stop after this layer
            )

        # Get attention pattern: [batch, heads, dest, src]
        attn_key = f"blocks.{layer_idx}.attn.hook_pattern"
        attn = cache[attn_key]  # [1, n_heads, seq_len, seq_len]

        # Compute BSI: mean attention from later span tokens to earlier span tokens
        bsi_per_head = []
        for head_idx in range(n_heads):
            head_attn = attn[0, head_idx]  # [seq_len, seq_len]

            # Collect attention from later to earlier within span
            later_to_earlier = []
            for i, dest_idx in enumerate(span_indices):
                for j, src_idx in enumerate(span_indices):
                    if dest_idx > src_idx:  # Later token attends to earlier
                        later_to_earlier.append(head_attn[dest_idx, src_idx].item())

            if later_to_earlier:
                bsi = sum(later_to_earlier) / len(later_to_earlier)
            else:
                bsi = 0.0

            bsi_per_head.append(bsi)

        bsi_per_head = torch.tensor(bsi_per_head)
        bsi_per_layer_per_head.append(bsi_per_head.tolist())

        # Compute EB: max - mean
        max_bsi = bsi_per_head.max().item()
        mean_bsi = bsi_per_head.mean().item()
        eb = max_bsi - mean_bsi
        eb_per_layer.append(eb)

        # Clear cache to free memory
        del cache
        torch.cuda.empty_cache()

    # EB* = max over layers (frozen aggregation for pilot)
    eb_star = max(eb_per_layer)
    best_layer = eb_per_layer.index(eb_star)

    # Entropy of head distribution at best layer
    bsi_best = torch.tensor(bsi_per_layer_per_head[best_layer])
    bsi_best_clamped = torch.clamp(bsi_best, min=1e-10)
    probs = bsi_best_clamped / bsi_best_clamped.sum()
    entropy = -(probs * torch.log(probs)).sum().item()

    return {
        "eb_star": round(eb_star, 6),
        "eb_per_layer": [round(x, 6) for x in eb_per_layer],
        "best_layer": best_layer,
        "entropy": round(entropy, 6),
        "span_indices": span_indices,
        "n_term_tokens": n_term_tokens,
    }


def extract_binding_for_checkpoint(
    model_size: str,
    checkpoint_step: str,
    prompts_file: str = "data/prompts/pilot_terms.jsonl",
    output_dir: str = "data/results/binding",
):
    """
    Extract binding metrics for all prompts at one checkpoint.
    """
    # Load model
    print(f"Loading {model_size} {checkpoint_step}...")
    model = load_pythia_with_checkpoint(model_size, checkpoint_step, DEVICE)
    tokenizer = model.tokenizer

    # Load prompts
    prompts = []
    with open(prompts_file) as f:
        for line in f:
            prompts.append(json.loads(line))

    # Process each prompt
    results = []
    for prompt in tqdm(prompts, desc=f"Extracting {model_size}/{checkpoint_step}"):
        # Use full template text
        template = prompt["template"]

        binding = extract_binding_for_prompt(
            model=model,
            prompt_text=template,
            term=prompt["term"],
            tokenizer=tokenizer,
        )

        result = {
            "model": f"pythia-{model_size}-deduped",
            "checkpoint": checkpoint_step,
            "term": prompt["term"],
            "task": prompt["task"],
            "prompt_id": prompt["prompt_id"],
            "prompt_template": template,
            **binding,
        }
        results.append(result)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = Path(output_dir) / f"{model_size}_{checkpoint_step}_binding.jsonl"

    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"✅ Saved {len(results)} results to {output_file}")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return output_file


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        size = sys.argv[1]
        step = sys.argv[2]
    else:
        size = "160m"
        step = "step120000"  # Peak performance checkpoint

    extract_binding_for_checkpoint(size, step)
