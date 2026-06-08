"""OLMo model loading and attention extraction utilities (B3).

Uses HuggingFace forward hooks to extract per-head attention patterns,
mirroring the TransformerLens approach in extract_attention.py for Pythia.
OLMo-1B-hf is natively supported in transformers >= 4.40 — no hf_olmo needed.
"""

import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Verified OLMo-1B-hf checkpoint branches (B1, verified Apr 2026)
OLMO_1B_CHECKPOINTS = {
    "step0":    "step1000-tokens4B",       # ~0%
    "step15k":  "step80000-tokens335B",    # ~11%
    "step30k":  "step110000-tokens461B",   # ~15%
    "step60k":  "step330000-tokens1383B",  # ~45%
    "step90k":  "step460000-tokens1928B",  # ~62%
    "step120k": "step620000-tokens2599B",  # ~84%
    "step140k": "step720000-tokens3018B",  # ~98%
    "step143k": "step738020-tokens3094B",  # 100% (final)
}


def load_olmo_with_checkpoint(
    checkpoint_key: str,
    device: Optional[str] = None,
):
    """Load OLMo-1B-hf at a specific training checkpoint.

    Args:
        checkpoint_key: One of the keys in OLMO_1B_CHECKPOINTS
                        (e.g., 'step0', 'step143k').
        device: Device string. Defaults to CUDA if available.

    Returns:
        (model, tokenizer) tuple.
    """
    if device is None:
        device = DEVICE

    if checkpoint_key not in OLMO_1B_CHECKPOINTS:
        raise ValueError(
            f"Unknown checkpoint '{checkpoint_key}'. "
            f"Valid keys: {list(OLMO_1B_CHECKPOINTS.keys())}"
        )

    revision = OLMO_1B_CHECKPOINTS[checkpoint_key]
    model_id = "allenai/OLMo-1B-hf"
    logger.info("Loading %s revision=%s on %s", model_id, revision, device)

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    logger.info("Loaded OLMo-1B %s successfully", revision)
    return model, tokenizer


class OLMoAttentionHookExtractor:
    """Extracts per-head attention patterns from OLMo via forward hooks.

    OLMo's attention module exposes attention weights via the
    output_attentions=True flag in the HuggingFace API.
    Pattern shape: (batch, n_heads, seq_len, seq_len).
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = model.config.num_hidden_layers
        self.n_heads = model.config.num_attention_heads

    def extract_binding_for_prompt(self, prompt_text: str, term: str) -> dict:
        """Compute EB* for a single prompt, matching the Pythia interface.

        Args:
            prompt_text: Full prompt string.
            term: Accessibility term to locate in the token sequence.

        Returns:
            Dict with eb_star, eb_per_layer, best_layer, entropy,
            span_indices, n_term_tokens — identical schema to extract_attention.py.
        """
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
        full_token_ids = inputs["input_ids"][0].tolist()
        seq_len = len(full_token_ids)

        # Locate term span (same logic as extract_attention.py)
        span_start, n_term_tokens = self._find_term_span(full_token_ids, term)
        span_end = min(span_start + n_term_tokens, seq_len)
        span_indices = list(range(span_start, span_end))

        # Forward pass with attention outputs
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                return_dict=True,
            )

        # outputs.attentions: tuple of (batch, n_heads, seq, seq) per layer
        eb_per_layer = []
        bsi_per_layer = []

        for layer_attn in outputs.attentions:
            # layer_attn: [1, n_heads, seq_len, seq_len] — cast to fp32 for numerics
            layer_attn = layer_attn.float()
            bsi_per_head = []
            for head_idx in range(self.n_heads):
                head_attn = layer_attn[0, head_idx]  # [seq_len, seq_len]
                later_to_earlier = [
                    head_attn[dest, src].item()
                    for i, dest in enumerate(span_indices)
                    for j, src in enumerate(span_indices)
                    if dest > src
                ]
                bsi = sum(later_to_earlier) / len(later_to_earlier) if later_to_earlier else 0.0
                bsi_per_head.append(bsi)

            bsi_t = torch.tensor(bsi_per_head)
            bsi_per_layer.append(bsi_per_head)

            max_bsi = bsi_t.max().item()
            mean_bsi = bsi_t.mean().item()
            eb_per_layer.append(max_bsi - mean_bsi)

        eb_star = max(eb_per_layer)
        best_layer = eb_per_layer.index(eb_star)

        bsi_best = torch.tensor(bsi_per_layer[best_layer])
        bsi_clamped = torch.clamp(bsi_best, min=1e-10)
        probs = bsi_clamped / bsi_clamped.sum()
        entropy = -(probs * torch.log(probs)).sum().item()

        return {
            "eb_star": round(eb_star, 6),
            "eb_per_layer": [round(x, 6) for x in eb_per_layer],
            "best_layer": best_layer,
            "entropy": round(entropy, 6),
            "span_indices": span_indices,
            "n_term_tokens": n_term_tokens,
        }

    def _find_term_span(self, token_ids: list, term: str) -> tuple[int, int]:
        """Locate term token span. Returns (span_start, n_term_tokens)."""
        variants = []
        for form in [term, term.capitalize(), term.title()]:
            variants.append(self.tokenizer.encode(form, add_special_tokens=False))
            variants.append(self.tokenizer.encode(" " + form, add_special_tokens=False))

        seen = set()
        unique = []
        for v in variants:
            k = tuple(v)
            if k not in seen:
                seen.add(k)
                unique.append(v)

        for variant in unique:
            for i in range(len(token_ids) - len(variant) + 1):
                if token_ids[i: i + len(variant)] == variant:
                    return i, len(variant)

        # Fallback: character-level match
        decoded = [self.tokenizer.decode([t]) for t in token_ids]
        joined = "".join(decoded)
        char_pos = joined.lower().find(term.lower())
        n_base = len(unique[0]) if unique else 2
        if char_pos >= 0:
            cum = 0
            for idx, dt in enumerate(decoded):
                if cum >= char_pos:
                    return idx, n_base
                cum += len(dt)

        fallback = max(0, len(token_ids) - n_base - 5)
        logger.warning("Could not find span for '%s', using position %d", term, fallback)
        return fallback, n_base
