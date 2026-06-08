"""SmolLM3-3B model loading and attention extraction utilities.

Uses HuggingFace output_attentions=True (same pattern as utils_model_olmo.py).
SmolLM3-3B is LLaMA-3 architecture — natively supported in transformers >= 4.44.
"""

import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Weights loaded from the checkpoints repo; tokenizer from the main model repo
SMOLLM3_MODEL_ID = "HuggingFaceTB/SmolLM3-3B-checkpoints"
SMOLLM3_TOKENIZER_ID = "HuggingFaceTB/SmolLM3-3B"

# 8 lifecycle checkpoints spanning the Stage-1 pretraining run (~3.44M steps).
# Stored as branches in HuggingFaceTB/SmolLM3-3B-checkpoints.
# Approximate token counts (batch ≈700M tokens/step at stage1 scale):
#   step-40k   ≈28B    (earliest available)
#   step-120k  ≈84B
#   step-400k  ≈280B
#   step-800k  ≈560B
#   step-1.2M  ≈840B
#   step-1.6M  ≈1.1T
#   step-2.4M  ≈1.7T
#   step-3.44M ≈2.4T  (final Stage-1 checkpoint)
SMOLLM3_CHECKPOINTS = {
    "step40k":   "stage1-step-40000",
    "step120k":  "stage1-step-120000",
    "step400k":  "stage1-step-400000",
    "step800k":  "stage1-step-800000",
    "step1200k": "stage1-step-1200000",
    "step1600k": "stage1-step-1600000",
    "step2400k": "stage1-step-2400000",
    "step3440k": "stage1-step-3440000",
}
# Ordered list for lifecycle iteration
CHECKPOINT_KEYS = list(SMOLLM3_CHECKPOINTS.keys())


def probe_checkpoints(model_id: str = SMOLLM3_MODEL_ID) -> list[str]:
    """Query HuggingFace API to list available revision branches for a model."""
    try:
        from huggingface_hub import list_repo_refs
        refs = list_repo_refs(model_id)
        branches = [b.name for b in refs.branches]
        logger.info("Available revisions for %s: %s", model_id, branches)
        return branches
    except Exception as e:
        logger.warning("Could not probe checkpoints for %s: %s", model_id, e)
        return []


def load_smollm3_with_checkpoint(
    checkpoint_key: str,
    device: Optional[str] = None,
):
    """Load SmolLM3-3B at a specific training checkpoint.

    Args:
        checkpoint_key: Key in SMOLLM3_CHECKPOINTS (e.g. 'step40k', 'step1m').
        device: Device string. Defaults to CUDA if available.

    Returns:
        (model, tokenizer) tuple — same interface as load_olmo_with_checkpoint.
    """
    if device is None:
        device = DEVICE

    if checkpoint_key not in SMOLLM3_CHECKPOINTS:
        raise ValueError(
            f"Unknown checkpoint '{checkpoint_key}'. "
            f"Valid keys: {list(SMOLLM3_CHECKPOINTS.keys())}"
        )

    revision = SMOLLM3_CHECKPOINTS[checkpoint_key]
    logger.info("Loading %s revision=%s on %s", SMOLLM3_MODEL_ID, revision, device)

    tokenizer = AutoTokenizer.from_pretrained(SMOLLM3_TOKENIZER_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        SMOLLM3_MODEL_ID,
        revision=revision,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    logger.info("Loaded SmolLM3-3B %s successfully", revision)
    return model, tokenizer


class SmolLM3AttentionExtractor:
    """Extracts per-head attention patterns from SmolLM3-3B via output_attentions.

    Same interface as OLMoAttentionHookExtractor — drop-in compatible.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = model.config.num_hidden_layers
        self.n_heads = model.config.num_attention_heads

    def extract_binding_for_prompt(self, prompt_text: str, term: str) -> dict:
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
        full_token_ids = inputs["input_ids"][0].tolist()

        span_start, n_term_tokens = self._find_term_span(full_token_ids, term)
        seq_len = len(full_token_ids)
        span_end = min(span_start + n_term_tokens, seq_len)
        span_indices = list(range(span_start, span_end))

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                return_dict=True,
            )

        eb_per_layer = []
        bsi_per_layer = []

        for layer_attn in outputs.attentions:
            layer_attn = layer_attn.float()
            bsi_per_head = []
            for head_idx in range(self.n_heads):
                head_attn = layer_attn[0, head_idx]
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
