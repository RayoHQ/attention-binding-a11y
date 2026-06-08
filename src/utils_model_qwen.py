"""Utility for loading Qwen2.5-1.5B and extracting attention patterns.

Model: Qwen/Qwen2.5-1.5B (single final checkpoint — no pretraining ckpts available)
Architecture: LlamaForCausalLM-style with GQA; output_attentions=True supported.
"""

import logging
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

QWEN_MODEL_ID = "Qwen/Qwen2.5-1.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_KEYS = ["final"]
CHECKPOINT_MAP = {"final": QWEN_MODEL_ID}


def load_qwen(device: str = DEVICE):
    """Load Qwen2.5-1.5B final checkpoint."""
    logger.info("Loading %s ...", QWEN_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_ID,
        dtype=torch.float32,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    logger.info("Loaded Qwen2.5-1.5B successfully")
    return model, tokenizer


class QwenAttentionExtractor:
    """Extracts per-head attention patterns from Qwen2.5-1.5B via output_attentions.

    Same interface as SmolLM3AttentionExtractor — drop-in compatible.
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
            # GQA: repeat heads if n_key_value_heads < n_heads
            n_attn_heads = layer_attn.shape[1]
            bsi_per_head = []
            for head_idx in range(n_attn_heads):
                head_attn = layer_attn[0, head_idx]
                later_to_earlier = [
                    head_attn[dest, src].item()
                    for dest in span_indices
                    for src in span_indices
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

        return max(0, len(token_ids) - n_base - 1), n_base
