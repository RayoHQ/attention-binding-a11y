"""Model loading utilities with checkpoint support.

Uses HuggingFace `revision` parameter (not deprecated `checkpoint_value`)
to load specific Pythia training checkpoints.
"""

import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

logger = logging.getLogger(__name__)

# Verified available Pythia checkpoints (8 of original 10)
CHECKPOINT_STEPS = {
    0: "step0",
    15: "step15000",
    30: "step30000",
    60: "step60000",
    90: "step90000",
    120: "step120000",
    140: "step140000",
    143: "step143000",
}

VALID_SIZES = ("160m", "1b", "2.8b", "6.9b")


def load_pythia_with_checkpoint(
    size: str,
    step: str,
    device: Optional[str] = None,
) -> HookedTransformer:
    """Load Pythia model with a specific checkpoint revision.

    Args:
        size: Model size ("160m", "1b", "2.8b", "6.9b").
        step: Checkpoint revision (e.g., "step0", "step30000", "step143000").
        device: Device to load model on. Defaults to CUDA if available.

    Returns:
        HookedTransformer with loaded weights.

    Raises:
        ValueError: If size is not in VALID_SIZES.
    """
    if size not in VALID_SIZES:
        raise ValueError(f"Invalid size '{size}'. Choose from {VALID_SIZES}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = f"EleutherAI/pythia-{size}-deduped"
    logger.info("Loading %s at %s on %s", model_name, step, device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=step)
    tokenizer.pad_token = tokenizer.eos_token

    # pythia-1b-deduped has a broken pytorch_model.bin on HuggingFace
    # (same blob for all revisions). Its model.safetensors files are correct.
    # 160m and 2.8b lack safetensors, so they must use pytorch_model.bin.
    use_safetensors = (size == "1b")

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=step,
        dtype=torch.float32,
        use_safetensors=use_safetensors,
    )

    model = HookedTransformer.from_pretrained(
        model_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
    )

    logger.info("Loaded %s at %s successfully", model_name, step)
    return model
