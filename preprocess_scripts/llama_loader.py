"""Load fairdataihub/Llama-3.1-8B-Poster-Extraction with bitsandbytes 4-bit
NF4 quantization. Loader config mirrors `poster2json.extract._load_json_model`
exactly, so summaries produced here are reproducible against the same
checkpoint already cited in the poster2json work.

Why 4-bit NF4 (not int8 or fp16):
  - VRAM: ~5 GB on disk after quant, ~6 GB resident. Leaves room for the
    MPNet classifier and an OS desktop.
  - Quality: NF4 + double-quant + bf16 compute matches fp16 generation
    quality for instruction-following tasks within ~1% perplexity in
    published benchmarks (Dettmers et al., QLoRA, 2023).
  - Reproducibility: bitsandbytes quantization is deterministic given a
    pinned bitsandbytes version and the same hardware; greedy decoding
    on top makes the full pipeline byte-identical run-to-run.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)

JSON_MODEL_ID = "fairdataihub/Llama-3.1-8B-Poster-Extraction"

_model = None
_tokenizer = None


def _pick_device() -> tuple[str, int | str]:
    """Pick the GPU with the most free VRAM, or CPU."""
    if not torch.cuda.is_available():
        return "cpu", "cpu"
    best_id, best_free = 0, 0
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_id, best_free = i, free
    return f"cuda:{best_id}", best_id


def load_llama(
    model_id: str = JSON_MODEL_ID,
    quantization: str = "4bit",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Llama-3.1-8B (4-bit NF4 by default) and its tokenizer.

    Subsequent calls with the same args return the cached instance.

    Args:
        model_id: HuggingFace repo. Defaults to
            ``fairdataihub/Llama-3.1-8B-Poster-Extraction``.
        quantization: ``"4bit"`` (NF4 + double-quant + bf16 compute),
            ``"8bit"``, or ``"fp16"`` (bf16 weights, no quant).
    """
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    device, device_map_value = _pick_device()
    logger.info("Loading %s on %s (%s)", model_id, device, quantization)

    _tokenizer = AutoTokenizer.from_pretrained(model_id)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    model_kwargs: dict = {
        "device_map": device_map_value,
        "low_cpu_mem_usage": True,
    }

    try:
        import flash_attn  # noqa: F401

        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Using Flash Attention 2")
    except ImportError:
        logger.info("Flash Attention not available; using default attention")

    if quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "8bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "fp16":
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        raise ValueError(f"quantization must be 4bit|8bit|fp16, got {quantization!r}")

    _model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    _model.eval()
    logger.info("Model loaded")
    return _model, _tokenizer


def unload() -> None:
    global _model, _tokenizer
    if _model is not None:
        del _model
        _model = None
    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
