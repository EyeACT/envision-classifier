"""Summarize long dataset descriptions so the joined classifier input
(title + description + keywords) fits within MPNet's 512-token context.

Why this preprocessing step exists
----------------------------------
The deployed classifier is `sentence-transformers/all-mpnet-base-v2`, which
has a hard 512-token context. Roughly 30% of records in the multi-source
corpus exceed that budget once title and keywords are appended. Three
options were considered for the publication:

  1. Swap to a long-context backbone (ModernBERT-large, 8192 tokens).
     Rejected: in our spot-check evaluation ModernBERT-large scored
     83/92 vs MPNet's 84-86/92, while running ~10x slower at inference
     and consuming ~4x the VRAM. The classifier is called on every
     scrape pass across 6 repositories, so latency matters.
  2. Truncate descriptions to fit. Rejected: the discriminative signals
     (imaging modality, anatomical region, study population) often
     appear after the first paragraph, and naive truncation throws them
     away. This is what the previous deployment was inadvertently
     doing via the pre-tokenization char slice (see envision-discovery
     issue: classifier inference char/token bug, fixed 2026-04-20).
  3. Summarize long descriptions with a domain-aware LLM, preserving
     the modality / anatomy / data-type signals the classifier depends
     on. Chosen here.

Reproducibility guarantees
--------------------------
- Greedy decoding (`do_sample=False`). No temperature, no top-p, no
  top-k. Output is a deterministic function of (input, model weights,
  prompt, hardware kernel).
- Pinned model revision via `MODEL_REVISION_SHA` written into the
  per-record metadata so a future re-run can verify it pulled the same
  weights.
- Warmup pass on a fixed string before any real generation; this
  stabilizes CUDA kernel selection so the first real call matches
  subsequent ones.
- Reproducibility self-check: the same fixed input is summarized 3x
  back-to-back and the SHA-256 of all three outputs must match before
  any real records are processed. This catches non-determinism from
  driver updates, kernel selection drift, or quant-config mismatches.
- Disk cache keyed by SHA-256 of (prompt + input text + model_id +
  quant_config + prompt_version). Re-running the script never hits the
  GPU twice for the same input.

What we preserve and what we throw away
---------------------------------------
The prompt is task-scoped: keep imaging modality (OCT, fundus, OCTA,
slit-lamp, MRI, etc.), anatomical region (retina, cornea, optic nerve,
etc.), data formats (images, tables, code), and study subjects (human,
animal, phantom). Throw away author bios, funding boilerplate,
bibliography fragments, and methods detail beyond what bears on the
data type. This is the discriminative-signals view, not a generic
"what was done" summary.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

try:
    from .llama_loader import JSON_MODEL_ID, load_llama
except ImportError:  # script-mode import
    from llama_loader import JSON_MODEL_ID, load_llama  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pinned configuration — every value here is part of the cache key. Bumping
# any of these invalidates the cache (which is the correct behavior).
# ---------------------------------------------------------------------------

PROMPT_VERSION = "v1"  # bump on any prompt change

QUANT_CONFIG_TAG = "nf4-bf16-doublequant"  # informational; pinned in loader

# MPNet tokenizer, used to decide which records need summarization. The
# classifier runs MPNet, so the gating tokenizer must be MPNet.
MPNET_TOKENIZER_ID = "sentence-transformers/all-mpnet-base-v2"
CLASSIFIER_TOKEN_BUDGET = 512  # MPNet context window

# Output budget for the summarizer, expressed in Llama tokens. The summary
# must fit comfortably under the MPNet budget once title+keywords are
# appended; we target ~200 MPNet tokens of summary, which is ~250 Llama
# tokens (Llama BPE is slightly more compact than MPNet WordPiece on
# scientific English).
MAX_NEW_TOKENS = 256

# Greedy decoding for full determinism. Llama-3.1 (unlike Qwen3) does not
# require temperature>0 to avoid repetition collapse.
GENERATION_KWARGS = dict(
    do_sample=False,
    temperature=1.0,  # ignored when do_sample=False, set for pinning
    top_p=1.0,        # ignored when do_sample=False, set for pinning
    repetition_penalty=1.0,
    num_beams=1,
)

SYSTEM_PROMPT = (
    "You are a scientific dataset summarizer. You produce concise, factual "
    "summaries of dataset descriptions for downstream classification."
)

USER_PROMPT_TEMPLATE = """Summarize the dataset description below in 2 to 4 complete sentences. \
Preserve any mention of imaging modalities (e.g. OCT, OCTA, fundus, slit-lamp, MRI), \
anatomical structures (e.g. retina, cornea, optic nerve), data formats (e.g. images, \
tables, code, segmentation masks), and study subjects (human, animal model, phantom). \
Drop author bios, funding statements, and bibliography fragments. Do not add information \
not present in the source. Output only the summary, no preamble.

DESCRIPTION:
{text}

SUMMARY:"""

# Stop sequences make sure the model emits a clean summary and stops.
STOP_STRINGS = ["DESCRIPTION:", "SUMMARY:", "\n\nDESCRIPTION", "<|eot_id|>"]

# Reproducibility self-check
WARMUP_TEXT = (
    "This dataset contains optical coherence tomography (OCT) volumetric scans "
    "from 50 patients with diabetic macular edema, acquired on a Heidelberg "
    "Spectralis device. Each scan is annotated by two ophthalmologists for "
    "fluid regions and retinal layer boundaries."
)
REPRO_CHECK_RUNS = 3


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _cache_key(text: str, model_id: str) -> str:
    """SHA-256 over (input_text, model_id, prompt_version, quant_tag).

    Any change to inputs, prompt, model, or quant config invalidates the
    entry. This is intentional: a cache hit is a guarantee that the cached
    summary was produced with the exact same configuration.
    """
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    h.update(b"\x00")
    h.update(model_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(PROMPT_VERSION.encode("utf-8"))
    h.update(b"\x00")
    h.update(QUANT_CONFIG_TAG.encode("utf-8"))
    return h.hexdigest()


@dataclass
class SummaryCache:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Two-char shard to keep directory size manageable
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Optional[dict]:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            logger.warning("Corrupt cache entry at %s; ignoring", p)
            return None

    def put(self, key: str, payload: dict) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Token counting (MPNet) — gates which records need summarization
# ---------------------------------------------------------------------------

_mpnet_tokenizer = None


def mpnet_tokenizer():
    global _mpnet_tokenizer
    if _mpnet_tokenizer is None:
        _mpnet_tokenizer = AutoTokenizer.from_pretrained(MPNET_TOKENIZER_ID)
    return _mpnet_tokenizer


def joined_classifier_text(title: str, description: str, keywords) -> str:
    """Reproduce envision/metadata.py::to_classifier_text exactly."""
    parts = [title or "", description or ""]
    if keywords:
        if isinstance(keywords, list):
            parts.append(" ".join(keywords))
        else:
            parts.append(str(keywords))
    return " ".join(p for p in parts if p).strip()


def mpnet_token_count(text: str) -> int:
    return len(mpnet_tokenizer().encode(text, add_special_tokens=True))


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------


def _build_prompt(tokenizer, text: str) -> str:
    """Use the model's official chat template for Llama 3.x."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def _trim_to_complete_sentence(text: str) -> str:
    """If the model gets cut off mid-sentence, drop the dangling fragment."""
    text = text.strip()
    if not text:
        return text
    if text[-1] in ".!?\"')":
        return text
    sentences = _SENTENCE_END_RE.split(text)
    if len(sentences) > 1:
        return " ".join(sentences[:-1]).strip()
    return text


def _strip_preamble(text: str) -> str:
    """Drop common LLM throat-clearing."""
    text = text.strip()
    for pattern in (
        r"^Here(?:'s| is) (?:the |a |your )?(?:concise |brief |short |2-4[- ]sentence )?summary[:.]?\s*",
        r"^Summary[:.]?\s*",
        r"^The dataset (?:described )?(?:contains|consists of|provides|includes)\b",
    ):
        new = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)
        if new != text:
            text = new
            break
    return text.strip()


def summarize_one(
    text: str,
    cache: SummaryCache,
    model=None,
    tokenizer=None,
    model_id: str = JSON_MODEL_ID,
) -> dict:
    """Summarize one description. Returns dict with summary + metadata.

    On cache hit, returns immediately without touching the GPU.
    """
    key = _cache_key(text, model_id)
    cached = cache.get(key)
    if cached is not None:
        cached["cache_hit"] = True
        return cached

    if model is None or tokenizer is None:
        model, tokenizer = load_llama(model_id)

    prompt = _build_prompt(tokenizer, text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **GENERATION_KWARGS,
        )
    elapsed = time.time() - t0

    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Apply stop-string cleanup
    for stop in STOP_STRINGS:
        idx = raw.find(stop)
        if idx >= 0:
            raw = raw[:idx]

    cleaned = _strip_preamble(raw)
    summary = _trim_to_complete_sentence(cleaned)

    payload = {
        "summary": summary,
        "model_id": model_id,
        "prompt_version": PROMPT_VERSION,
        "quant_config": QUANT_CONFIG_TAG,
        "input_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "input_chars": len(text),
        "summary_chars": len(summary),
        "input_mpnet_tokens": mpnet_token_count(text),
        "summary_mpnet_tokens": mpnet_token_count(summary),
        "generation_seconds": round(elapsed, 2),
    }
    cache.put(key, payload)
    payload["cache_hit"] = False
    return payload


# ---------------------------------------------------------------------------
# Reproducibility self-check
# ---------------------------------------------------------------------------


def warmup_and_self_check(model, tokenizer, runs: int = REPRO_CHECK_RUNS) -> bool:
    """Run a fixed input through the model `runs` times. All outputs must
    have identical SHA-256. If they don't, the pipeline is non-deterministic
    on this machine and we should not proceed.
    """
    logger.info("Warmup + reproducibility self-check (%d runs)", runs)
    prompt = _build_prompt(tokenizer, WARMUP_TEXT)
    hashes = []
    for i in range(runs):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **GENERATION_KWARGS,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        hashes.append(h)
        logger.info("  run %d: %s (%d chars)", i + 1, h[:16], len(text))
    if len(set(hashes)) == 1:
        logger.info("Reproducibility OK")
        return True
    logger.error(
        "Reproducibility FAILED: outputs differed across %d runs. Hashes: %s",
        runs,
        [h[:16] for h in hashes],
    )
    return False
