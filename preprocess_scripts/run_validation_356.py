"""Apply description summarization to eval/results/expert_validation_356_records.json.

Per-record behavior:
  - Compute combined classifier text (title + description + keywords) and
    its MPNet token count.
  - If <= 512 tokens: leave the record alone.
  - If > 512 tokens: summarize the description with the Llama summarizer,
    overwrite the `description` field with the summary, preserve the
    original under `description_original`, and attach a metadata block.

Run from envision-discovery repo root:
    python -m envision_classifier.preprocess_scripts.run_validation_356
or directly:
    python envision-classifier/preprocess_scripts/run_validation_356.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add the parent dir so sibling modules import as a flat namespace
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llama_loader import JSON_MODEL_ID, load_llama  # noqa: E402
from summarize_descriptions import (  # noqa: E402
    CLASSIFIER_TOKEN_BUDGET,
    SummaryCache,
    joined_classifier_text,
    mpnet_token_count,
    summarize_one,
    warmup_and_self_check,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    # envision-classifier/preprocess_scripts/run_validation_356.py
    # parents[2] = envision-discovery
    return here.parents[2]


def main() -> int:
    repo = find_repo_root()
    default_target = repo / "eval" / "results" / "expert_validation_356_records.json"
    default_cache = repo / "eval" / "results" / "summary_cache"

    p = argparse.ArgumentParser()
    p.add_argument("--target", type=Path, default=default_target)
    p.add_argument("--cache-dir", type=Path, default=default_cache)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be summarized without loading the model",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N qualifying records (debug)",
    )
    args = p.parse_args()

    log.info("Target: %s", args.target)
    log.info("Cache:  %s", args.cache_dir)

    with args.target.open() as f:
        records = json.load(f)

    # Identify records that need summarization
    needs = []
    for i, r in enumerate(records):
        text = joined_classifier_text(
            r.get("title", ""),
            r.get("description", ""),
            r.get("keywords"),
        )
        n_tok = mpnet_token_count(text)
        if n_tok > CLASSIFIER_TOKEN_BUDGET:
            needs.append((i, n_tok))

    log.info(
        "Records needing summarization: %d / %d (joined text > %d MPNet tokens)",
        len(needs),
        len(records),
        CLASSIFIER_TOKEN_BUDGET,
    )

    if not needs:
        log.info("Nothing to do.")
        return 0

    # Distribution
    over_count = {
        "513-700":   sum(1 for _, n in needs if 513 <= n <= 700),
        "701-1000":  sum(1 for _, n in needs if 701 <= n <= 1000),
        "1001-2000": sum(1 for _, n in needs if 1001 <= n <= 2000),
        "2001+":     sum(1 for _, n in needs if n > 2000),
    }
    log.info("Distribution of over-budget records: %s", over_count)

    if args.dry_run:
        log.info("--dry-run: stopping before model load")
        return 0

    if args.limit:
        needs = needs[: args.limit]
        log.info("--limit %d: processing only first %d", args.limit, len(needs))

    cache = SummaryCache(args.cache_dir)

    log.info("Loading Llama (4-bit NF4)...")
    model, tokenizer = load_llama(JSON_MODEL_ID, quantization="4bit")

    if not warmup_and_self_check(model, tokenizer):
        log.error("Aborting: pipeline is non-deterministic on this hardware")
        return 2

    summarized = 0
    cache_hits = 0
    failures = 0

    for k, (idx, n_tok) in enumerate(needs, 1):
        r = records[idx]
        original = r.get("description") or ""
        if not original:
            log.warning("[%d/%d] record idx=%d has empty description; skipping",
                        k, len(needs), idx)
            continue

        log.info(
            "[%d/%d] idx=%d source=%s/%s  joined=%d tok  desc=%d chars",
            k, len(needs), idx, r.get("source"), r.get("source_id"),
            n_tok, len(original),
        )
        try:
            payload = summarize_one(original, cache, model, tokenizer)
        except Exception as e:
            log.error("  failed: %s", e)
            failures += 1
            continue

        summary = payload["summary"]
        if not summary or len(summary) < 30:
            log.warning("  summary too short (%d chars); leaving record alone",
                        len(summary or ""))
            failures += 1
            continue

        cache_hits += int(payload.get("cache_hit", False))
        summarized += 1

        # Verify the new joined text fits the budget
        new_text = joined_classifier_text(
            r.get("title", ""), summary, r.get("keywords")
        )
        new_tok = mpnet_token_count(new_text)
        if new_tok > CLASSIFIER_TOKEN_BUDGET:
            log.warning("  summary still over budget: %d tokens (will still apply)",
                        new_tok)

        # Apply: preserve original, overwrite description, attach metadata
        r["description_original"] = original
        r["description"] = summary
        r["description_summary_meta"] = {
            "model_id": payload["model_id"],
            "prompt_version": payload["prompt_version"],
            "quant_config": payload["quant_config"],
            "input_sha256": payload["input_sha256"],
            "input_mpnet_tokens": payload["input_mpnet_tokens"],
            "summary_mpnet_tokens": payload["summary_mpnet_tokens"],
            "joined_mpnet_tokens_before": n_tok,
            "joined_mpnet_tokens_after": new_tok,
        }

        log.info(
            "  -> %d -> %d MPNet tokens (joined %d -> %d)  %s  %.1fs",
            payload["input_mpnet_tokens"],
            payload["summary_mpnet_tokens"],
            n_tok, new_tok,
            "[cache]" if payload.get("cache_hit") else "",
            payload.get("generation_seconds", 0.0),
        )

        # Incremental save every 10 non-cache-hit records: survives crashes
        if summarized % 10 == 0 and not payload.get("cache_hit"):
            with args.target.open("w") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            log.info("  [incremental save]")

    log.info(
        "Done. summarized=%d cache_hits=%d failures=%d",
        summarized, cache_hits, failures,
    )

    with args.target.open("w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    log.info("Wrote %s", args.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
