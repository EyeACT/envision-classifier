# preprocess_scripts

Preprocessing utilities that run *before* the classifier, not part of the
runtime classification package.

## What's here

- `llama_loader.py` — loads `fairdataihub/Llama-3.1-8B-Poster-Extraction` at
  4-bit NF4 quantization via bitsandbytes. Loader config mirrors
  `poster2json.extract._load_json_model` exactly so summaries produced by
  this pipeline are reproducible against the same checkpoint cited in the
  poster2json work.
- `summarize_descriptions.py` — task-scoped abstractive summarizer for
  dataset descriptions whose combined (title + description + keywords)
  length exceeds MPNet's 512-token budget. Preserves imaging modality,
  anatomical region, data format, and study subjects; drops author bios,
  funding boilerplate, and bibliography fragments.
- `run_validation_356.py` — apply the summarizer to the expert-validation
  set in `envision-discovery/eval/results/expert_validation_356_records.json`.
  Preserves originals in `description_original`; writes metadata per record.

## Reproducibility

- **Greedy decoding** (`do_sample=False`, all sampling params pinned). Output
  is deterministic given (input, model weights, prompt, hardware kernels).
- **Warmup + self-check**: a fixed reference string is summarized 3x
  back-to-back before any production generation; the SHA-256 of each output
  must match, else the run aborts.
- **Cache key**: SHA-256 of (input text, model id, prompt version, quant
  config). Any change to prompt or config invalidates the cache by design.
- **Version pins**: prompt version and quant config are written into every
  per-record metadata block.

## Running

From an envision-discovery checkout, with bitsandbytes, transformers, and
(optionally) flash-attn installed:

```bash
# Dry-run: report which records would be summarized, without loading the LLM
python envision-classifier/preprocess_scripts/run_validation_356.py --dry-run

# Real run: summarize, cache, write back to the JSON in place
python envision-classifier/preprocess_scripts/run_validation_356.py
```

Typical throughput on a single RTX 4090 is ~10–15 s/record including tokenize
and decode. First-run model load is ~80 s.

## Paper

Section 2.4.5 of the envision-discovery manuscript documents the
motivation, backbone comparison, and reproducibility guarantees that led to
this preprocessing step.
