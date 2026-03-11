# envision-classifier

SetFit few-shot classifier for identifying eye imaging datasets from scientific metadata.

Part of the [EyeACT](https://github.com/EyeACT) project by the [FAIR Data Innovations Hub](https://fairdataihub.org).

## Installation

```bash
pip install envision-classifier
```

## Python API

```python
from envision_classifier import EyeImagingClassifier

# Downloads model from HuggingFace on first use
clf = EyeImagingClassifier()

# Classify a single record
result = clf.classify("Retinal OCT dataset for diabetic retinopathy")
print(result)
# {'label': 'EYE_IMAGING', 'confidence': 0.999, 'probabilities': {...}}

# Classify a batch
results = clf.classify_batch([
    "Retinal fundus photography dataset for glaucoma screening",
    "COVID-19 genome sequencing data",
    {"title": "OCT images", "description": "Macular degeneration scans"},
])

# Use a local model instead of downloading
clf = EyeImagingClassifier(model_path="./my_model")
```

## CLI

After installing, the `envision-classifier` command is available:

```bash
# Classify a text string
envision-classifier classify --text "Retinal OCT dataset for diabetic retinopathy"

# Classify from a JSON file
envision-classifier classify records.json

# Pipe JSON via stdin
echo '{"title": "Fundus images", "description": "DR screening"}' | envision-classifier classify

# Train a new model from built-in training data
envision-classifier train --output ./my_model

# Show model info and training data counts
envision-classifier info
```

## Classification Labels

| Label | Description |
|-------|-------------|
| **EYE_IMAGING** | Actual eye imaging datasets (fundus, OCT, OCTA, cornea) |
| **EYE_SOFTWARE** | Code, tools, models for eye imaging (no actual data) |
| **EDGE_CASE** | Eye research papers, reviews, non-imaging data |
| **NEGATIVE** | Not eye-related |

## Model

- **Base model**: `sentence-transformers/all-mpnet-base-v2` (768-dim)
- **Training data**: 474 curated examples (77 EYE_IMAGING, 48 EYE_SOFTWARE, 79 EDGE_CASE, 270 NEGATIVE)
- **Test accuracy**: 0.937, **macro F1**: 0.902
- **Spot-check**: 29/33 (87.9%)
- **Model weights**: [fairdataihub/envision-eye-imaging-classifier](https://huggingface.co/fairdataihub/envision-eye-imaging-classifier)

## Related

- [envision-discovery](https://github.com/EyeACT/envision-discovery) -- Full pipeline (scraping + classification + export)
- [Model on HuggingFace](https://huggingface.co/fairdataihub/envision-eye-imaging-classifier)

## License

MIT
