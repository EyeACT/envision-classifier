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
# {'label': 'EYE_IMAGING', 'confidence': 0.98,
#  'probabilities': {'EYE_IMAGING': 0.98, 'NEGATIVE': 0.02}}

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
| **NEGATIVE** | Everything else (software, non-imaging eye data, unrelated domains) |

## Model

- **Base model**: `sentence-transformers/all-mpnet-base-v2` (768-dim)
- **Training data**: 891 curated examples (262 EYE_IMAGING, 629 NEGATIVE) from Zenodo, Figshare, Dryad, Kaggle, and NEI
- **Test accuracy**: 0.961, **EYE_IMAGING F1**: 0.936
- **Spot-check**: 30/33 (90.9%)
- **Model weights**: [fairdataihub/envision-eye-imaging-classifier](https://huggingface.co/fairdataihub/envision-eye-imaging-classifier)

## Multi-Repository Results

Applied across 6 repositories via [envision-discovery](https://github.com/EyeACT/envision-discovery):

| Source | EYE_IMAGING | NEGATIVE | Total |
|--------|-------------|----------|-------|
| Zenodo | 60 | 455 | 515 |
| DataCite | 752 | 1,084 | 1,836 |
| Figshare | 1,049 | 951 | 2,000 |
| Kaggle | 248 | 484 | 732 |
| Dryad | 32 | 57 | 89 |
| NEI | 686 | 976 | 1,662 |

Classification is based on metadata only (titles, descriptions, keywords, and file types inspected inside archives via HTTP Range requests) -- no dataset files are downloaded.

## Related

- [envision-discovery](https://github.com/EyeACT/envision-discovery) -- Full pipeline (scraping + classification + export)
- [Model on HuggingFace](https://huggingface.co/fairdataihub/envision-eye-imaging-classifier)

## License

MIT
