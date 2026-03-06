# envision-classifier

SetFit few-shot classifier for identifying eye imaging datasets from scientific metadata.

Part of the [EyeACT](https://github.com/EyeACT) project by the [FAIR Data Innovations Hub](https://fairdataihub.org).

## Installation

```bash
pip install git+https://github.com/EyeACT/envision-classifier.git
```

## Usage

```python
from envision_classifier import EyeImagingClassifier

clf = EyeImagingClassifier()
result = clf.classify("Retinal OCT dataset for diabetic retinopathy")
print(result)
# {'label': 'EYE_IMAGING', 'confidence': 0.999, 'probabilities': {...}}
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
- **Training data**: 474 curated examples
- **Test accuracy**: 0.937, **macro F1**: 0.902
- **Model weights**: [fairdataihub/envision-eye-imaging-classifier](https://huggingface.co/fairdataihub/envision-eye-imaging-classifier)

## License

MIT
