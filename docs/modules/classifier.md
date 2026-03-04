# Classifier

## EyeImagingClassifier

The core classifier for detecting eye imaging datasets using SetFit few-shot learning.

### Quick Start

```python
from envision_classifier import EyeImagingClassifier

clf = EyeImagingClassifier()
result = clf.classify("Retinal OCT dataset for diabetic retinopathy")
print(result)
# {'label': 'EYE_IMAGING', 'confidence': 0.999, 'probabilities': {...}}
```

### Classification Labels

| Label | Description |
|-------|-------------|
| `EYE_IMAGING` | Actual eye imaging datasets (fundus, OCT, OCTA, cornea, etc.) |
| `EYE_SOFTWARE` | Code, tools, models for eye imaging (no actual data) |
| `EDGE_CASE` | Eye research papers, reviews, borderline items |
| `NEGATIVE` | Unrelated domains |

### Batch Classification

```python
records = [
    "Retinal fundus photography dataset for glaucoma screening",
    "COVID-19 genome sequencing data",
    {"title": "OCT images", "description": "Macular degeneration scans"},
]
results = clf.classify_batch(records)
```

### Training a New Model

```python
clf = EyeImagingClassifier.train(output_dir="./my_model")
```
