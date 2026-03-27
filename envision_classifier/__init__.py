"""
envision-classifier: Eye Imaging Dataset Classifier

A binary SetFit classifier for detecting eye imaging datasets:
  - EYE_IMAGING: Actual eye imaging datasets (fundus, OCT, OCTA, etc.)
  - NEGATIVE: Everything else (software, non-imaging eye data, unrelated domains)

Usage:
    >>> from envision_classifier import EyeImagingClassifier
    >>> clf = EyeImagingClassifier()
    >>> clf.classify("Retinal OCT dataset for diabetic retinopathy")
    {'label': 'EYE_IMAGING', 'confidence': 0.999, 'probabilities': {...}}
"""

__version__ = "0.1.0"
__author__ = "James O'Neill"

from .classifier import EyeImagingClassifier, LABELS

__all__ = [
    "EyeImagingClassifier",
    "LABELS",
]
