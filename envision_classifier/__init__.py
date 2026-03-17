"""
envision-classifier: Eye Imaging Dataset Classifier

A 4-class SetFit classifier for detecting eye imaging datasets:
  - EYE_IMAGING: Actual eye imaging datasets (fundus, OCT, OCTA, etc.)
  - EYE_SOFTWARE: Code, models, tools for eye imaging
  - OTHER_EYE_DATA: Eye research papers, reviews, non-imaging data
  - NEGATIVE: Unrelated domains

Usage:
    >>> from envision_classifier import EyeImagingClassifier
    >>> clf = EyeImagingClassifier()
    >>> clf.classify("Retinal OCT dataset for diabetic retinopathy")
    {'label': 'EYE_IMAGING', 'confidence': 0.999, 'probabilities': {...}}
"""

__version__ = "0.1.2"
__author__ = "James O'Neill"

from .classifier import EyeImagingClassifier, LABELS

__all__ = [
    "EyeImagingClassifier",
    "LABELS",
]
