"""Tests for envision_classifier."""

from envision_classifier.classifier import (
    LABELS,
    EYE_IMAGING_EXAMPLES,
    EYE_SOFTWARE_EXAMPLES,
    OTHER_EYE_DATA_EXAMPLES,
    NEGATIVE_EXAMPLES,
)


def test_labels():
    assert len(LABELS) == 4
    assert "EYE_IMAGING" in LABELS
    assert "NEGATIVE" in LABELS


def test_training_data_not_empty():
    assert len(EYE_IMAGING_EXAMPLES) > 0
    assert len(EYE_SOFTWARE_EXAMPLES) > 0
    assert len(OTHER_EYE_DATA_EXAMPLES) > 0
    assert len(NEGATIVE_EXAMPLES) > 0


def test_extract_text():
    from envision_classifier.classifier import EyeImagingClassifier

    text = EyeImagingClassifier.extract_text({
        "title": "Retinal OCT",
        "description": "A dataset of <b>OCT</b> images",
        "keywords": ["retina", "OCT"],
    })
    assert "Retinal OCT" in text
    assert "OCT" in text
    assert "<b>" not in text


def test_strip_html():
    from envision_classifier.classifier import EyeImagingClassifier

    assert EyeImagingClassifier.strip_html("<p>Hello</p>") == "Hello"
    assert EyeImagingClassifier.strip_html("") == ""
    assert EyeImagingClassifier.strip_html(None) == ""
