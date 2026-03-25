# Changelog

## [0.2.0] - 2026-03-24

### Breaking

- Switched from 4-class to binary classification (EYE_IMAGING vs NEGATIVE)
- EYE_SOFTWARE and OTHER_EYE_DATA classes consolidated into NEGATIVE

### Changed

- Training data expanded to 891 examples from multiple repositories (Zenodo, Figshare, Dryad, Kaggle, NEI)
- Improved precision on Zenodo discovery (60 EYE_IMAGING from 515 records)
- Held-out test: EYE_IMAGING F1=0.936, Accuracy=0.961

## [0.1.2] - 2026-03-17

### Changed

- Renamed `EDGE_CASE` class to `OTHER_EYE_DATA`
- Renamed `prob_edge` output key to `prob_other_eye`
- Training data cleaned and expanded to 474 examples
- Improved spot-check accuracy to 87.9% (29/33)

## [0.1.0] - 2026-03-03

### Added

- Initial beta scaffold
- 4-class SetFit classifier (EYE_IMAGING, EYE_SOFTWARE, OTHER_EYE_DATA, NEGATIVE)
- CLI with `classify`, `train`, and `info` commands
- Auto-download of model weights from HuggingFace
- Batch classification support
