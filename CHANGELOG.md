# Changelog

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
