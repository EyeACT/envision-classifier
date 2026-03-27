# Changelog

## 0.1.0 (2026-03-27)


### Features

* binary classifier v0.2.0 (EYE_IMAGING vs NEGATIVE) ([e3a34fb](https://github.com/EyeACT/envision-classifier/commit/e3a34fb082e8d77f2b1bde79552dd3b2f1fd0b26))
* initial beta scaffold with SetFit classifier, CLI, and CI/CD ([e0c214f](https://github.com/EyeACT/envision-classifier/commit/e0c214fabab9eb09954f3115ac830f17213b82d2))
* initial beta scaffold with SetFit classifier, CLI, and CI/CD ([e9f0f6d](https://github.com/EyeACT/envision-classifier/commit/e9f0f6de0faa381a668e9621161c8b254e98632a))
* memory optimizations for low-RAM servers ([7add3a2](https://github.com/EyeACT/envision-classifier/commit/7add3a2039453b6d4099c75a3f8d08b704ad9c6c))


### Bug Fixes

* add print statements for model loading and encoding errors ([a91e7f5](https://github.com/EyeACT/envision-classifier/commit/a91e7f541d498a09a036f81573fc0563125d9829))


### Documentation

* expand README with CLI usage and batch examples ([6635a36](https://github.com/EyeACT/envision-classifier/commit/6635a36aedead65522467623d86efefba92d346d))
* update install to pip install from PyPI ([029389e](https://github.com/EyeACT/envision-classifier/commit/029389e4e08a31802f7dde71e528c4bc0c500640))

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
