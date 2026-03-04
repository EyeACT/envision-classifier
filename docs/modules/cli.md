# CLI Reference

## Commands

### classify

Classify metadata as eye imaging datasets.

```bash
# Classify a text string
envision-classifier classify --text "Retinal OCT dataset"

# Classify from JSON file
envision-classifier classify input.json

# Pipe JSON via stdin
echo '{"title": "Fundus images"}' | envision-classifier classify
```

### train

Train a new classifier from built-in training data.

```bash
envision-classifier train
envision-classifier train --output ./my_model --device cuda
```

### info

Display classifier information.

```bash
envision-classifier info
```
