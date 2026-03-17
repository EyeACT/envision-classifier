"""
envision-classifier CLI

Command-line interface for classifying eye imaging datasets.
"""

import json
import sys

import click


@click.group()
@click.version_option(package_name="envision-classifier")
def cli():
    """ENVISION: Eye imaging dataset classifier."""


@cli.command()
@click.argument("input_file", required=False, type=click.Path(exists=True))
@click.option("--text", "-t", help="Classify a text string directly.")
@click.option("--model", "-m", help="Path to trained model directory.")
@click.option("--device", "-d", help="Device (cuda/cpu).")
@click.option("--pretty/--compact", default=True, help="Pretty-print JSON output.")
def classify(input_file, text, model, device, pretty):
    """Classify metadata as eye imaging datasets.

    Accepts a JSON file, --text string, or stdin.
    """
    from .classifier import EyeImagingClassifier

    classifier = EyeImagingClassifier(model_path=model, device=device)
    indent = 2 if pretty else None

    if text:
        result = classifier.classify(text)
        click.echo(json.dumps(result, indent=indent))
    elif input_file:
        with open(input_file) as f:
            data = json.load(f)
        if isinstance(data, list):
            results = classifier.classify_batch(data)
        else:
            results = classifier.classify(data)
        click.echo(json.dumps(results, indent=indent))
    elif not sys.stdin.isatty():
        data = json.load(sys.stdin)
        if isinstance(data, list):
            results = classifier.classify_batch(data)
        else:
            results = classifier.classify(data)
        click.echo(json.dumps(results, indent=indent))
    else:
        click.echo("Provide a JSON file, --text, or pipe JSON via stdin.", err=True)
        raise SystemExit(1)


@cli.command()
@click.option("--output", "-o", help="Output directory for trained model.")
@click.option("--device", "-d", help="Device (cuda/cpu).")
def train(output, device):
    """Train a new classifier from built-in training data."""
    from .classifier import EyeImagingClassifier

    classifier = EyeImagingClassifier.train(output_dir=output, device=device)
    click.echo(f"\nModel ready. Labels: {classifier.LABELS}")


@cli.command()
def info():
    """Display classifier information."""
    from . import __version__
    from .classifier import (
        BASE_MODEL_NAME,
        HF_MODEL_REPO,
        LABELS,
        EYE_IMAGING_EXAMPLES,
        EYE_SOFTWARE_EXAMPLES,
        OTHER_EYE_DATA_EXAMPLES,
        NEGATIVE_EXAMPLES,
    )

    click.echo(f"envision-classifier v{__version__}")
    click.echo(f"Base model:       {BASE_MODEL_NAME}")
    click.echo(f"HuggingFace repo: {HF_MODEL_REPO}")
    click.echo(f"Labels:           {', '.join(LABELS)}")
    click.echo(f"Training data:    {len(EYE_IMAGING_EXAMPLES)} eye_imaging, "
               f"{len(EYE_SOFTWARE_EXAMPLES)} eye_software, "
               f"{len(OTHER_EYE_DATA_EXAMPLES)} other_eye_data, "
               f"{len(NEGATIVE_EXAMPLES)} negative")
