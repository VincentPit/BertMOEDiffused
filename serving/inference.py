"""
MLflow inference utilities — load models from the registry and generate text.

Usage:

    from serving.inference import load_model_from_registry, generate_text

    model = load_model_from_registry("BertMoEDiffusion", stage="Production")
    texts = generate_text(model, prompts=["The weather today is"], steps=100)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import mlflow
import mlflow.pytorch
import pandas as pd

logger = logging.getLogger(__name__)


def load_model_from_registry(
    model_name: str,
    version: Optional[int] = None,
    alias: Optional[str] = None,
    tracking_uri: str = "mlruns",
) -> mlflow.pyfunc.PyFuncModel:
    """Load a model from the MLflow Model Registry.

    Args:
        model_name:   Registered model name.
        version:      Specific version number (mutually exclusive with alias).
        alias:        Model alias e.g. "champion", "challenger" (mutually exclusive with version).
        tracking_uri: MLflow tracking URI.

    Returns:
        Loaded PyFunc model ready for `.predict()`.
    """
    mlflow.set_tracking_uri(tracking_uri)

    if alias:
        model_uri = f"models:/{model_name}@{alias}"
    elif version:
        model_uri = f"models:/{model_name}/{version}"
    else:
        model_uri = f"models:/{model_name}/latest"

    logger.info(f"Loading model from registry: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    logger.info("Model loaded successfully.")
    return model


def load_pytorch_model_from_registry(
    model_name: str,
    version: Optional[int] = None,
    alias: Optional[str] = None,
    tracking_uri: str = "mlruns",
):
    """Load the raw PyTorch model (not wrapped in PyFunc) from registry."""
    mlflow.set_tracking_uri(tracking_uri)

    if alias:
        model_uri = f"models:/{model_name}@{alias}"
    elif version:
        model_uri = f"models:/{model_name}/{version}"
    else:
        model_uri = f"models:/{model_name}/latest"

    logger.info(f"Loading PyTorch model from registry: {model_uri}")
    model = mlflow.pytorch.load_model(model_uri)
    logger.info("PyTorch model loaded successfully.")
    return model


def generate_text(
    model: mlflow.pyfunc.PyFuncModel,
    prompts: List[str],
    mode: str = "generate",
    steps: int = 100,
) -> List[str]:
    """Generate text using a loaded MLflow model.

    Args:
        model:   Loaded PyFunc model.
        prompts: List of input prompts.
        mode:    "generate" or "infill".
        steps:   Number of diffusion steps.

    Returns:
        List of generated text strings.
    """
    input_df = pd.DataFrame({
        "prompt": prompts,
        "mode": [mode] * len(prompts),
        "steps": [steps] * len(prompts),
    })
    result = model.predict(input_df)
    return result["generated_text"].tolist()
