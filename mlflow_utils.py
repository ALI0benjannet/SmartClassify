"""Utilities to configure and log MLflow runs for the obesity project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TRACKING_DIR = BASE_DIR / "mlruns"
DEFAULT_EXPERIMENT_NAME = "obesity-classification"


def configure_mlflow(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tracking_dir: Path = DEFAULT_TRACKING_DIR,
) -> str:
    """Configure local MLflow tracking and return the tracking URI."""

    tracking_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = tracking_dir.resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return tracking_uri


def log_training_run(
    model: Any,
    params: dict[str, Any],
    metrics: dict[str, Any],
    model_artifact_path: str = "model",
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> dict[str, str]:
    """Log parameters, metrics, and model to MLflow."""

    tracking_uri = configure_mlflow(experiment_name=experiment_name)

    scalar_metrics: dict[str, float] = {}
    if "accuracy" in metrics:
        scalar_metrics["accuracy"] = float(metrics["accuracy"])

    report = metrics.get("classification_report", {})
    macro_avg = report.get("macro avg", {})
    weighted_avg = report.get("weighted avg", {})

    if "f1-score" in macro_avg:
        scalar_metrics["macro_f1"] = float(macro_avg["f1-score"])
    if "f1-score" in weighted_avg:
        scalar_metrics["weighted_f1"] = float(weighted_avg["f1-score"])

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(scalar_metrics)
        mlflow.sklearn.log_model(model, model_artifact_path)

        return {
            "run_id": run.info.run_id,
            "experiment_name": experiment_name,
            "tracking_uri": tracking_uri,
        }
