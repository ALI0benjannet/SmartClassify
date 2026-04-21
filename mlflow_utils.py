"""Utilities to configure and log MLflow runs for the obesity project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import mlflow.data
import mlflow.genai.datasets
import mlflow.sklearn
from mlflow.tracking import MlflowClient

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TRACKING_DB = BASE_DIR / "mlflow.db"
DEFAULT_ARTIFACTS_DIR = BASE_DIR / "mlartifacts"
DEFAULT_EXPERIMENT_NAME = "obesity-classification"
DEFAULT_EVALUATION_DATASET_NAME = "obesity-evaluation"


def ensure_evaluation_dataset(experiment_id: str) -> None:
    """Create a managed evaluation dataset linked to the project experiment."""

    try:
        dataset = mlflow.genai.datasets.get_dataset(name=DEFAULT_EVALUATION_DATASET_NAME)
    except Exception:
        mlflow.genai.datasets.create_dataset(
            name=DEFAULT_EVALUATION_DATASET_NAME,
            experiment_id=experiment_id,
        )
        return

    if experiment_id not in dataset.experiment_ids:
        mlflow.genai.datasets.add_dataset_to_experiments(
            dataset.dataset_id,
            [experiment_id],
        )


def configure_mlflow(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tracking_db: Path = DEFAULT_TRACKING_DB,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
) -> str:
    """Configure local MLflow tracking and return the tracking URI."""

    tracking_db.parent.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{tracking_db.resolve().as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=artifacts_dir.resolve().as_uri(),
        )
        experiment = client.get_experiment(experiment_id)

    mlflow.set_experiment(experiment_name)
    if experiment is not None:
        ensure_evaluation_dataset(experiment.experiment_id)
    return tracking_uri


def log_training_run(
    model: Any,
    params: dict[str, Any],
    metrics: dict[str, Any],
    training_data: Any | None = None,
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

        if training_data is not None:
            dataset = mlflow.data.from_pandas(
                training_data,
                name="obesity-training-data",
            )
            mlflow.log_input(dataset, context="training")

        mlflow.sklearn.log_model(model, model_artifact_path)

        return {
            "run_id": run.info.run_id,
            "experiment_name": experiment_name,
            "tracking_uri": tracking_uri,
        }
