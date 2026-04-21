"""Command line entry point for the obesity ML pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlflow_utils import log_training_run
from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Run the obesity classification pipeline."
    )
    parser.add_argument(
        "--data-path",
        default=str(
            Path(__file__).resolve().parent / "archive (1)" / "Obesity_Dataset.arff"
        ),
        help="Path to the ARFF dataset.",
    )
    parser.add_argument(
        "--model-path",
        default=str(
            Path(__file__).resolve().parent / "artifacts" / "obesity_model.joblib"
        ),
        help="Path where the trained model will be saved or loaded.",
    )
    parser.add_argument(
        "--mode",
        choices=("train", "evaluate"),
        default="train",
        help="Train a new model or evaluate an existing one.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size.")
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for splitting."
    )
    return parser


def main() -> None:
    """Run the requested pipeline mode."""

    parser = build_parser()
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = prepare_data(
        args.data_path,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    if args.mode == "train":
        model = train_model(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        saved_path = save_model(model, args.model_path)
        mlflow_info = log_training_run(
            model=model,
            params={
                "model_type": "RandomForestClassifier",
                "test_size": args.test_size,
                "random_state": args.random_state,
                "n_estimators": model.n_estimators,
                "class_weight": str(model.class_weight),
                "data_path": str(args.data_path),
                "model_path": str(args.model_path),
            },
            metrics=metrics,
            training_data=X_train.assign(Class=y_train),
        )

        print(f"Model saved to: {saved_path}")
        print(f"MLflow run_id: {mlflow_info['run_id']}")
        print(f"MLflow tracking URI: {mlflow_info['tracking_uri']}")
        print(json.dumps(metrics, indent=2))
        return

    model = load_model(args.model_path)
    metrics = evaluate_model(model, X_test, y_test)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
