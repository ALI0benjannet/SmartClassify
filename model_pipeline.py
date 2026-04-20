"""Modular machine learning pipeline for the obesity dataset.

This module separates the notebook workflow into reusable steps:
data preparation, model training, evaluation, saving, and loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def load_arff_data(data_path: str | Path) -> pd.DataFrame:
    """Load an ARFF file into a pandas DataFrame.

    The dataset used in this project contains only simple numeric attributes,
    so a lightweight parser is enough and avoids an additional dependency.
    """

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    attributes: list[str] = []
    rows: list[list[str]] = []
    in_data_section = False

    with path.open("r", encoding="utf-8") as file_handle:
        for raw_line in file_handle:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue

            lower_line = line.lower()
            if lower_line.startswith("@attribute"):
                parts = line.split(None, 2)
                if len(parts) < 3:
                    raise ValueError(f"Invalid ARFF attribute line: {line}")
                attributes.append(parts[1].strip("'\""))
                continue

            if lower_line.startswith("@data"):
                in_data_section = True
                continue

            if in_data_section:
                rows.append([value.strip() for value in line.split(",")])

    if not attributes:
        raise ValueError(f"No ARFF attributes found in file: {path}")
    if not rows:
        raise ValueError(f"No data rows found in file: {path}")

    dataframe = pd.DataFrame(rows, columns=attributes)
    for column in dataframe.columns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    return dataframe


def prepare_data(
    data_path: str | Path,
    target_column: str = "Class",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load, clean, and split the dataset into train and test sets."""

    dataframe = load_arff_data(data_path)

    if target_column not in dataframe.columns:
        raise KeyError(f"Target column '{target_column}' not found in data.")

    features = dataframe.drop(columns=[target_column])
    target = dataframe[target_column]

    features = features.apply(pd.to_numeric, errors="coerce")
    if features.isnull().any().any():
        features = features.fillna(features.median(numeric_only=True))

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    return X_train, X_test, y_train, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: Dict[str, Any] | None = None,
) -> RandomForestClassifier:
    """Train a classification model on the provided training set."""

    params: Dict[str, Any] = {
        "n_estimators": 200,
        "random_state": 42,
        "class_weight": "balanced",
    }
    if model_params:
        params.update(model_params)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Evaluate the model and return a dictionary of metrics."""

    predictions = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            zero_division=0,
            output_dict=True,
        ),
    }


def save_model(model: RandomForestClassifier, model_path: str | Path) -> Path:
    """Save a trained model to disk using joblib."""

    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(model_path: str | Path) -> RandomForestClassifier:
    """Load a previously saved model from disk."""

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)
