"""FastAPI application exposing prediction and retraining endpoints.

Example REST request for prediction:
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"Sex\":2,\"Age\":21,\"Height\":170,\"Overweight_Obese_Family\":2,\"Consumption_of_Fast_Food\":2,\"Frequency_of_Consuming_Vegetables\":3,\"Number_of_Main_Meals_Daily\":2,\"Food_Intake_Between_Meals\":2,\"Smoking\":2,\"Liquid_Intake_Daily\":2,\"Calculation_of_Calorie_Intake\":2,\"Physical_Excercise\":3,\"Schedule_Dedicated_to_Technology\":3,\"Type_of_Transportation_Used\":4}"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "archive (1)" / "Obesity_Dataset.arff"
DEFAULT_MODEL_PATH = BASE_DIR / "artifacts" / "obesity_model.joblib"


class PredictionRequest(BaseModel):
    Sex: int = Field(..., ge=1, le=2)
    Age: int = Field(..., ge=1)
    Height: int = Field(..., ge=1)
    Overweight_Obese_Family: int
    Consumption_of_Fast_Food: int
    Frequency_of_Consuming_Vegetables: int
    Number_of_Main_Meals_Daily: int
    Food_Intake_Between_Meals: int
    Smoking: int
    Liquid_Intake_Daily: int
    Calculation_of_Calorie_Intake: int
    Physical_Excercise: int
    Schedule_Dedicated_to_Technology: int
    Type_of_Transportation_Used: int


class PredictionResponse(BaseModel):
    predicted_class: int


class RetrainRequest(BaseModel):
    data_path: str = str(DEFAULT_DATA_PATH)
    model_path: str = str(DEFAULT_MODEL_PATH)
    test_size: float = Field(0.2, gt=0.0, lt=1.0)
    random_state: int = 42


class RetrainResponse(BaseModel):
    message: str
    model_path: str
    metrics: dict[str, Any]


app = FastAPI(
    title="Obesity Model API",
    description="REST API to predict obesity class and retrain the model.",
    version="1.0.0",
)

_model: RandomForestClassifier | None = None


def _load_model_once(model_path: Path = DEFAULT_MODEL_PATH) -> RandomForestClassifier:
    global _model
    if _model is None:
        _model = load_model(model_path)
    return _model


def _to_dataframe(payload: PredictionRequest) -> pd.DataFrame:
    # Keep a deterministic feature order matching the training columns.
    return pd.DataFrame(
        [
            {
                "Sex": payload.Sex,
                "Age": payload.Age,
                "Height": payload.Height,
                "Overweight_Obese_Family": payload.Overweight_Obese_Family,
                "Consumption_of_Fast_Food": payload.Consumption_of_Fast_Food,
                "Frequency_of_Consuming_Vegetables": payload.Frequency_of_Consuming_Vegetables,
                "Number_of_Main_Meals_Daily": payload.Number_of_Main_Meals_Daily,
                "Food_Intake_Between_Meals": payload.Food_Intake_Between_Meals,
                "Smoking": payload.Smoking,
                "Liquid_Intake_Daily": payload.Liquid_Intake_Daily,
                "Calculation_of_Calorie_Intake": payload.Calculation_of_Calorie_Intake,
                "Physical_Excercise": payload.Physical_Excercise,
                "Schedule_Dedicated_to_Technology": payload.Schedule_Dedicated_to_Technology,
                "Type_of_Transportation_Used": payload.Type_of_Transportation_Used,
            }
        ]
    )


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Obesity prediction API is running."}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        model = _load_model_once()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=(
                "Model not found. Train first with main.py or call /retrain. "
                f"Expected path: {DEFAULT_MODEL_PATH}"
            ),
        ) from exc

    features = _to_dataframe(payload)
    prediction = int(model.predict(features)[0])
    return PredictionResponse(predicted_class=prediction)


@app.post("/retrain", response_model=RetrainResponse)
def retrain(request: RetrainRequest) -> RetrainResponse:
    global _model

    data_path = Path(request.data_path)
    model_path = Path(request.model_path)

    try:
        X_train, X_test, y_train, y_test = prepare_data(
            data_path=data_path,
            test_size=request.test_size,
            random_state=request.random_state,
        )
        model = train_model(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        save_model(model, model_path)
    except Exception as exc:  # pragma: no cover - defensive API error handling
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _model = model
    return RetrainResponse(
        message="Model retrained and saved successfully.",
        model_path=str(model_path),
        metrics=metrics,
    )


@app.get("/example-request")
def example_request() -> dict[str, str]:
    return {
        "curl": (
            'curl -X POST "http://127.0.0.1:8000/predict" '
            '-H "Content-Type: application/json" '
            '-d "{\\"Sex\\":2,\\"Age\\":21,\\"Height\\":170,'
            '\\"Overweight_Obese_Family\\":2,\\"Consumption_of_Fast_Food\\":2,'
            '\\"Frequency_of_Consuming_Vegetables\\":3,'
            '\\"Number_of_Main_Meals_Daily\\":2,'
            '\\"Food_Intake_Between_Meals\\":2,\\"Smoking\\":2,'
            '\\"Liquid_Intake_Daily\\":2,\\"Calculation_of_Calorie_Intake\\":2,'
            '\\"Physical_Excercise\\":3,'
            '\\"Schedule_Dedicated_to_Technology\\":3,'
            '\\"Type_of_Transportation_Used\\":4}"'
        )
    }
