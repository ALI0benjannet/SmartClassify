"""FastAPI application exposing prediction and retraining endpoints.

Example REST request for prediction:
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"Sex\":2,\"Age\":21,\"Height\":170,\"Overweight_Obese_Family\":2,\"Consumption_of_Fast_Food\":2,\"Frequency_of_Consuming_Vegetables\":3,\"Number_of_Main_Meals_Daily\":2,\"Food_Intake_Between_Meals\":2,\"Smoking\":2,\"Liquid_Intake_Daily\":2,\"Calculation_of_Calorie_Intake\":2,\"Physical_Excercise\":3,\"Schedule_Dedicated_to_Technology\":3,\"Type_of_Transportation_Used\":4}"
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier

from mlflow_utils import configure_mlflow, log_training_run
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
DEFAULT_TRACKING_DB = BASE_DIR / "mlflow.db"

CLASS_LABELS = {
    1: "Poids insuffisant",
    2: "Poids normal",
    3: "Surpoids",
    4: "Obesite",
}


class PredictionRequest(BaseModel):
    Sex: int = Field(..., ge=1, le=2)
    Age: int = Field(..., ge=1)
    Height: int = Field(..., ge=1)
    Weight: float | None = Field(default=None, gt=0)
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
    predicted_label: str
    confidence: float
    model_predicted_class: int
    model_predicted_label: str
    bmi: float | None = None
    bmi_class: int | None = None
    decision_source: str
    reason: str | None = None


class RetrainRequest(BaseModel):
    data_path: str = str(DEFAULT_DATA_PATH)
    model_path: str = str(DEFAULT_MODEL_PATH)
    test_size: float = Field(0.2, gt=0.0, lt=1.0)
    random_state: int = 42


class RetrainResponse(BaseModel):
    message: str
    model_path: str
    metrics: dict[str, Any]


class DashboardResponse(BaseModel):
    api_status: str
    mlflow_status: str
    latest_run_id: str | None
    latest_run_status: str | None
    runs: int
    metrics: int
    datasets: int
    inputs: int
    evaluation_datasets: int
    traces: int


app = FastAPI(
    title="Obesity Model API",
    description="REST API to predict obesity class and retrain the model.",
    version="1.0.0",
)

configure_mlflow()

_model: RandomForestClassifier | None = None


@app.middleware("http")
async def mlflow_request_observability(request: Request, call_next):
    """Log one MLflow trace span per HTTP request with real request metadata."""

    started = time.perf_counter()
    span_name = f"{request.method} {request.url.path}"

    with mlflow.start_span(
        name=span_name,
        span_type="HTTP",
        attributes={
            "http.method": request.method,
            "http.path": request.url.path,
        },
    ):
        response = await call_next(request)

    duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
    response.headers["X-Process-Time-Ms"] = str(duration_ms)
    return response


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


def _compute_bmi(weight_kg: float | None, height_cm: int) -> float | None:
    if weight_kg is None or weight_kg <= 0 or height_cm <= 0:
        return None
    height_m = height_cm / 100.0
    return float(weight_kg / (height_m * height_m))


def _bmi_to_class(bmi: float) -> int:
    if bmi < 18.5:
        return 1
    if bmi < 25.0:
        return 2
    if bmi < 30.0:
        return 3
    return 4


def _read_dashboard_snapshot() -> dict[str, Any]:
    """Read live counts and latest run info from the MLflow SQLite database."""

    if not DEFAULT_TRACKING_DB.exists():
        return {
            "api_status": "up",
            "mlflow_status": "missing-db",
            "latest_run_id": None,
            "latest_run_status": None,
            "runs": 0,
            "metrics": 0,
            "datasets": 0,
            "inputs": 0,
            "evaluation_datasets": 0,
            "traces": 0,
        }

    conn = sqlite3.connect(DEFAULT_TRACKING_DB)
    cur = conn.cursor()

    counts: dict[str, int] = {}
    for table in [
        "runs",
        "metrics",
        "datasets",
        "inputs",
        "evaluation_datasets",
        "trace_info",
    ]:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        counts[table] = int(cur.fetchone()[0])

    cur.execute(
        "SELECT run_uuid, status FROM runs ORDER BY start_time DESC LIMIT 1"
    )
    latest_run = cur.fetchone()
    conn.close()

    return {
        "api_status": "up",
        "mlflow_status": "up",
        "latest_run_id": latest_run[0] if latest_run else None,
        "latest_run_status": latest_run[1] if latest_run else None,
        "runs": counts["runs"],
        "metrics": counts["metrics"],
        "datasets": counts["datasets"],
        "inputs": counts["inputs"],
        "evaluation_datasets": counts["evaluation_datasets"],
        "traces": counts["trace_info"],
    }


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Obesity prediction API is running."}


@app.get("/stats", response_model=DashboardResponse)
def stats() -> DashboardResponse:
    """Return live project data for the dashboard page."""

    return DashboardResponse(**_read_dashboard_snapshot())


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    """Live project dashboard showing real MLflow and API data."""

    html = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Obesity Project Dashboard</title>
    <style>
        :root {
            --bg: #f5f7fb;
            --card: #ffffff;
            --text: #122033;
            --muted: #62708a;
            --accent: #1565d8;
            --accent-soft: #e8f1ff;
            --border: #dbe3f0;
        }
        body { margin: 0; font-family: Arial, sans-serif; background: linear-gradient(180deg, #eef4ff 0%, var(--bg) 35%, var(--bg) 100%); color: var(--text); }
        .wrap { max-width: 1180px; margin: 0 auto; padding: 32px 20px 48px; }
        .hero { display: flex; justify-content: space-between; align-items: end; gap: 20px; margin-bottom: 24px; }
        .hero h1 { margin: 0; font-size: 34px; }
        .hero p { margin: 8px 0 0; color: var(--muted); }
        .pill { display: inline-flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: 999px; background: var(--accent-soft); color: var(--accent); font-weight: 700; }
        .grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; margin-top: 18px; }
        .card { background: var(--card); border: 1px solid var(--border); border-radius: 18px; padding: 18px; box-shadow: 0 10px 30px rgba(18, 32, 51, 0.06); }
        .card h2 { margin: 0 0 10px; font-size: 18px; }
        .value { font-size: 34px; font-weight: 800; }
        .small { color: var(--muted); font-size: 13px; margin-top: 8px; line-height: 1.5; }
        .status { display: inline-block; margin-top: 10px; padding: 6px 10px; border-radius: 999px; background: #edf7ee; color: #1f7a2e; font-weight: 700; }
        .status.bad { background: #fdecec; color: #b42318; }
        .span { color: var(--accent); font-weight: 700; }
        .wide { grid-column: span 3; }
        .controls { display: flex; gap: 12px; flex-wrap: wrap; }
        .button { cursor: pointer; border: none; border-radius: 12px; padding: 10px 14px; font-weight: 700; }
        .button.primary { background: var(--accent); color: white; }
        .button.secondary { background: white; color: var(--text); border: 1px solid var(--border); }
        code { background: #f3f6fb; padding: 2px 6px; border-radius: 6px; }
        @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } .wide { grid-column: span 1; } .hero { flex-direction: column; align-items: start; } }
    </style>
</head>
<body>
    <div class="wrap">
        <div class="hero">
            <div>
                <div class="pill">Live Project Dashboard</div>
                <h1>Obesity ML Project</h1>
                <p>Real data from the API, MLflow database, traces, and datasets.</p>
            </div>
            <div class="controls">
                <button class="button primary" onclick="refreshStats()">Refresh data</button>
                <button class="button secondary" onclick="window.open('/docs', '_blank')">Open Swagger</button>
                <button class="button secondary" onclick="window.open('http://127.0.0.1:5000', '_blank')">Open MLflow</button>
            </div>
        </div>

        <div class="grid" id="cards">
            <div class="card"><h2>API status</h2><div class="value" id="apiStatus">...</div><div class="small">Live endpoint: <code>/stats</code></div></div>
            <div class="card"><h2>MLflow status</h2><div class="value" id="mlflowStatus">...</div><div class="small">SQLite tracking store is read directly.</div></div>
            <div class="card"><h2>Latest run</h2><div class="value" id="latestRun">...</div><div class="small" id="latestRunStatus">Waiting for data...</div></div>

            <div class="card"><h2>Runs</h2><div class="value" id="runs">0</div><div class="small">Logged training or API retrain runs.</div></div>
            <div class="card"><h2>Metrics</h2><div class="value" id="metrics">0</div><div class="small">Accuracy, F1-score, and related values.</div></div>
            <div class="card"><h2>Traces</h2><div class="value" id="traces">0</div><div class="small">HTTP request traces from the API middleware.</div></div>

            <div class="card"><h2>Datasets</h2><div class="value" id="datasets">0</div><div class="small">Training datasets logged with MLflow.</div></div>
            <div class="card"><h2>Inputs</h2><div class="value" id="inputs">0</div><div class="small">Logged input datasets for training.</div></div>
            <div class="card"><h2>Evaluation datasets</h2><div class="value" id="evaluationDatasets">0</div><div class="small">Managed evaluation datasets available to the UI.</div></div>

            <div class="card wide">
                <h2>How to populate with real data</h2>
                <div class="small">
                    1. Start the API and MLflow UI.<br />
                    2. Send real predictions to <code>/predict</code> or run <code>make traffic</code>.<br />
                    3. Run <code>make train</code> to log a new training run.<br />
                    4. Refresh this page to see updated live values.
                </div>
            </div>
        </div>
    </div>

    <script>
        async function refreshStats() {
            try {
                const response = await fetch('/stats');
                const data = await response.json();

                document.getElementById('apiStatus').textContent = data.api_status;
                document.getElementById('mlflowStatus').textContent = data.mlflow_status;
                document.getElementById('latestRun').textContent = data.latest_run_id || 'none';
                document.getElementById('latestRunStatus').textContent = data.latest_run_status ? ('status: ' + data.latest_run_status) : 'Waiting for data...';
                document.getElementById('runs').textContent = data.runs;
                document.getElementById('metrics').textContent = data.metrics;
                document.getElementById('traces').textContent = data.traces;
                document.getElementById('datasets').textContent = data.datasets;
                document.getElementById('inputs').textContent = data.inputs;
                document.getElementById('evaluationDatasets').textContent = data.evaluation_datasets;

                document.getElementById('apiStatus').className = data.api_status === 'up' ? 'value' : 'value bad';
                document.getElementById('mlflowStatus').className = data.mlflow_status === 'up' ? 'value' : 'value bad';
            } catch (error) {
                document.getElementById('latestRunStatus').textContent = 'Error loading dashboard: ' + error;
            }
        }

        refreshStats();
        setInterval(refreshStats, 5000);
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


def _render_modern_test_interface() -> HTMLResponse:
    """Render a modern, professional testing interface for the project model."""

    html = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Obesity Model Studio</title>
    <style>
        :root {
            --bg: #0b1220;
            --bg-2: #101a31;
            --panel: rgba(255,255,255,0.06);
            --panel-strong: rgba(255,255,255,0.10);
            --border: rgba(255,255,255,0.14);
            --text: #eaf1ff;
            --muted: #9db0d0;
            --accent: #7dd3fc;
            --accent-2: #8b5cf6;
            --good: #34d399;
            --warn: #fbbf24;
            --danger: #fb7185;
            --shadow: 0 22px 60px rgba(0,0,0,0.35);
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(125,211,252,0.16), transparent 28%),
                radial-gradient(circle at top right, rgba(139,92,246,0.18), transparent 24%),
                linear-gradient(180deg, #08101d 0%, var(--bg) 100%);
            min-height: 100vh;
        }
        .shell { max-width: 1380px; margin: 0 auto; padding: 28px 20px 42px; }
        .hero {
            display: grid;
            grid-template-columns: 1.35fr 0.65fr;
            gap: 18px;
            margin-bottom: 18px;
            align-items: stretch;
        }
        .panel {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 24px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(14px);
        }
        .hero-main { padding: 28px; }
        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(125,211,252,0.12);
            color: var(--accent);
            font-weight: 700;
            letter-spacing: .02em;
        }
        h1 { margin: 14px 0 10px; font-size: clamp(2rem, 4vw, 3.5rem); line-height: 1.02; }
        .subtitle { color: var(--muted); max-width: 760px; font-size: 1.02rem; line-height: 1.6; }
        .hero-actions { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px; }
        .btn {
            appearance: none; border: 0; cursor: pointer; border-radius: 14px;
            padding: 12px 16px; font-weight: 700; color: #fff;
            background: linear-gradient(135deg, var(--accent-2), var(--accent));
            box-shadow: 0 10px 25px rgba(139,92,246,0.25);
        }
        .btn.secondary { background: rgba(255,255,255,0.08); box-shadow: none; border: 1px solid var(--border); }
        .stats { padding: 22px; display: grid; gap: 14px; }
        .stat {
            padding: 16px;
            border-radius: 18px;
            background: rgba(255,255,255,0.05);
            border: 1px solid var(--border);
        }
        .stat .label { color: var(--muted); font-size: .88rem; }
        .stat .value { font-size: 2rem; font-weight: 800; margin-top: 4px; }
        .grid { display: grid; grid-template-columns: 1.25fr 0.75fr; gap: 18px; margin-top: 18px; }
        .card { padding: 22px; }
        .card h2 { margin: 0 0 12px; font-size: 1.2rem; }
        .card p.desc { margin: 0 0 18px; color: var(--muted); line-height: 1.55; }
        .form-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
        label { display: flex; flex-direction: column; gap: 8px; font-size: .92rem; color: var(--muted); }
        input, select {
            width: 100%; appearance: none; outline: none;
            border: 1px solid var(--border); border-radius: 14px;
            padding: 12px 14px; background: rgba(255,255,255,0.06); color: var(--text);
        }
        input:focus, select:focus { border-color: rgba(125,211,252,0.7); box-shadow: 0 0 0 3px rgba(125,211,252,0.16); }
        .stepper { display: grid; grid-template-columns: 1fr auto; gap: 8px; align-items: center; }
        .stepper .buttons { display: inline-flex; gap: 6px; }
        .mini {
            width: 34px; height: 34px; border-radius: 10px; border: 1px solid var(--border);
            background: rgba(255,255,255,0.08); color: var(--text); cursor: pointer; font-weight: 700;
        }
        .range-wrap { display: grid; gap: 6px; }
        .range-head { display: flex; justify-content: space-between; align-items: center; color: var(--muted); font-size: .84rem; }
        .form-actions { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }
        .result {
            margin-top: 16px; padding: 16px; border-radius: 18px;
            background: linear-gradient(135deg, rgba(52,211,153,0.12), rgba(125,211,252,0.10));
            border: 1px solid rgba(125,211,252,0.24);
        }
        .result .big { font-size: 2rem; font-weight: 800; }
        .result .meta { color: var(--muted); margin-top: 6px; }
        .logs {
            display: grid; gap: 10px; margin-top: 10px; max-height: 650px; overflow: auto;
        }
        .log-item {
            padding: 14px 16px; border-radius: 14px; border: 1px solid var(--border);
            background: rgba(255,255,255,0.05);
        }
        .log-item .k { color: var(--muted); font-size: .84rem; }
        .log-item .v { font-weight: 700; margin-top: 4px; }
        .bad { color: #fda4af; }
        .good { color: #86efac; }
        .muted { color: var(--muted); }
        .wide { margin-top: 18px; }
        .bottom-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-top: 18px; }
        code { background: rgba(255,255,255,0.08); padding: 2px 6px; border-radius: 8px; }
        @media (max-width: 1100px) {
            .hero, .grid, .bottom-grid { grid-template-columns: 1fr; }
            .form-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="shell">
        <div class="hero">
            <section class="panel hero-main">
                <div class="eyebrow">Live Test Interface</div>
                <h1>Obesity Model Studio</h1>
                <p class="subtitle">
                    Interface moderne pour tester ton modèle, vérifier les résultats en direct,
                    et suivre les vraies données du projet dans MLflow.
                </p>
                <div class="hero-actions">
                    <button class="btn" onclick="predict()">Run prediction</button>
                    <button class="btn secondary" onclick="fillSample()">Load sample</button>
                    <button class="btn secondary" onclick="refreshStats()">Refresh live stats</button>
                    <button class="btn secondary" onclick="window.open('/dashboard', '_blank')">Open dashboard</button>
                </div>
            </section>

            <aside class="panel stats">
                <div class="stat"><div class="label">API</div><div class="value" id="apiBadge">...</div></div>
                <div class="stat"><div class="label">MLflow</div><div class="value" id="mlflowBadge">...</div></div>
                <div class="stat"><div class="label">Latest run</div><div class="value" id="latestRun">...</div></div>
                <div class="stat"><div class="label">Traces</div><div class="value" id="traceCount">0</div></div>
            </aside>
        </div>

        <div class="grid">
            <section class="panel card">
                <h2>Test the model</h2>
                <p class="desc">
                    Enter the values and submit them to <code>/predict</code>. The result is shown immediately,
                    and the request is also traced in MLflow. You can also provide weight for your own follow-up.
                </p>
                <div class="form-grid" id="fields"></div>
                <div class="form-actions">
                    <button class="btn" onclick="predict()">Predict now</button>
                    <button class="btn secondary" onclick="clearFields()">Clear</button>
                </div>
                <div class="result" id="resultBox">
                    <div class="big">En attente d'une prédiction</div>
                    <div class="meta">Remplis les champs puis clique sur "Prédire maintenant".</div>
                </div>
            </section>

            <section class="panel card">
                <h2>Live project data</h2>
                <p class="desc">These values are read from the SQLite MLflow store and update automatically.</p>
                <div class="logs">
                    <div class="log-item"><div class="k">Runs</div><div class="v" id="runs">0</div></div>
                    <div class="log-item"><div class="k">Metrics</div><div class="v" id="metrics">0</div></div>
                    <div class="log-item"><div class="k">Datasets</div><div class="v" id="datasets">0</div></div>
                    <div class="log-item"><div class="k">Inputs</div><div class="v" id="inputs">0</div></div>
                    <div class="log-item"><div class="k">Evaluation datasets</div><div class="v" id="evaluationDatasets">0</div></div>
                    <div class="log-item"><div class="k">Latest run status</div><div class="v" id="latestRunStatus">...</div></div>
                </div>
            </section>
        </div>

        <div class="bottom-grid wide">
            <section class="panel card">
                <h2>Comment obtenir un résultat fiable</h2>
                <p class="desc">
                    1. Utilise les valeurs proches du cas patient.<br />
                    2. Clique sur <code>Prédire maintenant</code>.<br />
                    3. Lis le libellé métier, le code de classe et la confiance.
                    <br />4. Les champs sont encodés selon le dataset source (classes 1 a 4).
                </p>
            </section>
            <section class="panel card">
                <h2>Liens rapides</h2>
                <p class="desc">
                    Swagger: <code>/docs</code><br />
                    Dashboard: <code>/dashboard</code><br />
                    Stats JSON: <code>/stats</code>
                </p>
            </section>
        </div>
    </div>

    <script>
        const fieldConfig = [
            { key: 'Age', label: 'Age', type: 'stepper', min: 1, step: 1, defaultValue: 21 },
            { key: 'Sex', label: 'Genre', type: 'select', defaultValue: 2, options: [
                { value: 2, text: 'Femme' },
                { value: 1, text: 'Homme' }
            ] },
            { key: 'Height', label: 'Taille (cm)', type: 'stepper', min: 1, step: 1, defaultValue: 170 },
            { key: 'Weight', label: 'Poids (kg) - optionnel pour suivi IMC', type: 'stepper', min: 1, step: 1, defaultValue: 70 },
            { key: 'Overweight_Obese_Family', label: 'Antecedents familiaux de surpoids', type: 'select', defaultValue: 2, options: [
                { value: 2, text: 'yes' },
                { value: 1, text: 'no' }
            ] },
            { key: 'Consumption_of_Fast_Food', label: "Consommation frequente d'aliments caloriques", type: 'select', defaultValue: 2, options: [
                { value: 2, text: 'yes' },
                { value: 1, text: 'no' }
            ] },
            { key: 'Frequency_of_Consuming_Vegetables', label: 'Frequence de consommation de legumes (1-3)', type: 'range', min: 1, max: 3, step: 1, defaultValue: 3 },
            { key: 'Number_of_Main_Meals_Daily', label: 'Nombre de repas principaux (1-4)', type: 'range', min: 1, max: 4, step: 1, defaultValue: 2 },
            { key: 'Food_Intake_Between_Meals', label: 'Consommation entre les repas', type: 'select', defaultValue: 2, options: [
                { value: 1, text: 'No' },
                { value: 2, text: 'Sometimes' },
                { value: 3, text: 'Frequently' },
                { value: 4, text: 'Always' }
            ] },
            { key: 'Smoking', label: 'Fumeur', type: 'select', defaultValue: 2, options: [
                { value: 2, text: 'no' },
                { value: 1, text: 'yes' }
            ] },
            { key: 'Liquid_Intake_Daily', label: "Consommation d'eau par jour (1-3)", type: 'range', min: 1, max: 3, step: 1, defaultValue: 2 },
            { key: 'Calculation_of_Calorie_Intake', label: 'Suivi calorique', type: 'select', defaultValue: 2, options: [
                { value: 2, text: 'no' },
                { value: 1, text: 'yes' }
            ] },
            { key: 'Physical_Excercise', label: 'Activite physique par semaine (1-4)', type: 'range', min: 1, max: 4, step: 1, defaultValue: 3 },
            { key: 'Schedule_Dedicated_to_Technology', label: "Temps d'utilisation d'appareils electroniques (1-5)", type: 'range', min: 1, max: 5, step: 1, defaultValue: 3 },
            { key: 'Type_of_Transportation_Used', label: 'Moyen de transport', type: 'select', defaultValue: 4, options: [
                { value: 1, text: 'Walking' },
                { value: 2, text: 'Bike' },
                { value: 3, text: 'Motorbike' },
                { value: 4, text: 'Public_Transportation' },
                { value: 5, text: 'Car' }
            ] }
        ];

        const defaultValues = Object.fromEntries(fieldConfig.map((f) => [f.key, f.defaultValue]));
    const modelFeatureKeys = fieldConfig.filter((f) => f.key !== 'Weight').map((f) => f.key);

        const fieldsContainer = document.getElementById('fields');
        const orderedKeys = fieldConfig.map((f) => f.key);

        function createRangeField(config) {
            const label = document.createElement('label');
            const wrap = document.createElement('div');
            wrap.className = 'range-wrap';
            const head = document.createElement('div');
            head.className = 'range-head';
            head.innerHTML = `<span>${config.label}</span><strong id="${config.key}_value">${config.defaultValue}</strong>`;
            const input = document.createElement('input');
            input.type = 'range';
            input.id = config.key;
            input.min = config.min;
            input.max = config.max;
            input.step = config.step || 1;
            input.value = config.defaultValue;
            input.oninput = () => {
                const out = document.getElementById(`${config.key}_value`);
                if (out) out.textContent = input.value;
            };
            wrap.appendChild(head);
            wrap.appendChild(input);
            label.appendChild(wrap);
            return label;
        }

        function createSelectField(config) {
            const label = document.createElement('label');
            label.innerHTML = `<span>${config.label}</span>`;
            const select = document.createElement('select');
            select.id = config.key;
            config.options.forEach((opt) => {
                const option = document.createElement('option');
                option.value = opt.value;
                option.textContent = opt.text;
                if (Number(opt.value) === Number(config.defaultValue)) option.selected = true;
                select.appendChild(option);
            });
            label.appendChild(select);
            return label;
        }

        function createStepperField(config) {
            const label = document.createElement('label');
            label.innerHTML = `<span>${config.label}</span>`;
            const wrapper = document.createElement('div');
            wrapper.className = 'stepper';
            const input = document.createElement('input');
            input.type = 'number';
            input.id = config.key;
            if (config.min !== undefined) input.min = config.min;
            if (config.max !== undefined) input.max = config.max;
            input.step = config.step || 1;
            input.value = config.defaultValue;
            const buttons = document.createElement('div');
            buttons.className = 'buttons';
            const minus = document.createElement('button');
            minus.type = 'button';
            minus.className = 'mini';
            minus.textContent = '-';
            minus.onclick = () => {
                const step = Number(config.step || 1);
                const min = config.min !== undefined ? Number(config.min) : -Infinity;
                const next = Math.max(min, Number(input.value) - step);
                input.value = String(next);
            };
            const plus = document.createElement('button');
            plus.type = 'button';
            plus.className = 'mini';
            plus.textContent = '+';
            plus.onclick = () => {
                const step = Number(config.step || 1);
                const max = config.max !== undefined ? Number(config.max) : Infinity;
                const next = Math.min(max, Number(input.value) + step);
                input.value = String(next);
            };
            buttons.appendChild(minus);
            buttons.appendChild(plus);
            wrapper.appendChild(input);
            wrapper.appendChild(buttons);
            label.appendChild(wrapper);
            return label;
        }

        function renderFields() {
            fieldsContainer.innerHTML = '';
            fieldConfig.forEach((config) => {
                let node;
                if (config.type === 'range') node = createRangeField(config);
                else if (config.type === 'select') node = createSelectField(config);
                else node = createStepperField(config);
                fieldsContainer.appendChild(node);
            });
        }

        function fillSample() {
            Object.entries(defaultValues).forEach(([key, value]) => {
                const input = document.getElementById(key);
                if (input) input.value = value;
            });
            document.getElementById('resultBox').innerHTML = '<div class="big">Sample loaded</div><div class="meta">You can submit the sample or edit any field.</div>';
        }

        function clearFields() {
            orderedKeys.forEach((key) => {
                const input = document.getElementById(key);
                if (input) input.value = defaultValues[key];
                const out = document.getElementById(`${key}_value`);
                if (out) out.textContent = String(defaultValues[key]);
            });
            document.getElementById('resultBox').innerHTML = '<div class="big">Valeurs reinitialisees</div><div class="meta">Les valeurs de demonstration ont ete rechargees.</div>';
        }

        async function refreshStats() {
            try {
                const response = await fetch('/stats');
                const data = await response.json();

                document.getElementById('apiBadge').textContent = data.api_status;
                document.getElementById('mlflowBadge').textContent = data.mlflow_status;
                document.getElementById('latestRun').textContent = data.latest_run_id || 'none';
                document.getElementById('latestRunStatus').textContent = data.latest_run_status ? ('status: ' + data.latest_run_status) : 'Waiting for data...';
                document.getElementById('runs').textContent = data.runs;
                document.getElementById('metrics').textContent = data.metrics;
                document.getElementById('datasets').textContent = data.datasets;
                document.getElementById('inputs').textContent = data.inputs;
                document.getElementById('evaluationDatasets').textContent = data.evaluation_datasets;
                document.getElementById('traceCount').textContent = data.traces;

                document.getElementById('apiBadge').className = data.api_status === 'up' ? 'value good' : 'value bad';
                document.getElementById('mlflowBadge').className = data.mlflow_status === 'up' ? 'value good' : 'value bad';
            } catch (error) {
                document.getElementById('resultBox').innerHTML = `<div class="big bad">Stats unavailable</div><div class="meta">${error}</div>`;
            }
        }

        async function predict() {
            const payload = {};
            for (const key of modelFeatureKeys) {
                const value = document.getElementById(key).value;
                payload[key] = Number(value);
            }

            const weightValue = Number(document.getElementById('Weight').value || 0);
            const heightCm = Number(document.getElementById('Height').value || 0);
            const bmi = heightCm > 0 && weightValue > 0 ? weightValue / ((heightCm / 100) ** 2) : null;
            payload.Weight = weightValue > 0 ? weightValue : null;

            const resultBox = document.getElementById('resultBox');
            resultBox.innerHTML = '<div class="big">Prediction en cours...</div><div class="meta">Envoi des donnees au modele.</div>';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();

                if (!response.ok) {
                    resultBox.innerHTML = `<div class="big bad">Requete invalide</div><div class="meta">${JSON.stringify(data)}</div>`;
                    return;
                }

                const confidence = Number(data.confidence || 0) * 100;
                const serverBmi = data.bmi !== null && data.bmi !== undefined ? Number(data.bmi) : null;
                const sourceText = data.decision_source === 'bmi-adjusted' ? 'IMC + modele' : 'Modele uniquement';
                resultBox.innerHTML = `
                    <div class="big good">${data.predicted_label}</div>
                    <div class="meta">Classe numerique: ${data.predicted_class}</div>
                    <div class="meta">Confiance du modele: ${confidence.toFixed(1)}%</div>
                    <div class="meta">Prediction brute du modele: ${data.model_predicted_label} (classe ${data.model_predicted_class})</div>
                    <div class="meta">Source de decision: ${sourceText}</div>
                    <div class="meta">Poids saisi: ${weightValue > 0 ? `${weightValue.toFixed(1)} kg` : 'non renseigne'}</div>
                    <div class="meta">IMC estime: ${serverBmi !== null ? serverBmi.toFixed(1) : (bmi !== null ? bmi.toFixed(1) : 'non calcule')}</div>
                    <div class="meta">Classe IMC: ${data.bmi_class ?? 'non calculee'}</div>
                    ${data.reason ? `<div class="meta">Note: ${data.reason}</div>` : ''}
                `;
                refreshStats();
            } catch (error) {
                resultBox.innerHTML = `<div class="big bad">Erreur reseau</div><div class="meta">${error}</div>`;
            }
        }

        renderFields();
        refreshStats();
        setInterval(refreshStats, 5000);
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


@app.get("/web", response_class=HTMLResponse)
def web_predict_form() -> HTMLResponse:
    """Compatibility route for the modern test interface."""

    return _render_modern_test_interface()


@app.get("/interface", response_class=HTMLResponse)
def interface() -> HTMLResponse:
    """Modern professional interface to test the model and view live results."""

    return _render_modern_test_interface()


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
    with mlflow.start_span(
        name="predict",
        span_type="PREDICT",
        attributes={"model_path": str(DEFAULT_MODEL_PATH)},
    ):
        probabilities = model.predict_proba(features)[0]
        prediction_index = int(probabilities.argmax())
        model_prediction = int(model.classes_[prediction_index])
        model_predicted_label = CLASS_LABELS.get(model_prediction, f"Classe {model_prediction}")
        confidence = float(probabilities[prediction_index])

    final_prediction = model_prediction
    decision_source = "model"
    reason: str | None = None

    bmi = _compute_bmi(payload.Weight, payload.Height)
    bmi_class = _bmi_to_class(bmi) if bmi is not None else None

    # Keep the most conservative class when BMI clearly indicates higher obesity risk.
    if bmi_class is not None and bmi_class > model_prediction:
        final_prediction = bmi_class
        decision_source = "bmi-adjusted"
        reason = (
            "Classe ajustee par IMC (poids/taille) pour un resultat plus logique "
            "quand le poids indique un risque plus eleve."
        )

    predicted_label = CLASS_LABELS.get(final_prediction, f"Classe {final_prediction}")

    return PredictionResponse(
        predicted_class=final_prediction,
        predicted_label=predicted_label,
        confidence=confidence,
        model_predicted_class=model_prediction,
        model_predicted_label=model_predicted_label,
        bmi=round(bmi, 2) if bmi is not None else None,
        bmi_class=bmi_class,
        decision_source=decision_source,
        reason=reason,
    )


@app.post("/retrain", response_model=RetrainResponse)
def retrain(request: RetrainRequest) -> RetrainResponse:
    global _model

    data_path = Path(request.data_path)
    model_path = Path(request.model_path)

    try:
        with mlflow.start_span(
            name="retrain",
            span_type="TRAIN",
            attributes={"data_path": str(data_path), "model_path": str(model_path)},
        ):
            X_train, X_test, y_train, y_test = prepare_data(
                data_path=data_path,
                test_size=request.test_size,
                random_state=request.random_state,
            )
            model = train_model(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)
            save_model(model, model_path)
            mlflow_info = log_training_run(
                model=model,
                params={
                    "model_type": "RandomForestClassifier",
                    "test_size": request.test_size,
                    "random_state": request.random_state,
                    "n_estimators": model.n_estimators,
                    "class_weight": str(model.class_weight),
                    "data_path": str(data_path),
                    "model_path": str(model_path),
                    "source": "api_retrain",
                },
                metrics=metrics,
                training_data=pd.concat([X_train, y_train.rename("Class")], axis=1),
            )
    except Exception as exc:  # pragma: no cover - defensive API error handling
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _model = model
    return RetrainResponse(
        message=(
            "Model retrained and saved successfully. "
            f"MLflow run_id={mlflow_info['run_id']}"
        ),
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
