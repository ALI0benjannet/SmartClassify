"""Session 7 monitoring report for API and MLflow."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def check_http(url: str, timeout: int = 5) -> dict[str, object]:
    """Return basic health details for an HTTP endpoint."""

    started = time.time()
    try:
        with urlopen(url, timeout=timeout) as response:  # nosec B310
            latency_ms = round((time.time() - started) * 1000.0, 2)
            return {
                "url": url,
                "ok": 200 <= response.status < 400,
                "status": int(response.status),
                "latency_ms": latency_ms,
            }
    except URLError as exc:
        latency_ms = round((time.time() - started) * 1000.0, 2)
        return {
            "url": url,
            "ok": False,
            "status": None,
            "latency_ms": latency_ms,
            "error": str(exc),
        }


def read_mlflow_counts(db_path: Path) -> dict[str, object]:
    """Collect key counters from MLflow SQLite tables."""

    if not db_path.exists():
        return {"db_path": str(db_path), "exists": False}

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    counters: dict[str, int] = {}
    for table in [
        "runs",
        "metrics",
        "datasets",
        "inputs",
        "evaluation_datasets",
        "trace_info",
    ]:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        counters[table] = int(cur.fetchone()[0])

    cur.execute(
        "SELECT run_uuid, status, start_time FROM runs ORDER BY start_time DESC LIMIT 1"
    )
    latest_run = cur.fetchone()
    conn.close()

    latest = None
    if latest_run is not None:
        latest = {
            "run_uuid": latest_run[0],
            "status": latest_run[1],
            "start_time": int(latest_run[2]),
        }

    return {
        "db_path": str(db_path),
        "exists": True,
        "counters": counters,
        "latest_run": latest,
    }


def build_alerts(
    report: dict[str, object],
    min_traces: int,
    min_evaluation_datasets: int,
) -> list[dict[str, object]]:
    """Build alert entries from health and MLflow counters."""

    alerts: list[dict[str, object]] = []

    api = report.get("api", {})
    if not bool(api.get("ok")):
        alerts.append(
            {
                "severity": "critical",
                "message": "API endpoint is unavailable.",
                "details": {"url": api.get("url"), "status": api.get("status")},
            }
        )

    mlflow = report.get("mlflow", {})
    if not bool(mlflow.get("ok")):
        alerts.append(
            {
                "severity": "critical",
                "message": "MLflow UI endpoint is unavailable.",
                "details": {
                    "url": mlflow.get("url"),
                    "status": mlflow.get("status"),
                },
            }
        )

    store = report.get("mlflow_store", {})
    if not bool(store.get("exists")):
        alerts.append(
            {
                "severity": "critical",
                "message": "MLflow SQLite database is missing.",
                "details": {"db_path": store.get("db_path")},
            }
        )
        return alerts

    counters = store.get("counters", {})
    trace_count = int(counters.get("trace_info", 0))
    eval_dataset_count = int(counters.get("evaluation_datasets", 0))

    if trace_count < min_traces:
        alerts.append(
            {
                "severity": "warning",
                "message": "Trace volume is below threshold.",
                "details": {
                    "actual": trace_count,
                    "minimum": min_traces,
                    "hint": "Trigger /predict or /retrain to generate traces.",
                },
            }
        )

    if eval_dataset_count < min_evaluation_datasets:
        alerts.append(
            {
                "severity": "warning",
                "message": "Evaluation dataset count is below threshold.",
                "details": {
                    "actual": eval_dataset_count,
                    "minimum": min_evaluation_datasets,
                    "hint": "Run training and ensure MLflow evaluation dataset setup is active.",
                },
            }
        )

    return alerts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a supervision report.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/docs")
    parser.add_argument("--mlflow-url", default="http://127.0.0.1:5000")
    parser.add_argument("--db-path", default="mlflow.db")
    parser.add_argument("--min-traces", type=int, default=1)
    parser.add_argument("--min-evaluation-datasets", type=int, default=1)
    parser.add_argument("--fail-on-alert", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = {
        "api": check_http(args.api_url),
        "mlflow": check_http(args.mlflow_url),
        "mlflow_store": read_mlflow_counts(Path(args.db_path)),
    }
    report["alerts"] = build_alerts(
        report=report,
        min_traces=args.min_traces,
        min_evaluation_datasets=args.min_evaluation_datasets,
    )

    print(json.dumps(report, indent=2))

    api_ok = bool(report["api"].get("ok"))
    mlflow_ok = bool(report["mlflow"].get("ok"))
    db_ok = bool(report["mlflow_store"].get("exists"))

    alerts_ok = not args.fail_on_alert or len(report["alerts"]) == 0

    return 0 if api_ok and mlflow_ok and db_ok and alerts_ok else 1


if __name__ == "__main__":
    sys.exit(main())
