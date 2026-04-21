"""Generate real API traffic for MLflow observability charts."""

from __future__ import annotations

import argparse
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def post_json(url: str, payload: dict[str, int], timeout: int = 10) -> tuple[int, float]:
    """POST a JSON payload and return HTTP status and request latency (ms)."""

    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:  # nosec B310
            latency_ms = (time.perf_counter() - started) * 1000.0
            return int(response.status), round(latency_ms, 2)
    except HTTPError as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return int(exc.code), round(latency_ms, 2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate traffic for /predict endpoint.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--ok-count", type=int, default=20)
    parser.add_argument("--error-count", type=int, default=5)
    parser.add_argument("--sleep-ms", type=int, default=50)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    ok_payload = {
        "Sex": 2,
        "Age": 21,
        "Height": 170,
        "Overweight_Obese_Family": 2,
        "Consumption_of_Fast_Food": 2,
        "Frequency_of_Consuming_Vegetables": 3,
        "Number_of_Main_Meals_Daily": 2,
        "Food_Intake_Between_Meals": 2,
        "Smoking": 2,
        "Liquid_Intake_Daily": 2,
        "Calculation_of_Calorie_Intake": 2,
        "Physical_Excercise": 3,
        "Schedule_Dedicated_to_Technology": 3,
        "Type_of_Transportation_Used": 4,
    }
    bad_payload = {"Sex": 99}

    success = 0
    errors = 0
    latencies: list[float] = []

    for _ in range(max(args.ok_count, 0)):
        status, latency = post_json(args.url, ok_payload)
        latencies.append(latency)
        if 200 <= status < 300:
            success += 1
        else:
            errors += 1
        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    for _ in range(max(args.error_count, 0)):
        status, latency = post_json(args.url, bad_payload)
        latencies.append(latency)
        if 400 <= status < 500:
            success += 1
        else:
            errors += 1
        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
    print(
        json.dumps(
            {
                "url": args.url,
                "ok_count_requested": args.ok_count,
                "error_count_requested": args.error_count,
                "requests_sent": len(latencies),
                "checks_passed": success,
                "checks_failed": errors,
                "avg_latency_ms": avg_latency,
            },
            indent=2,
        )
    )

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
