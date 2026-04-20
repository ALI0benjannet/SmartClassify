PYTHON ?= .venv/Scripts/python.exe
PIP ?= $(PYTHON) -m pip

DATA_PATH := archive (1)/Obesity_Dataset.arff
MODEL_PATH := artifacts/obesity_model.joblib

.PHONY: install train evaluate api mlflow-ui docker-build docker-run docker-stop docker-logs docker-compose-up docker-compose-down format lint security test ci clean help

install:
	$(PIP) install -r requirements.txt

train:
	$(PYTHON) main.py --mode train --data-path "$(DATA_PATH)" --model-path "$(MODEL_PATH)"

evaluate:
	$(PYTHON) main.py --mode evaluate --data-path "$(DATA_PATH)" --model-path "$(MODEL_PATH)"

api:
	$(PYTHON) -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

mlflow-ui:
	$(PYTHON) -m mlflow ui --host 127.0.0.1 --port 5000 --backend-store-uri file:./mlruns

docker-build:
	docker build -t obesity-mlops:latest .

docker-run:
	docker run --name obesity-api -p 8000:8000 -v "$(CURDIR)/artifacts:/app/artifacts" -v "$(CURDIR)/mlruns:/app/mlruns" -v "$(CURDIR)/archive (1):/app/archive (1)" obesity-mlops:latest

docker-stop:
	docker stop obesity-api ; docker rm obesity-api

docker-logs:
	docker logs obesity-api

docker-compose-up:
	docker compose up --build

docker-compose-down:
	docker compose down

format:
	$(PYTHON) -m black main.py model_pipeline.py app.py mlflow_utils.py test_model_pipeline.py

lint:
	$(PYTHON) -m ruff check main.py model_pipeline.py app.py mlflow_utils.py test_model_pipeline.py

security:
	$(PYTHON) -m bandit -r main.py model_pipeline.py app.py mlflow_utils.py

test:
	$(PYTHON) -m pytest test_model_pipeline.py

ci: format lint security test

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(path, ignore_errors=True) for path in (Path('__pycache__'), Path('.pytest_cache'), Path('.ruff_cache'), Path('artifacts')) if path.exists()]"

help:
	@echo "Available targets:"
	@echo "  install   - Install project dependencies"
	@echo "  train     - Train the Random Forest model"
	@echo "  evaluate  - Evaluate the saved model"
	@echo "  api       - Start FastAPI server (http://0.0.0.0:8000)"
	@echo "  mlflow-ui - Start MLflow UI (http://127.0.0.1:5000)"
	@echo "  docker-build       - Build Docker image for the project"
	@echo "  docker-run         - Run FastAPI container on port 8000"
	@echo "  docker-stop        - Stop and remove FastAPI container"
	@echo "  docker-logs        - Show logs from FastAPI container"
	@echo "  docker-compose-up  - Start API + MLflow with docker compose"
	@echo "  docker-compose-down- Stop docker compose services"
	@echo "  format    - Format the code with Black"
	@echo "  lint      - Check code quality with Ruff"
	@echo "  security  - Run a basic security scan with Bandit"
	@echo "  test      - Run automated tests with Pytest"
	@echo "  ci        - Run format, lint, security, and tests"
	@echo "  clean     - Remove caches and generated model artifacts"
	@echo "  help      - Show this message"