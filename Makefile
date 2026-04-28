PYTHON ?= .venv/Scripts/python.exe
PIP ?= $(PYTHON) -m pip

DATA_PATH := archive (1)/Obesity_Dataset.arff
MODEL_PATH := artifacts/obesity_model.joblib
IMAGE_NAME ?= obesity-mlops
IMAGE_TAG ?= latest
DOCKERHUB_USER ?= your-dockerhub-user
LOCAL_IMAGE := $(IMAGE_NAME):$(IMAGE_TAG)
REMOTE_IMAGE := $(DOCKERHUB_USER)/$(IMAGE_NAME):$(IMAGE_TAG)

.PHONY: install train evaluate api mlflow-ui monitor monitor-strict traffic traffic-heavy docker-login docker-build docker-tag docker-push docker-publish docker-run docker-stop docker-logs docker-compose-up docker-compose-down format lint security test ci clean help

install:
	$(PIP) install -r requirements.txt

train:
	$(PYTHON) main.py --mode train --data-path "$(DATA_PATH)" --model-path "$(MODEL_PATH)"

evaluate:
	$(PYTHON) main.py --mode evaluate --data-path "$(DATA_PATH)" --model-path "$(MODEL_PATH)"

api:
	$(PYTHON) -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

mlflow-ui:
	$(PYTHON) -m mlflow ui --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///./mlflow.db

monitor:
	$(PYTHON) monitoring_report.py

monitor-strict:
	$(PYTHON) monitoring_report.py --fail-on-alert --min-traces 1 --min-evaluation-datasets 1

traffic:
	$(PYTHON) generate_traffic.py --ok-count 20 --error-count 5 --sleep-ms 50

traffic-heavy:
	$(PYTHON) generate_traffic.py --ok-count 100 --error-count 20 --sleep-ms 20

docker-login:
	docker login --username $(DOCKERHUB_USER)

docker-build:
	docker build -t $(LOCAL_IMAGE) .

docker-tag:
	docker tag $(LOCAL_IMAGE) $(REMOTE_IMAGE)

docker-push: docker-tag
	docker push $(REMOTE_IMAGE)

docker-publish: docker-build docker-tag docker-push

docker-run:
	docker run --name obesity-api -p 8000:8000 -v "$(CURDIR)/artifacts:/app/artifacts" -v "$(CURDIR)/mlflow.db:/app/mlflow.db" -v "$(CURDIR)/mlartifacts:/app/mlartifacts" -v "$(CURDIR)/archive (1):/app/archive (1)" $(LOCAL_IMAGE)

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
	@echo "  monitor   - Show supervision report (API/MLflow/DB)"
	@echo "  monitor-strict - Fail when supervision alert thresholds are not met"
	@echo "  traffic   - Send real requests to /predict for MLflow charts"
	@echo "  traffic-heavy - Send larger request volume for denser charts"
	@echo "  docker-login       - Log in to Docker Hub"
	@echo "  docker-build       - Build local Docker image"
	@echo "  docker-tag         - Tag image for Docker Hub ($(REMOTE_IMAGE))"
	@echo "  docker-push        - Push tagged image to Docker Hub"
	@echo "  docker-publish     - Build, tag, and push in one command"
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