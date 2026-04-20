PYTHON ?= .venv/Scripts/python.exe
PIP ?= $(PYTHON) -m pip

DATA_PATH := archive (1)/Obesity_Dataset.arff
MODEL_PATH := artifacts/obesity_model.joblib

.PHONY: install train evaluate api format lint security test ci clean help

install:
	$(PIP) install -r requirements.txt

train:
	$(PYTHON) main.py --mode train --data-path "$(DATA_PATH)" --model-path "$(MODEL_PATH)"

evaluate:
	$(PYTHON) main.py --mode evaluate --data-path "$(DATA_PATH)" --model-path "$(MODEL_PATH)"

api:
	$(PYTHON) -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

format:
	$(PYTHON) -m black main.py model_pipeline.py app.py test_model_pipeline.py

lint:
	$(PYTHON) -m ruff check main.py model_pipeline.py app.py test_model_pipeline.py

security:
	$(PYTHON) -m bandit -r main.py model_pipeline.py app.py

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
	@echo "  format    - Format the code with Black"
	@echo "  lint      - Check code quality with Ruff"
	@echo "  security  - Run a basic security scan with Bandit"
	@echo "  test      - Run automated tests with Pytest"
	@echo "  ci        - Run format, lint, security, and tests"
	@echo "  clean     - Remove caches and generated model artifacts"
	@echo "  help      - Show this message"