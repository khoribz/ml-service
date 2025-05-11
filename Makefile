venv:
	uv venv .venv && . .venv/bin/activate && \
	uv pip install -r pyproject.toml

train:
	python -m src.ml_service.training

serve:
	uvicorn src.ml_service.api.main:app --workers 1 --host 127.0.0.1 --port 8000

dvc-init:
	dvc init -f

dvc:
	$(MAKE) dvc-init && dvc repro

test:
	pytest -q