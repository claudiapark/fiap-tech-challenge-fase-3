.PHONY: setup data train test lint run stack down benchmark

setup:
	uv sync --locked

data:
	uv run python scripts/generate_dataset.py

train: data
	uv run python scripts/train.py

test:
	uv run pytest

lint:
	uv run ruff format --check .
	uv run ruff check .

run: train
	uv run uvicorn medical_triage.api.main:app --reload

stack:
	docker compose up --build -d

down:
	docker compose down

benchmark: train
	uv run python scripts/benchmark.py
