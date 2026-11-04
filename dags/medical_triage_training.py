"""Airflow DAG for deterministic ingestion, training and model validation."""

import json
import os
from datetime import datetime
from pathlib import Path

try:
    from airflow.sdk import dag, task
except ImportError:  # Airflow 2 compatibility
    from airflow.decorators import dag, task

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/opt/airflow/project"))


@dag(
    dag_id="medical_triage_retraining",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["fiap", "nlp", "retraining"],
)
def medical_triage_retraining():
    """Generate data, retrain artifacts and enforce a simple quality gate."""

    @task
    def ingest() -> str:
        from medical_triage.data import generate_dataset

        output = PROJECT_ROOT / "data/raw/medical_reports.csv"
        return str(generate_dataset(output))

    @task
    def train(dataset_path: str) -> str:
        from medical_triage.training import train_and_export

        train_and_export(
            Path(dataset_path),
            PROJECT_ROOT / "models",
            PROJECT_ROOT / "reports",
        )
        return str(PROJECT_ROOT / "reports/metrics.json")

    @task
    def validate(metrics_path: str) -> None:
        metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        if metrics["model"]["macro_f1"] < 0.80:
            raise ValueError("Model quality gate failed: macro F1 below 0.80")

    validate(train(ingest()))


medical_triage_retraining()
