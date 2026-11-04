"""Static validation for orchestration and observability assets."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_airflow_dag_has_ingest_train_and_validate_tasks() -> None:
    source = (ROOT / "dags/medical_triage_training.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert {"ingest", "train", "validate"}.issubset(functions)
    assert "medical_triage_retraining" in source


def test_grafana_dashboard_has_at_least_three_panels() -> None:
    dashboard = json.loads((ROOT / "monitoring/grafana/dashboards/medical-triage.json").read_text())
    titles = {panel["title"] for panel in dashboard["panels"]}
    assert len(titles) >= 3
    assert {"Total de requisições", "Latência p95", "Taxa de erros"}.issubset(titles)


def test_prometheus_scrapes_api_metrics() -> None:
    config = (ROOT / "monitoring/prometheus/prometheus.yml").read_text()
    assert "api:8000" in config
    assert "metrics_path: /metrics" in config
