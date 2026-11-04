"""FastAPI contracts and Prometheus instrumentation tests."""

from fastapi.testclient import TestClient

from medical_triage.api.main import app
from medical_triage.config import get_settings


def build_client(monkeypatch, model_dir) -> TestClient:
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.setenv("INFERENCE_ENGINE", "onnx")
    get_settings.cache_clear()
    return TestClient(app)


def test_health_and_prediction(trained_artifacts, monkeypatch) -> None:
    model_dir, _, _ = trained_artifacts
    with build_client(monkeypatch, model_dir) as client:
        health = client.get("/health")
        response = client.post(
            "/predict",
            json={"report_text": "Dor torácica intensa e falta de ar súbita, emergência."},
        )
    assert health.json() == {"status": "ok", "model_loaded": True, "engine": "onnx"}
    assert response.status_code == 200
    assert response.json()["urgency"] == "urgent"
    assert "não substitui" in response.json()["disclaimer"]


def test_validation_and_metrics_endpoint(trained_artifacts, monkeypatch) -> None:
    model_dir, _, _ = trained_artifacts
    with build_client(monkeypatch, model_dir) as client:
        invalid = client.post("/predict", json={"report_text": "curto"})
        metrics = client.get("/metrics")
    assert invalid.status_code == 422
    assert metrics.status_code == 200
    assert "triage_request_latency_seconds" in metrics.text
    assert "triage_requests_total" in metrics.text


def test_root_does_not_expose_sensitive_data(trained_artifacts, monkeypatch) -> None:
    model_dir, _, _ = trained_artifacts
    with build_client(monkeypatch, model_dir) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"
    assert (
        "clinical" in response.json()["disclaimer"]
        or "profissional" in response.json()["disclaimer"]
    )
