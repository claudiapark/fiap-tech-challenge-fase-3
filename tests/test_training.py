"""Training, export and inference tests."""

import json

import pytest

from medical_triage.inference import TriagePredictor


def test_training_exports_models_and_metrics(trained_artifacts) -> None:
    model_dir, report_dir, metrics = trained_artifacts
    assert (model_dir / "vectorizer.joblib").exists()
    assert (model_dir / "classifier.joblib").exists()
    assert (model_dir / "classifier.onnx").exists()
    assert metrics["dataset"]["rows"] == 2100
    assert metrics["model"]["macro_f1"] >= 0.80
    assert metrics["latency"]["baseline_sklearn"]["mean_ms"] > 0
    assert metrics["latency"]["optimized_onnx"]["mean_ms"] > 0
    persisted = json.loads((report_dir / "metrics.json").read_text())
    assert persisted["model"]["classes"] == ["attention", "normal", "urgent"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Exame de rotina sem alterações e paciente estável.", "normal"),
        ("Febre persistente e dor moderada, avaliar ainda hoje.", "attention"),
        ("Dor torácica intensa e falta de ar súbita, emergência.", "urgent"),
    ],
)
def test_onnx_predictions_match_expected_class(trained_artifacts, text, expected) -> None:
    model_dir, _, _ = trained_artifacts
    prediction = TriagePredictor(model_dir, "onnx").predict(text)
    assert prediction.urgency == expected
    assert 0 <= prediction.confidence <= 1
    assert prediction.engine == "onnx"


def test_sklearn_engine_is_available(trained_artifacts) -> None:
    model_dir, _, _ = trained_artifacts
    prediction = TriagePredictor(model_dir, "sklearn").predict(
        "Dificuldade respiratória grave e saturação crítica."
    )
    assert prediction.urgency == "urgent"
    assert prediction.engine == "sklearn"


def test_invalid_engine_is_rejected(trained_artifacts) -> None:
    model_dir, _, _ = trained_artifacts
    with pytest.raises(ValueError, match="Unsupported"):
        TriagePredictor(model_dir, "invalid")
