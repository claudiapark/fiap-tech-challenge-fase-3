"""Shared fixtures for trained academic model artifacts."""

import pytest

from medical_triage.data import generate_dataset
from medical_triage.training import train_and_export


@pytest.fixture(scope="session")
def trained_artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("trained")
    dataset = generate_dataset(root / "medical_reports.csv", rows=2100)
    model_dir = root / "models"
    report_dir = root / "reports"
    metrics = train_and_export(dataset, model_dir, report_dir)
    return model_dir, report_dir, metrics
