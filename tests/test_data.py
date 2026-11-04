"""Dataset generation and validation tests."""

from pathlib import Path

import pytest

from medical_triage.data import LABELS, generate_dataset, load_dataset


def test_generate_dataset_is_balanced_and_large_enough(tmp_path: Path) -> None:
    path = generate_dataset(tmp_path / "reports.csv", rows=2100)
    texts, labels = load_dataset(path)
    assert len(texts) == 2100
    assert set(labels) == set(LABELS)
    assert all("@" not in text for text in texts)


def test_generate_dataset_rejects_small_sample(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2,000"):
        generate_dataset(tmp_path / "small.csv", rows=100)


def test_load_dataset_rejects_missing_classes(tmp_path: Path) -> None:
    path = tmp_path / "invalid.csv"
    path.write_text("report_text,urgency\ntext normal,normal\n" * 2000, encoding="utf-8")
    with pytest.raises(ValueError, match="urgency classes"):
        load_dataset(path)
