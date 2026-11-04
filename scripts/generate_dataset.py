"""Generate the deterministic academic triage dataset."""

from pathlib import Path

from medical_triage.data import generate_dataset

if __name__ == "__main__":
    path = generate_dataset(Path("data/raw/medical_reports.csv"))
    print(f"Dataset generated at {path}")
