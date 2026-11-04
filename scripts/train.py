"""Train and export baseline and optimized classifiers."""

import json
from pathlib import Path

from medical_triage.training import train_and_export

if __name__ == "__main__":
    result = train_and_export(
        Path("data/raw/medical_reports.csv"),
        Path("models"),
        Path("reports"),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
