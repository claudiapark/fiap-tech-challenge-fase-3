"""Display the persisted baseline versus ONNX latency comparison."""

import json
from pathlib import Path

if __name__ == "__main__":
    report = json.loads(Path("reports/metrics.json").read_text(encoding="utf-8"))
    latency = report["latency"]
    print("Baseline Scikit-Learn:", latency["baseline_sklearn"])
    print("Optimized ONNX:", latency["optimized_onnx"])
    print(f"Speedup: {latency['speedup']:.2f}x")
