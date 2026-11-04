"""Train, export and benchmark the medical text classifier."""

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from medical_triage.data import generate_dataset, load_dataset


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values), percentile))


def _measure_latency(callable_: Any, repeats: int = 150) -> dict[str, float]:
    for _ in range(10):
        callable_()
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        callable_()
        samples.append((perf_counter() - start) * 1000)
    return {
        "mean_ms": float(np.mean(samples)),
        "p50_ms": _percentile(samples, 50),
        "p95_ms": _percentile(samples, 95),
    }


def train_and_export(
    dataset_path: Path,
    model_dir: Path,
    report_dir: Path,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train the baseline, export ONNX and compare classifier latency."""
    if not dataset_path.exists():
        generate_dataset(dataset_path, seed=random_state)
    texts, labels = load_dataset(dataset_path)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=random_state,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
    train_features = vectorizer.fit_transform(train_texts)
    test_features = vectorizer.transform(test_texts)
    classifier = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=random_state,
    )
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, model_dir / "vectorizer.joblib")
    joblib.dump(classifier, model_dir / "classifier.joblib")

    onnx_model = convert_sklearn(
        classifier,
        initial_types=[("features", FloatTensorType([None, train_features.shape[1]]))],
        options={id(classifier): {"zipmap": False}},
        target_opset=17,
    )
    onnx_path = model_dir / "classifier.onnx"
    onnx_path.write_bytes(onnx_model.SerializeToString())

    benchmark_batch = test_features[:128].astype(np.float32)
    dense_batch = benchmark_batch.toarray()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    baseline_latency = _measure_latency(lambda: classifier.predict_proba(benchmark_batch))
    optimized_latency = _measure_latency(lambda: session.run(None, {input_name: dense_batch}))
    speedup = baseline_latency["mean_ms"] / optimized_latency["mean_ms"]

    metrics: dict[str, Any] = {
        "dataset": {
            "rows": len(texts),
            "train_rows": len(train_texts),
            "test_rows": len(test_texts),
            "source": "deterministic synthetic academic dataset",
        },
        "model": {
            "type": "TF-IDF + LogisticRegression",
            "classes": classifier.classes_.tolist(),
            "accuracy": accuracy_score(test_labels, predictions),
            "macro_f1": f1_score(test_labels, predictions, average="macro"),
            "classification_report": classification_report(
                test_labels,
                predictions,
                output_dict=True,
                zero_division=0,
            ),
        },
        "latency": {
            "batch_size": len(dense_batch),
            "baseline_sklearn": baseline_latency,
            "optimized_onnx": optimized_latency,
            "speedup": speedup,
        },
    }
    (report_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (model_dir / "metadata.json").write_text(
        json.dumps({"classes": classifier.classes_.tolist()}, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics
