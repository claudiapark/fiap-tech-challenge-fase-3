"""Inference service supporting baseline Scikit-Learn and optimized ONNX."""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class Prediction:
    """One triage prediction."""

    urgency: str
    confidence: float
    engine: str


class TriagePredictor:
    """Load model artifacts once and execute text classification."""

    def __init__(self, model_dir: Path, engine: str = "onnx") -> None:
        self.engine = engine
        self.vectorizer = joblib.load(model_dir / "vectorizer.joblib")
        self.metadata = json.loads((model_dir / "metadata.json").read_text())
        if engine == "onnx":
            self.session = ort.InferenceSession(
                str(model_dir / "classifier.onnx"),
                providers=["CPUExecutionProvider"],
            )
            self.classifier = None
        elif engine == "sklearn":
            self.classifier = joblib.load(model_dir / "classifier.joblib")
            self.session = None
        else:
            raise ValueError(f"Unsupported inference engine: {engine}")

    def predict(self, text: str) -> Prediction:
        """Classify one medical report text."""
        features = self.vectorizer.transform([text])
        if self.engine == "onnx" and self.session is not None:
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: features.astype(np.float32).toarray()})
            label = str(outputs[0][0])
            confidence = float(np.max(outputs[1][0]))
        else:
            probabilities = self.classifier.predict_proba(features)[0]
            index = int(np.argmax(probabilities))
            label = str(self.classifier.classes_[index])
            confidence = float(probabilities[index])
        return Prediction(label, round(confidence, 4), self.engine)
