"""Model Training, ONNX Export, and Low-Latency Serving Session."""

import os
import time
from typing import Dict, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np
import onnxruntime as ort
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from src.features.feature_store import FEATURE_NAMES


class FraudModelTrainer:
    """Trains Deep Neural Classifier and exports to optimized ONNX runtime graph."""

    def __init__(self, model_path: str = "models/fraud_detector.onnx"):
        self.model_path = model_path
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

    def train_and_export(self, df: pd.DataFrame) -> str:
        X = df[FEATURE_NAMES].values.astype(np.float32)
        y = df["is_fraud_ground_truth"].values.astype(np.int32)

        # Fused Pipeline: StandardScaler + Deep Feedforward Neural Net
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    max_iter=100,
                    random_state=42,
                    early_stopping=True,
                ),
            ),
        ])
        pipe.fit(X, y)

        # Convert to fused ONNX graph
        initial_type = [("float_input", FloatTensorType([None, len(FEATURE_NAMES)]))]
        onnx_model = convert_sklearn(pipe, initial_types=initial_type, target_opset=15)

        with open(self.model_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        return self.model_path


class ONNXScoringEngine:
    """Sub-5ms C++ accelerated inference engine using ONNX Runtime with AVX-512."""

    def __init__(self, model_path: str = "models/fraud_detector.onnx"):
        self.model_path = model_path
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def score(self, features: Dict[str, float]) -> Tuple[str, float, float]:
        """Score single feature vector. Returns (verdict, fraud_prob, latency_ms)."""
        t0 = time.perf_counter()

        vec = np.array([[features[k] for k in FEATURE_NAMES]], dtype=np.float32)
        raw_outputs = self.session.run(None, {self.input_name: vec})

        # raw_outputs[1] contains list of probability dictionaries [{0: p0, 1: p1}]
        prob_dict = raw_outputs[1][0]
        fraud_prob = float(prob_dict.get(1, 0.0))

        latency_ms = (time.perf_counter() - t0) * 1000.0

        if fraud_prob >= 0.70:
            verdict = "BLOCK"
        elif fraud_prob >= 0.35:
            verdict = "REVIEW"
        else:
            verdict = "ALLOW"

        return verdict, round(fraud_prob, 4), round(latency_ms, 3)
