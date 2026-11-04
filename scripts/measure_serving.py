"""Measure model-call latency, drift checks, and offline writes on synthetic events.

Run from the repository root: python scripts/measure_serving.py
Writes reports/serving_benchmark.json. Absolute timings depend on the machine.
"""

import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.feature_store import FEATURE_NAMES, StreamFeatureStore
from src.models.pipeline import FraudModelTrainer, ONNXScoringEngine
from src.monitoring.drift_detector import DriftMonitoringService
from src.streaming.event_stream import generate_synthetic_event

SEEDS = [42, 7, 123]
N_CALLS = 1000
N_WRITES = 500


def percentiles(values):
    return {f"p{p}": round(float(np.percentile(values, p)), 4) for p in (50, 95, 99)}


def make_rows(store, start, count, drift=False):
    rows = []
    for i in range(start, start + count):
        event = generate_synthetic_event(i, drift_mode=drift)
        rows.append({**store.update_and_get_features(event), "is_fraud_ground_truth": event.is_fraud_ground_truth})
    return pd.DataFrame(rows)


def run_seed(seed, tmp_dir):
    random.seed(seed)
    store = StreamFeatureStore(":memory:")
    train_df = pd.concat([make_rows(store, 0, 4000), make_rows(store, 4000, 400, drift=True)], ignore_index=True)

    model_path = os.path.join(tmp_dir, f"fraud_{seed}.onnx")
    FraudModelTrainer(model_path).train_and_export(train_df)
    engine = ONNXScoringEngine(model_path)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=100, random_state=42, early_stopping=True)),
    ])
    pipe.fit(train_df[FEATURE_NAMES].values.astype(np.float32), train_df["is_fraud_ground_truth"].values)

    sample = make_rows(store, 5000, N_CALLS)
    records = sample[FEATURE_NAMES].to_dict("records")
    onnx_ms, sklearn_ms = [], []
    for row in records:
        t0 = time.perf_counter()
        engine.score(row)
        onnx_ms.append((time.perf_counter() - t0) * 1000)
        vector = np.array([[row[k] for k in FEATURE_NAMES]], dtype=np.float32)
        t0 = time.perf_counter()
        pipe.predict_proba(vector)
        sklearn_ms.append((time.perf_counter() - t0) * 1000)

    pipeline_ms = []
    for i in range(N_CALLS):
        event = generate_synthetic_event(10000 + i)
        t0 = time.perf_counter()
        engine.score(store.update_and_get_features(event))
        pipeline_ms.append((time.perf_counter() - t0) * 1000)

    def probabilities(frame):
        return np.array([engine.score(r)[1] for r in frame[FEATURE_NAMES].to_dict("records")])

    baseline = make_rows(store, 20000, 2000)
    stable = make_rows(store, 22000, 1000)
    drifted = make_rows(store, 23000, 1000, drift=True)
    monitor = DriftMonitoringService(baseline, probabilities(baseline))
    stable_report = monitor.evaluate_drift(stable, probabilities(stable))
    drift_report = monitor.evaluate_drift(drifted, probabilities(drifted))

    write_store = StreamFeatureStore(":memory:")
    events = [generate_synthetic_event(30000 + i) for i in range(N_WRITES * 2)]
    pairs = [(e, write_store.update_and_get_features(e)) for e in events]
    t0 = time.perf_counter()
    for event, features in pairs[:N_WRITES]:
        write_store.log_to_offline_store(event, features)
    row_seconds = time.perf_counter() - t0
    batch = [
        {
            "transaction_id": e.transaction_id,
            "user_id": e.user_id,
            "merchant_id": e.merchant_id,
            "timestamp": e.timestamp,
            "is_fraud_ground_truth": e.is_fraud_ground_truth,
            **f,
        }
        for e, f in pairs[N_WRITES:]
    ]
    t0 = time.perf_counter()
    write_store.log_batch_to_offline_store(batch)
    batch_seconds = time.perf_counter() - t0
    write_store.close()
    store.close()

    return {
        "seed": seed,
        "onnx_model_call_ms": percentiles(onnx_ms),
        "sklearn_model_call_ms": percentiles(sklearn_ms),
        "feature_update_plus_onnx_ms": percentiles(pipeline_ms),
        "stable_window": {k: stable_report[k] for k in ("status", "prediction_psi", "drifted_features_count")},
        "botnet_window": {k: drift_report[k] for k in ("status", "prediction_psi", "drifted_features_count")},
        "offline_write_seconds": {"row_by_row": round(row_seconds, 4), "batch": round(batch_seconds, 4)},
        "batch_speedup": round(row_seconds / batch_seconds, 1),
    }


def main():
    with tempfile.TemporaryDirectory() as tmp_dir:
        runs = [run_seed(seed, tmp_dir) for seed in SEEDS]
    report = {
        "scope": "Synthetic events, in-process calls. No HTTP, network, or broker time is included.",
        "machine": platform.processor() or platform.machine(),
        "calls_per_seed": N_CALLS,
        "offline_writes_per_method": N_WRITES,
        "median_p50_speedup_onnx_vs_sklearn": round(
            statistics.median(r["sklearn_model_call_ms"]["p50"] / r["onnx_model_call_ms"]["p50"] for r in runs), 1
        ),
        "runs": runs,
    }
    out = Path(__file__).resolve().parents[1] / "reports" / "serving_benchmark.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
