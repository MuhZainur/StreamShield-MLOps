"""End-to-End Test Suite for StreamShield Real-Time MLOps Pipeline."""

import os
import time
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.streaming.event_stream import generate_synthetic_event
from src.features.feature_store import StreamFeatureStore, FEATURE_NAMES
from src.models.pipeline import FraudModelTrainer, ONNXScoringEngine
from src.monitoring.drift_detector import DriftMonitoringService, calculate_psi
from src.serving.api import app, initialize_mlops_pipeline


@pytest.fixture(scope="session")
def temp_db_path(tmp_path_factory):
    fn = tmp_path_factory.mktemp("data") / "test_features.duckdb"
    return str(fn)


def test_synthetic_event_generation():
    ev_normal = generate_synthetic_event(1, drift_mode=False)
    assert ev_normal.transaction_id == "tx_00000001"
    assert ev_normal.amount > 0.0

    ev_drift = generate_synthetic_event(2, drift_mode=True)
    assert ev_drift.amount <= 5.0  # Micro-amounts in botnet card-testing mode


def test_feature_store_dual_storage(temp_db_path):
    store = StreamFeatureStore(offline_db_path=temp_db_path)
    ev = generate_synthetic_event(10, drift_mode=False)

    # Online feature retrieval (< 5ms)
    t0 = time.perf_counter()
    feats = store.update_and_get_features(ev)
    lat_ms = (time.perf_counter() - t0) * 1000.0
    assert lat_ms < 5.0

    for name in FEATURE_NAMES:
        assert name in feats

    # Offline logging & historical join check
    store.log_to_offline_store(ev, feats)
    df_hist = store.get_historical_training_df()
    assert len(df_hist) >= 1
    assert "user_tx_count_10m" in df_hist.columns


def test_onnx_model_training_and_low_latency_scoring(tmp_path):
    # Create synthetic dataset
    rows = []
    store = StreamFeatureStore(offline_db_path=str(tmp_path / "temp.duckdb"))
    for i in range(200):
        ev = generate_synthetic_event(i, drift_mode=(i % 5 == 0))
        f = store.update_and_get_features(ev)
        r = dict(f)
        r["is_fraud_ground_truth"] = ev.is_fraud_ground_truth
        rows.append(r)

    df_train = pd.DataFrame(rows)
    onnx_path = str(tmp_path / "test_model.onnx")

    trainer = FraudModelTrainer(model_path=onnx_path)
    trainer.train_and_export(df_train)
    assert os.path.exists(onnx_path)

    # Test ONNX Scoring Engine Latency SLA (< 12ms)
    engine = ONNXScoringEngine(model_path=onnx_path)
    test_features = {k: float(df_train[k].iloc[0]) for k in FEATURE_NAMES}

    verdict, prob, lat = engine.score(test_features)
    assert verdict in ["ALLOW", "REVIEW", "BLOCK"]
    assert 0.0 <= prob <= 1.0
    assert lat < 12.0  # P99 SLA constraint


def test_drift_detector_sensitivity():
    # Stable baseline vs drifted actual
    base_amounts = np.random.normal(50.0, 10.0, size=500)
    drifted_amounts = np.random.uniform(0.5, 4.0, size=500)  # Botnet attack shift

    base_probs = np.random.beta(0.5, 20.0, size=500)  # Low fraud prob
    drifted_probs = np.random.beta(5.0, 2.0, size=500)  # High fraud prob spike

    base_df = pd.DataFrame({
        "amount": base_amounts,
        "user_tx_count_10m": np.random.poisson(2, size=500),
        "user_amount_sum_1h": base_amounts * 2,
        "user_avg_amount_30d": base_amounts,
        "amount_to_avg_ratio": np.ones(500),
        "merchant_risk_score": np.full(500, 0.02),
    })

    drift_df = pd.DataFrame({
        "amount": drifted_amounts,
        "user_tx_count_10m": np.random.poisson(8, size=500),
        "user_amount_sum_1h": drifted_amounts * 5,
        "user_avg_amount_30d": base_amounts,
        "amount_to_avg_ratio": drifted_amounts / base_amounts,
        "merchant_risk_score": np.full(500, 0.75),
    })

    monitor = DriftMonitoringService(baseline_df=base_df, baseline_probs=base_probs)
    report = monitor.evaluate_drift(drift_df, drifted_probs)

    assert report["concept_drift_detected"] is True
    assert report["prediction_psi"] > 0.20
    assert report["status"] == "ALERT_DRIFT_DETECTED"


def test_fastapi_scoring_endpoint():
    initialize_mlops_pipeline(n_warmup_events=300)
    client = TestClient(app)

    # Health check
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

    # Score transaction endpoint
    payload = {
        "user_id": "usr_test_99",
        "merchant_id": "merch_001",
        "amount": 45.0,
        "device_id": "dev_test",
        "ip_address": "127.0.0.1",
    }
    t0 = time.perf_counter()
    res = client.post("/api/v1/score", json=payload)
    total_time_ms = (time.perf_counter() - t0) * 1000.0

    assert res.status_code == 200
    body = res.json()
    assert body["verdict"] in ["ALLOW", "REVIEW", "BLOCK"]
    assert "fraud_probability" in body
    assert body["latency_ms"] < 15.0
