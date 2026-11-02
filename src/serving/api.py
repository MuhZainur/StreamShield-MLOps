"""Low-Latency FastAPI Real-Time Scoring Microservice."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import time
from typing import Dict, List, Literal
from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.features.feature_store import StreamFeatureStore, FEATURE_NAMES
from src.models.pipeline import FraudModelTrainer, ONNXScoringEngine
from src.monitoring.drift_detector import DriftMonitoringService
from src.streaming.event_stream import TransactionEvent, generate_synthetic_event


feature_store: StreamFeatureStore | None = None
scoring_engine: ONNXScoringEngine | None = None
drift_service: DriftMonitoringService | None = None

# In-memory inference ring buffer for live drift scoring
recent_features_log: List[Dict[str, float]] = []
recent_probs_log: List[float] = []


def initialize_mlops_pipeline(n_warmup_events: int = 500):
    """Warm up Feature Store, train baseline model, export ONNX, and initialize drift baseline."""
    global feature_store, scoring_engine, drift_service

    if scoring_engine is not None:
        return

    feature_store = StreamFeatureStore("data/offline/features.duckdb")

    # Ingest baseline events
    baseline_records = []
    for i in range(n_warmup_events):
        ev = generate_synthetic_event(i, drift_mode=False)
        feats = feature_store.update_and_get_features(ev)
        feature_store.log_to_offline_store(ev, feats)
        rec = dict(feats)
        rec["is_fraud_ground_truth"] = ev.is_fraud_ground_truth
        baseline_records.append(rec)

    baseline_df = pd.DataFrame(baseline_records)

    # Train and export ONNX
    trainer = FraudModelTrainer("models/fraud_detector.onnx")
    trainer.train_and_export(baseline_df)
    scoring_engine = ONNXScoringEngine("models/fraud_detector.onnx")

    # Baseline probabilities for drift comparison
    baseline_probs = []
    for rec in baseline_records:
        _, p, _ = scoring_engine.score(rec)
        baseline_probs.append(p)

    drift_service = DriftMonitoringService(
        baseline_df=baseline_df,
        baseline_probs=np.array(baseline_probs),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_mlops_pipeline()
    yield


app = FastAPI(
    title="StreamShield Real-Time MLOps Engine",
    version="0.1.0",
    description="Sub-15ms Event-Driven Fraud Scoring and Drift Observability Service",
    lifespan=lifespan,
)


class ScoreRequest(BaseModel):
    user_id: str = Field(..., json_schema_extra={"example": "usr_00042"})
    merchant_id: str = Field(..., json_schema_extra={"example": "merch_005"})
    amount: float = Field(..., gt=0.0, json_schema_extra={"example": 79.50})
    device_id: str = Field(default="dev_web_01")
    ip_address: str = Field(default="192.168.1.1")


class ScoreResponse(BaseModel):
    transaction_id: str
    user_id: str
    verdict: Literal["ALLOW", "REVIEW", "BLOCK"]
    fraud_probability: float
    latency_ms: float
    features_used: Dict[str, float]


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "engine_ready": scoring_engine is not None,
        "feature_store_ready": feature_store is not None,
        "drift_monitor_ready": drift_service is not None,
    }


@app.post("/api/v1/score", response_model=ScoreResponse)
def score_transaction(req: ScoreRequest):
    if feature_store is None or scoring_engine is None:
        raise HTTPException(status_code=503, detail="Scoring engine not ready.")

    tx_id = f"tx_{int(time.time() * 1000)}"
    now_iso = datetime.now(timezone.utc).isoformat()

    event = TransactionEvent(
        transaction_id=tx_id,
        user_id=req.user_id,
        merchant_id=req.merchant_id,
        amount=req.amount,
        timestamp=now_iso,
        device_id=req.device_id,
        ip_address=req.ip_address,
        is_fraud_ground_truth=0,
    )

    # 1. Sub-millisecond online feature store lookup
    feats = feature_store.update_and_get_features(event)

    # 2. Accelerated ONNX Runtime prediction
    verdict, fraud_prob, latency_ms = scoring_engine.score(feats)

    # 3. Log to async buffer for drift monitoring
    recent_features_log.append(feats)
    recent_probs_log.append(fraud_prob)
    if len(recent_features_log) > 5000:
        recent_features_log.pop(0)
        recent_probs_log.pop(0)

    return ScoreResponse(
        transaction_id=tx_id,
        user_id=req.user_id,
        verdict=verdict,
        fraud_probability=fraud_prob,
        latency_ms=latency_ms,
        features_used=feats,
    )


@app.get("/api/v1/drift-status")
def get_drift_status():
    if drift_service is None or len(recent_features_log) < 50:
        return {
            "status": "INSUFFICIENT_DATA",
            "message": f"Need at least 50 scored transactions; currently {len(recent_features_log)}.",
        }

    curr_df = pd.DataFrame(recent_features_log[-500:])
    curr_probs = np.array(recent_probs_log[-500:])
    return drift_service.evaluate_drift(curr_df, curr_probs)
