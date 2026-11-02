"""StreamShield Operations Dashboard: Real-Time Stream, Latency & Drift Observability."""

import sys
from pathlib import Path

# Ensure repository root is on sys.path for Streamlit Cloud execution
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.features.feature_store import StreamFeatureStore, FEATURE_NAMES
from src.models.pipeline import FraudModelTrainer, ONNXScoringEngine
from src.monitoring.drift_detector import DriftMonitoringService
from src.streaming.event_stream import generate_synthetic_event

st.set_page_config(
    page_title="StreamShield MLOps Console",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ StreamShield-MLOps: Real-Time Event-Driven Observability Console")
st.caption("P99 < 12ms ONNX Serving | Online Feature Store | Automated KS & PSI Drift Detection")

# Setup Feature Store and Pipeline in Session
if "store" not in st.session_state:
    with st.spinner("Warming up baseline feature store and ONNX engine..."):
        store = StreamFeatureStore("data/offline/features.duckdb")
        records = []
        for i in range(1500):
            ev = generate_synthetic_event(i, drift_mode=False)
            feats = store.update_and_get_features(ev)
            store.log_to_offline_store(ev, feats)
            r = dict(feats)
            r["is_fraud_ground_truth"] = ev.is_fraud_ground_truth
            records.append(r)

        base_df = pd.DataFrame(records)
        trainer = FraudModelTrainer("models/fraud_detector.onnx")
        trainer.train_and_export(base_df)
        engine = ONNXScoringEngine("models/fraud_detector.onnx")

        base_probs = [engine.score(r)[1] for r in records]
        drift_mon = DriftMonitoringService(base_df, np.array(base_probs))

        st.session_state.store = store
        st.session_state.engine = engine
        st.session_state.drift_mon = drift_mon
        st.session_state.scored_history = []
        st.session_state.latencies = []
        st.session_state.probs = []

store = st.session_state.store
engine = st.session_state.engine
drift_mon = st.session_state.drift_mon

# Sidebar controls
with st.sidebar:
    st.header("🎮 Stream Controller")
    stream_batch = st.slider("Events to Ingest per Batch", 50, 500, 150, 50)
    drift_toggle = st.toggle("🚨 Inject Botnet Card-Testing Attack (Drift Wave)", value=False)
    ingest_btn = st.button("▶️ Stream Transactions", type="primary")

if ingest_btn:
    batch_records = []
    for i in range(stream_batch):
        ev = generate_synthetic_event(i, drift_mode=drift_toggle)
        feats = store.update_and_get_features(ev)
        verdict, prob, lat = engine.score(feats)

        st.session_state.latencies.append(lat)
        st.session_state.probs.append(prob)
        st.session_state.scored_history.append({
            "transaction_id": ev.transaction_id,
            "user_id": ev.user_id,
            "amount": ev.amount,
            "verdict": verdict,
            "fraud_probability": prob,
            "latency_ms": lat,
            **feats,
        })

# Compute Top Metrics
total_scored = len(st.session_state.scored_history)
if total_scored > 0:
    recent_df = pd.DataFrame(st.session_state.scored_history[-stream_batch:])
    p99_lat = float(np.percentile(st.session_state.latencies[-stream_batch:], 99))
    block_rate = float(np.mean(recent_df["verdict"] == "BLOCK")) * 100
    avg_lat = float(np.mean(st.session_state.latencies[-stream_batch:]))
else:
    recent_df = pd.DataFrame()
    p99_lat = 0.0
    block_rate = 0.0
    avg_lat = 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions Scored", f"{total_scored:,}")
col2.metric("P99 Scoring Latency", f"{p99_lat:.2f} ms", delta="< 12ms SLA" if p99_lat <= 12.0 else "SLA BREACH", delta_color="normal" if p99_lat <= 12.0 else "inverse")
col3.metric("Avg Latency", f"{avg_lat:.2f} ms")
col4.metric("Blocked Fraud Rate", f"{block_rate:.1f}%")

tab1, tab2, tab3 = st.tabs(["📊 Live Transaction Feed", "🛡️ Drift Observability (KS & PSI)", "⚙️ System Topology"])

with tab1:
    if not recent_df.empty:
        st.subheader("Live Scoring Decisions")
        st.dataframe(
            recent_df[["transaction_id", "user_id", "amount", "verdict", "fraud_probability", "latency_ms"]].tail(15),
            use_container_width=True,
        )

        col_pie, col_lat = st.columns(2)
        with col_pie:
            fig_v = px.pie(recent_df, names="verdict", title="Decision Verdicts", color="verdict", color_discrete_map={"ALLOW": "#10b981", "REVIEW": "#f59e0b", "BLOCK": "#ef4444"}, hole=0.4)
            st.plotly_chart(fig_v, use_container_width=True)
        with col_lat:
            fig_l = px.histogram(recent_df, x="latency_ms", nbins=20, title="Inference Latency Distribution (ms)", color_discrete_sequence=["#3b82f6"])
            st.plotly_chart(fig_l, use_container_width=True)
    else:
        st.info("Click 'Stream Transactions' in the sidebar to generate live traffic.")

with tab2:
    st.subheader("Statistical Data & Concept Drift Monitoring")
    if total_scored >= 50:
        drift_data = pd.DataFrame(st.session_state.scored_history[-300:])
        curr_probs = np.array(st.session_state.probs[-300:])
        drift_res = drift_mon.evaluate_drift(drift_data[FEATURE_NAMES], curr_probs)

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.markdown(f"### Status: `{drift_res['status']}`")
            if drift_res["concept_drift_detected"]:
                st.error(f"🚨 **CRITICAL CONCEPT DRIFT**: PSI = {drift_res['prediction_psi']:.4f} (Threshold: 0.20)")
            else:
                st.success(f"✅ **CONCEPT HEALTHY**: PSI = {drift_res['prediction_psi']:.4f} (Threshold: 0.20)")

            st.write(f"Drifted Features: **{drift_res['drifted_features_count']} / {drift_res['features_evaluated']}**")

        with d_col2:
            ks_rows = []
            for feat, details in drift_res["feature_details"].items():
                ks_rows.append({"Feature": feat, "KS Statistic": details["ks_statistic"], "p-value": details["p_value"], "Drift": details["drifted"]})
            st.dataframe(pd.DataFrame(ks_rows), use_container_width=True)
    else:
        st.warning("Need at least 50 streamed events to calculate statistical drift.")

with tab3:
    st.subheader("StreamShield Architecture Components")
    st.markdown("""
    | Component | Implementation | Performance Metric |
    | :--- | :--- | :--- |
    | **Streaming Broker** | In-Memory Async Ring Buffer (Kafka-compatible) | Zero external daemon required for local dev |
    | **Online Feature Store** | In-Memory / Redis Hash Table | $P99 < 1.0\text{ ms}$ feature retrieval |
    | **Offline Feature Store** | DuckDB + Columnar Parquet | Point-in-time joins with zero train-serve skew |
    | **Inference Engine** | ONNX Runtime C++ backend with AVX-512 | $P99 < 5\text{ ms}$ scoring latency |
    | **Drift Monitoring** | Kolmogorov-Smirnov (KS) test & Population Stability Index | Continuous sliding-window drift auditing |
    """)
