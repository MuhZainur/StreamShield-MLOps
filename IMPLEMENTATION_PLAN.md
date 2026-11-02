# Implementation Plan & Milestone Roadmap
## StreamShield-MLOps: Real-Time Event-Driven MLOps Pipeline

---

### Phase 1: Infrastructure Scaffolding & Local Containers (Days 1–2)
- [ ] Initialize Python 3.11 virtual environment with `uv`.
- [ ] Configure `docker-compose.yml` for local service orchestration:
  - Redpanda / Kafka single-node broker + console UI.
  - Redis 7 server for online feature caching.
  - Prometheus and Grafana for metrics scraping and dashboards.
- [ ] Verify inter-container connectivity and network health checks.

---

### Phase 2: Feature Store Definition & Ingestion Pipeline (Days 3–5)
- [ ] Setup Feast repository in `feast_repo/`:
  - Define entity definitions: `user_id`, `merchant_id`.
  - Define feature views: rolling velocities (1h, 24h, 7d), historical refund rate, distance anomaly.
  - Configure online store (Redis) and offline store (DuckDB/Parquet).
- [ ] Implement `src/streaming/event_consumer.py`:
  - Consume raw transaction events from Kafka.
  - Compute sliding window stateful aggregates.
  - Push real-time updates directly into Redis online store via Feast push API.
- [ ] Implement `src/streaming/event_producer.py`:
  - High-speed synthetic transaction generator simulating normal vs fraud attacks.

---

### Phase 3: Model Training, ONNX Export & Low-Latency Serving (Days 6–7)
- [ ] Implement `src/training/train_baseline.py`:
  - Train LightGBM / XGBoost fraud classification model using Feast historical point-in-time join.
  - Optimize hyperparameters with Optuna.
  - Export trained model to ONNX Runtime format (`models/fraud_detector.onnx`).
- [ ] Implement `src/serving/api.py`:
  - FastAPI endpoint `POST /api/v1/score`:
    - Fetch online features from Redis.
    - Run ONNX session.
    - Return verdict and latency telemetry.
  - Benchmark P99 latency with Locust / vegeta load generator.

---

### Phase 4: Drift Observability & Automated Retraining Loop (Days 8–10)
- [ ] Implement `src/monitoring/drift_detector.py`:
  - Ingest live scoring logs in 5-minute micro-batches.
  - Execute Kolmogorov-Smirnov test and Population Stability Index (PSI) via Evidently AI.
  - Export Prometheus metrics (`data_drift_p_value`, `model_psi_score`).
- [ ] Implement `src/pipeline/retraining_trigger.py`:
  - Trigger automated retraining workflow upon critical drift condition ($\text{PSI} > 0.2$).
  - Execute shadow model evaluation against historical holdout split.
  - Output automated HTML and Markdown drift reports in `reports/`.

---

### Phase 5: Interactive Dashboard & Chaos Testing (Days 11–12)
- [ ] Build Streamlit operational dashboard (`src/dashboard/app.py`):
  - Live transaction stream telemetry.
  - Real-time fraud alert feed.
  - Real-time feature drift visualizer and KS-test p-value heatmap.
- [ ] Run Chaos Test: simulate sudden concept drift (flash sale fraud wave) and verify automated detection and alert triggering.
