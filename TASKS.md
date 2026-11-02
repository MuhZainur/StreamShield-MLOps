# Task Backlog & Execution Status
## StreamShield-MLOps

---

| Task ID | Task Description | Status | Assignee | Deliverable |
| :--- | :--- | :---: | :---: | :--- |
| **TSK-OPS-01** | Create project scaffolding & docker-compose stack | ✅ Completed | System | Project structure & config |
| **TSK-OPS-02** | Complete PRD, Architecture, and SLA blueprints | ✅ Completed | System | Markdown specs |
| **TSK-OPS-03** | Configure Feast feature store repo (Redis + DuckDB) | ⏳ Pending | `@feature-store` | `feast_repo/feature_store.yaml` |
| **TSK-OPS-04** | Implement streaming event consumer & feature writer | ⏳ Pending | `@streaming-infra` | `src/streaming/event_consumer.py` |
| **TSK-OPS-05** | Implement synthetic transaction generator (drift injection) | ⏳ Pending | `@streaming-infra` | `src/streaming/event_producer.py` |
| **TSK-OPS-06** | Train LightGBM model & export to ONNX runtime | ⏳ Pending | `@mlops-monitoring` | `src/training/train_baseline.py` |
| **TSK-OPS-07** | Implement FastAPI scoring service with sub-15ms P99 SLA | ⏳ Pending | `@mlops-monitoring` | `src/serving/api.py` |
| **TSK-OPS-08** | Implement Evidently AI drift detection worker | ⏳ Pending | `@mlops-monitoring` | `src/monitoring/drift_detector.py` |
| **TSK-OPS-09** | Build Prometheus metrics exporter & alerts | ⏳ Pending | System | `src/monitoring/prometheus_exporter.py` |
| **TSK-OPS-10** | Build Streamlit real-time observability dashboard | ⏳ Pending | System | `src/dashboard/app.py` |
