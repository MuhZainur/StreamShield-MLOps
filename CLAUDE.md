# CLAUDE.md - Developer Guidelines for StreamShield-MLOps

## Project Overview
StreamShield-MLOps is a real-time event-driven MLOps and anomaly detection pipeline using Apache Kafka/Redpanda, Feast Feature Store, Redis online caching, ONNX Runtime inference, and Evidently AI drift monitoring.

## Development Commands
- Start infrastructure containers: `docker-compose up -d`
- Install dependencies: `uv pip install -e ".[dev]"`
- Apply Feast feature store registry: `cd feast_repo && feast apply`
- Run test suite: `pytest tests/ -v`
- Run linting: `ruff check src/ tests/`
- Start scoring API: `uvicorn src.serving.api:app --reload --port 8000`
- Launch Streamlit dashboard: `streamlit run src/dashboard/app.py`

## Architecture & Code Invariants
- P99 latency is an absolute SLA constraint: all code paths in `src/serving/api.py` must execute without blocking I/O or unnecessary memory copies.
- Never write ad-hoc SQL feature pipelines in the API: all features must resolve strictly through the Feast feature store client to preserve zero train-serve skew.
- Ingestion errors or corrupt JSON events must be routed to the Dead Letter Queue (DLQ), never silently swallowed.
- Retraining triggers must always be gated by automated validation checks against a golden holdout set before new weights replace production ONNX binaries.
