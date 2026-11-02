# AGENTS.md - Agent Roles & Operational Scopes

## Agent Personas & Operational Scopes

### 1. Streaming Infrastructure Engineer (`@streaming-infra`)
- Scope: Kafka/Redpanda topic provisioning, consumer group partition balancing, Redis memory policy.
- Tooling: Docker Compose, aiokafka, redis-py.
- Rules: Guarantee at-least-once message delivery semantics with strict partition ordering by `user_id`.

### 2. Feature Store Architect (`@feature-store`)
- Scope: Feast feature definitions, entity schema contracts, point-in-time offline historical joins.
- Tooling: Feast, DuckDB, Apache Parquet.
- Rules: Never modify an existing feature view schema without generating a versioned migration.

### 3. Model & Observability Engineer (`@mlops-monitoring`)
- Scope: ONNX conversion, inference engine optimization, Evidently AI drift metrics, Prometheus exporters.
- Tooling: ONNX Runtime, LightGBM, Evidently, Prometheus Client.
- Rules: Ensure drift calculation is completely decoupled from the low-latency inference path.
