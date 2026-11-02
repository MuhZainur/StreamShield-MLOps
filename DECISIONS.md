# Architectural Decision Records (ADR)
## StreamShield-MLOps

---

### ADR-001: Redpanda over Vanilla Apache Kafka for Local & Edge Scaffolding
- **Date:** 2026-11-02
- **Context:** Standard Apache Kafka requires JVM runtime and Apache ZooKeeper or KRaft metadata management, consuming high memory in lightweight local environments.
- **Decision:** Use Redpanda (written in C++, single binary container, zero JVM) while maintaining 100% Kafka wire protocol compatibility.
- **Consequences:** 70% lower container memory overhead, sub-second startup, and native Kafka client library compatibility.

### ADR-002: Feast with Redis Online & DuckDB Offline Storage
- **Date:** 2026-11-02
- **Context:** We need sub-3ms online feature retrieval for scoring while enabling fast point-in-time historical joins without running a heavy Spark cluster.
- **Decision:** Pair Feast with Redis for online storage and DuckDB for offline columnar parquet processing.
- **Consequences:** Near-instantaneous feature lookups without cloud database dependencies.

### ADR-003: ONNX Runtime for Low-Latency Tree Inference
- **Date:** 2026-11-02
- **Context:** Python-based model serving via native `xgboost.predict` introduces GIL bottlenecks and inconsistent P99 latency spikes under concurrent load.
- **Decision:** Compile trained tree models into ONNX format (`.onnx`) and execute using ONNX Runtime C++ backend.
- **Consequences:** Sub-4ms P99 inference latency and reduced thread-locking overhead.
