# System Architecture & Technical Specification
## StreamShield-MLOps: Real-Time Event-Driven MLOps Pipeline

---

### 1. High-Level Architecture Flowchart

```mermaid
flowchart TD
    subgraph EventStream ["1. Real-Time Streaming Ingestion"]
        TxGen["Client / Transaction Simulator"] -->|JSON Event Stream| Kafka["Apache Kafka / Redpanda Cluster"]
        Kafka --> StreamConsumer["Streaming Feature Transformer (Python / Faust)"]
    end

    subgraph FeatureStoreLayer ["2. Unified Feature Store (Feast)"]
        StreamConsumer -->|Write Online Features| Redis["Redis In-Memory (Online Store <3ms)"]
        StreamConsumer -->|Append Historical Batch| Parquet["DuckDB / Parquet (Offline Store)"]
        FeastRepo["Feast Feature Definitions & Registry"] --> Redis & Parquet
    end

    subgraph InferenceLayer ["3. Real-Time Scoring Microservice"]
        API["FastAPI Scoring Endpoint"] -->|Fetch Entities| Redis
        Redis -->|Return Feature Vector| API
        API --> ONNX["ONNX Runtime Engine (LightGBM/XGBoost)"]
        ONNX --> Verdict["Decision Verdict: ALLOW / REVIEW / BLOCK"]
        ONNX --> LogStream["Inference Log Queue (Redis / Kafka)"]
    end

    subgraph MonitoringLayer ["4. Drift Observability & Retraining Loop"]
        LogStream --> Evidently["Evidently AI Drift Engine"]
        Evidently -->|Compute KS-Test & PSI| Prom["Prometheus Metrics & Grafana"]
        Evidently -->|Threshold Exceeded: PSI > 0.2| Trigger["Automated Retraining Webhook"]
        Trigger --> RetrainJob["Offline Training Job (DuckDB + LightGBM)"]
        RetrainJob --> Evaluator["Model Validation & Canary Deployment"]
        Evaluator -->|Shadow Traffic Test Passed| ONNX
    end
```

---

### 2. Feature Store Architecture: Eliminating Train-Serve Skew

```mermaid
sequenceDiagram
    autonumber
    actor Client as Checkout Client
    participant API as FastAPI Inference
    participant Redis as Redis Online Store
    participant ONNX as ONNX Model Engine
    participant Kafka as Event Kafka Stream
    participant Offline as Parquet Offline Store

    Note over Client,API: Real-Time Low Latency Inference Path (<15ms)
    Client->>API: POST /api/v1/score {user_id: 1042, amount: $1250, merchant_id: "M89"}
    API->>Redis: Feast.get_online_features(user_id=1042)
    Redis-->>API: [tx_count_1h=7, avg_amount_30d=45.2, velocity_score=3.8]
    API->>ONNX: onnx_session.run(features)
    ONNX-->>API: {fraud_prob: 0.94, verdict: "BLOCK"}
    API-->>Client: HTTP 200 {verdict: "BLOCK", reason: "Abnormal velocity"}

    Note over API,Offline: Asynchronous Event Sinks
    API-)Kafka: Emit transaction_scored_event
    Kafka-)Offline: Persist historical row for point-in-time training
```

---

### 3. Mathematical Foundations of Drift Detection

#### A. Kolmogorov-Smirnov (KS) Test for Numerical Data Drift
To determine whether an incoming feature distribution $F(x)$ differs significantly from the baseline training distribution $F_0(x)$, we compute the supremum statistic:

$$D_{n, m} = \sup_{x} |F_n(x) - F_{0, m}(x)|$$

Under the null hypothesis $H_0: F(x) = F_0(x)$, if the p-value $< 0.05$, the feature is flagged as statistically drifted.

#### B. Population Stability Index (PSI) for Concept & Output Drift
Used to evaluate whether the distribution of model output probabilities or bucketed features has shifted:

$$\text{PSI} = \sum_{k=1}^{B} \left( P_k - B_k \right) \times \ln\left( \frac{P_k}{B_k} \right)$$
Where:
- $B_k$: Proportion of transactions falling into bin $k$ during baseline reference.
- $P_k$: Proportion of transactions falling into bin $k$ during the current evaluation window.

**Action Thresholds:**
- $\text{PSI} < 0.10$: No significant change; model is healthy.
- $0.10 \le \text{PSI} \le 0.20$: Moderate shift; trigger diagnostic inspection.
- $\text{PSI} > 0.20$: Severe concept drift; trigger automated retraining immediately.

---

### 4. Technical Stack Manifest

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Streaming Broker** | Redpanda / Apache Kafka | High-throughput distributed message bus, 100% Kafka API compatible. |
| **Feature Store** | Feast 0.40+ | Industry-standard open-source feature store supporting dual Redis/Parquet backends. |
| **Online Storage** | Redis 7+ (Alpine) | Sub-3ms P99 key-value lookup for high-frequency entity features. |
| **Offline Storage** | DuckDB + Apache Parquet | Columnar zero-copy OLAP query engine for historical point-in-time training joins. |
| **Inference Engine** | ONNX Runtime (C++ backend with Python bindings) | 3.5x faster inference latency than native Python XGBoost. |
| **Drift Engine** | Evidently AI + SciPy | Comprehensive statistical drift and data quality reporting. |
| **Metrics & Alerting** | Prometheus + Grafana | Standard SRE observability stack for p99 latency and drift metrics. |
