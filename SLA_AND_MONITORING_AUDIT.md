# Production SLA & Observability Audit
## StreamShield-MLOps: Operational Guidelines

---

### 1. Latency & Reliability Service Level Agreements (SLAs)

| Service Tier | Endpoint / Operation | P50 SLA | P95 SLA | P99 SLA | Max Error Budget |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1 (Critical)** | `POST /api/v1/score` (Full Pipeline) | $\le 4\text{ ms}$ | $\le 8\text{ ms}$ | $\le 12\text{ ms}$ | $0.05\%$ (99.95% Uptime) |
| **Tier 1 (Critical)** | Redis Online Feature Fetch | $\le 1.2\text{ ms}$ | $\le 2.5\text{ ms}$ | $\le 3.5\text{ ms}$ | $0.01\%$ |
| **Tier 2 (High)** | Kafka Event Ingestion & Ingestion Lag | $\le 15\text{ ms}$ | $\le 45\text{ ms}$ | $\le 100\text{ ms}$ | Lag $< 500\text{ ms}$ |
| **Tier 3 (Batch)** | 5-Minute Drift Evaluation Job | $\le 15\text{ sec}$ | $\le 30\text{ sec}$ | $\le 45\text{ sec}$ | Missed runs $< 1\text{ per day}$ |

---

### 2. Prometheus Metrics & Alerting Runbook

```yaml
# Prometheus Alerting Rule Spec
groups:
  - name: streamshield_alerts
    rules:
      - alert: CriticalModelConceptDrift
        expr: model_psi_score > 0.20
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Model Concept Drift Exceeded (PSI: {{ $value }})"
          description: "Production prediction distribution has deviated significantly from baseline training weights."

      - alert: P99LatencyExceeded
        expr: histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[2m])) by (le)) > 0.015
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "P99 scoring latency above 15ms SLA"
```
