# Dataset Specification & Synthetic Stream Generator
## StreamShield-MLOps: Real-Time Fraud & Anomaly Detection

---

### 1. Schema Definition & Event Topology
Incoming event payloads follow the strict JSON schema:

```json
{
  "transaction_id": "tx_984182903",
  "timestamp": "2026-11-02T15:30:00Z",
  "user_id": "usr_40219",
  "merchant_id": "merch_target_89",
  "amount": 142.50,
  "currency": "USD",
  "payment_method": "credit_card",
  "device_id": "dev_android_98a7",
  "ip_address": "192.168.1.100",
  "location": {
    "latitude": 37.7749,
    "longitude": -122.4194
  }
}
```

---

### 2. Feature Store Engineered Features (Feast)
From the raw stream, Feast derives online and offline features:

| Feature Name | Feature Type | Time Window | Calculation Logic |
| :--- | :--- | :--- | :--- |
| `user_tx_count_10m` | Integer | 10 minutes | Count of transactions initiated by user in last 10m. |
| `user_amount_sum_1h` | Float | 1 hour | Total currency volume spent by user in last 1 hour. |
| `user_avg_amount_30d` | Float | 30 days | Historical baseline spending average for this user. |
| `amount_to_avg_ratio` | Float | Real-time | `amount / (user_avg_amount_30d + 1e-5)` |
| `speed_kmh_since_last_tx` | Float | Inter-event | Great-circle distance between current and last location divided by $\Delta t$. |
| `merchant_risk_score` | Float | 7 days | Historical fraud dispute chargeback rate for merchant. |

---

### 3. Drift Simulation Scenarios
To rigorously test the Evidently AI drift engine, the synthetic generator supports 3 anomaly injection modes:
1. **Normal Baseline Mode:** Stable Gaussian spending patterns ($\mu = \$45, \sigma = \$20$), steady inter-arrival times.
2. **Flash Sale Covariate Shift (Data Drift):** Spikes in overall transaction amount without fraud rate increase (higher volumes, normal distributions).
3. **Card-Testing Botnet Concept Drift:** Rapid-fire micro-transactions ($\$0.50 - \$2.00$) from rotating IP addresses with sudden geographic teleportation.
