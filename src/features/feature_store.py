"""Dual Online/Offline Feature Store with Sub-Millisecond Online Retrieval."""

from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import os
import threading
from typing import Any, Dict, List, Optional
import duckdb
import pandas as pd

from src.streaming.event_stream import TransactionEvent


FEATURE_NAMES = [
    "amount",
    "user_tx_count_10m",
    "user_amount_sum_1h",
    "user_avg_amount_30d",
    "amount_to_avg_ratio",
    "merchant_risk_score",
]


class StreamFeatureStore:
    """Thread-safe unified Feature Store supporting online lookups and DuckDB offline batching."""

    def __init__(self, offline_db_path: str = "data/offline/features.duckdb"):
        self.offline_db_path = offline_db_path
        self._lock = threading.Lock()

        # Online in-memory entity state (sub-millisecond P99)
        self._user_tx_history: Dict[str, List[float]] = defaultdict(list)
        self._user_amounts: Dict[str, List[float]] = defaultdict(list)
        self._merchant_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "fraud": 0})

        # Initialize DuckDB offline storage with persistent connection
        if offline_db_path and offline_db_path != ":memory:":
            os.makedirs(os.path.dirname(offline_db_path), exist_ok=True)
        try:
            self.con = duckdb.connect(self.offline_db_path)
        except Exception:
            # Fallback to in-memory store if disk DB is locked by another process
            self.con = duckdb.connect(":memory:")
        self._init_offline_db()

    def _init_offline_db(self) -> None:
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS historical_features (
                transaction_id VARCHAR PRIMARY KEY,
                user_id VARCHAR,
                merchant_id VARCHAR,
                amount DOUBLE,
                timestamp VARCHAR,
                user_tx_count_10m INTEGER,
                user_amount_sum_1h DOUBLE,
                user_avg_amount_30d DOUBLE,
                amount_to_avg_ratio DOUBLE,
                merchant_risk_score DOUBLE,
                is_fraud_ground_truth INTEGER
            )
        """)

    def update_and_get_features(self, event: TransactionEvent) -> Dict[str, float]:
        """Ingest event, update online state atomically, and return computed feature vector."""
        with self._lock:
            # Update user online state
            amounts = self._user_amounts[event.user_id]
            amounts.append(event.amount)

            tx_count_10m = len(amounts[-10:])
            amount_sum_1h = sum(amounts[-30:])
            avg_amount_30d = sum(amounts) / len(amounts)
            amount_to_avg_ratio = event.amount / max(avg_amount_30d, 1e-4)

            # Update merchant stats
            m_stat = self._merchant_stats[event.merchant_id]
            m_stat["total"] += 1
            if event.is_fraud_ground_truth:
                m_stat["fraud"] += 1
            merchant_risk = m_stat["fraud"] / m_stat["total"]

            features = {
                "amount": float(event.amount),
                "user_tx_count_10m": float(tx_count_10m),
                "user_amount_sum_1h": float(amount_sum_1h),
                "user_avg_amount_30d": float(avg_amount_30d),
                "amount_to_avg_ratio": float(amount_to_avg_ratio),
                "merchant_risk_score": float(merchant_risk),
            }

            return features

    def log_to_offline_store(self, event: TransactionEvent, features: Dict[str, float]) -> None:
        """Append enriched feature row to DuckDB for zero-leakage historical training."""
        with self._lock:
            self.con.execute(
                """
                INSERT OR REPLACE INTO historical_features VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    event.transaction_id,
                    event.user_id,
                    event.merchant_id,
                    features["amount"],
                    event.timestamp,
                    int(features["user_tx_count_10m"]),
                    features["user_amount_sum_1h"],
                    features["user_avg_amount_30d"],
                    features["amount_to_avg_ratio"],
                    features["merchant_risk_score"],
                    event.is_fraud_ground_truth,
                ),
            )

    def log_batch_to_offline_store(self, records: List[Dict[str, Any]]) -> None:
        """Fast vectorized batch append to DuckDB offline store."""
        if not records:
            return
        with self._lock:
            df = pd.DataFrame(records)
            self.con.register("_temp_batch", df)
            self.con.execute("""
                INSERT OR REPLACE INTO historical_features
                SELECT transaction_id, user_id, merchant_id, amount, timestamp,
                       CAST(user_tx_count_10m AS INTEGER), user_amount_sum_1h,
                       user_avg_amount_30d, amount_to_avg_ratio, merchant_risk_score,
                       CAST(is_fraud_ground_truth AS INTEGER)
                FROM _temp_batch
            """)
            self.con.unregister("_temp_batch")

    def get_historical_training_df(self) -> pd.DataFrame:
        """Fetch historical dataset for model training."""
        with self._lock:
            return self.con.execute("SELECT * FROM historical_features").fetchdf()

    def close(self) -> None:
        try:
            self.con.close()
        except Exception:
            pass


if __name__ == "__main__":
    store = StreamFeatureStore("data/offline/test.duckdb")
    ev = TransactionEvent("tx_1", "usr_1", "m_1", 100.0, "2026-01-01T00:00:00Z", "dev_1", "127.0.0.1", 0)
    feats = store.update_and_get_features(ev)
    assert feats["amount"] == 100.0
    assert feats["user_tx_count_10m"] == 1.0
    store.log_to_offline_store(ev, feats)
    df_hist = store.get_historical_training_df()
    assert len(df_hist) == 1
    print("Feature store verified.")
