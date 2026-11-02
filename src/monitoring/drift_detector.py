"""Continuous Statistical Drift Observability Engine (KS-Test & PSI)."""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.features.feature_store import FEATURE_NAMES


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10,
) -> float:
    """Calculate Population Stability Index (PSI) between reference and current distribution."""
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Define bins based on reference distribution quantiles
    quantiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(expected, quantiles)
    bins[0] = -np.inf
    bins[-1] = np.inf
    # Ensure strictly increasing bins
    bins = np.unique(bins)
    if len(bins) < 3:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    expected_pct = np.maximum(expected_counts / max(len(expected), 1), 1e-4)
    actual_pct = np.maximum(actual_counts / max(len(actual), 1), 1e-4)

    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_value)


class DriftMonitoringService:
    """Monitors live transaction feature and score distributions against baseline."""

    def __init__(
        self,
        baseline_df: pd.DataFrame,
        baseline_probs: np.ndarray,
        psi_threshold: float = 0.20,
        ks_p_value_threshold: float = 0.05,
    ):
        self.baseline_df = baseline_df[FEATURE_NAMES]
        self.baseline_probs = baseline_probs
        self.psi_threshold = psi_threshold
        self.ks_p_value_threshold = ks_p_value_threshold

    def evaluate_drift(
        self,
        current_df: pd.DataFrame,
        current_probs: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute KS-tests for all features and PSI for prediction probabilities."""
        feature_drift_results: Dict[str, Dict[str, float]] = {}
        drifted_features_count = 0

        for col in FEATURE_NAMES:
            base_col = self.baseline_df[col].values
            curr_col = current_df[col].values

            ks_stat, p_val = ks_2samp(base_col, curr_col)
            is_drifted = bool(p_val < self.ks_p_value_threshold)
            if is_drifted:
                drifted_features_count += 1

            feature_drift_results[col] = {
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": round(float(p_val), 5),
                "drifted": is_drifted,
            }

        psi_score = calculate_psi(self.baseline_probs, current_probs)
        is_concept_drift = psi_score >= self.psi_threshold

        status = "ALERT_DRIFT_DETECTED" if (is_concept_drift or drifted_features_count >= 2) else "HEALTHY"

        return {
            "status": status,
            "prediction_psi": round(psi_score, 4),
            "concept_drift_detected": is_concept_drift,
            "drifted_features_count": drifted_features_count,
            "features_evaluated": len(FEATURE_NAMES),
            "feature_details": feature_drift_results,
        }
