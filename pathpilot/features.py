# Functions for generating journey fingerprints and feature engineering.

from __future__ import annotations
import pandas as pd
from typing import Tuple, Dict

__all__ = ["compute_fingerprint", "build_feature_matrix"]


# ---------- per‑session fingerprint ---------- #
def compute_fingerprint(df: pd.DataFrame) -> pd.Series:
    """
    Aggregate a single session (already filtered) into numeric features.
    """
    duration_ms = df["end_ms"].max() - df["start_ms"].min()
    counts = df["status"].value_counts().to_dict()

    # basic counts defaulting to 0
    get = lambda k: counts.get(k, 0)

    features: Dict[str, float] = {
        "n_steps": len(df),
        "duration_ms": duration_ms,
        "n_non_converted": get("non-converted"),
        "n_attempted": get("attempted_conversion"),
        "n_converted": get("converted"),
        "n_at_risk": get("conversion_at_risk"),
    }

    # ratios (protect div‑by‑zero)
    total = features["n_steps"]
    features["attempt_ratio"] = features["n_attempted"] / total if total else 0.0
    features["risk_ratio"] = features["n_at_risk"] / total if total else 0.0

    return pd.Series(features)


# ---------- full feature matrix ---------- #
def build_feature_matrix(
    df_all: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    From the flattened & cleaned DataFrame produce:
        X  – DataFrame of features (one row per session)
        y  – Series (1 = upgrade, 0 = cancellation)
    """
    # group and compute fingerprints
    grouped = df_all.groupby("session_id").apply(compute_fingerprint)

    # ground‑truth from directory name
    y = (df_all.groupby("session_id")["label_dir"]
                .first()
                .map({"upgrades": 1, "cancellations": 0}))

    # align index
    X = grouped.sort_index()
    y = y.loc[X.index]

    return X, y 