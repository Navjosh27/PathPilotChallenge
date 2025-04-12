# Tests for journey fingerprint generation and feature engineering.

import pandas as pd
from pathpilot.features import compute_fingerprint


def test_generate_journey_fingerprint():
    pass

def test_fingerprint_counts():
    dummy = pd.DataFrame(
        {
            "session_id": ["s"] * 3,
            "status": ["non-converted", "attempted_conversion", "converted"],
            "start_ms": [0, 1000, 2000],
            "end_ms": [500, 1500, 2500],
        }
    )
    fp = compute_fingerprint(dummy)
    assert fp["n_steps"] == 3
    assert fp["n_converted"] == 1
    assert fp["attempt_ratio"] == 1 / 3 