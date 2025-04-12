import joblib
import pandas as pd
from pathlib import Path

# Load the model
model = joblib.load("model.joblib")

# Get feature importances
importances = pd.Series(
    model.feature_importances_,
    index=[
        "n_steps",
        "duration_ms",
        "n_non_converted",
        "n_attempted",
        "n_converted",
        "n_at_risk",
        "attempt_ratio",
        "risk_ratio"
    ]
)

# Sort and display
print("\nFeature Importances:")
print(importances.sort_values(ascending=False)) 