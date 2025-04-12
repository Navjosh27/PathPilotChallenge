# Machine learning model training and inference utilities.

from pathlib import Path
from typing import Dict
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

__all__ = ["train", "evaluate", "save", "load"]


def train(X, y, n_estimators: int = 200, random_state: int = 42) -> RandomForestClassifier:
    """Fit and return a Random‑Forest model."""
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
    )
    clf.fit(X, y)
    return clf


def evaluate(model: RandomForestClassifier, X_test, y_test) -> Dict[str, str]:
    """Return accuracy and full sklearn classification_report as a dict."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3, output_dict=False)
    return {"accuracy": f"{acc:.3f}", "report": report}


def save(model: RandomForestClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load(path: Path) -> RandomForestClassifier:
    return joblib.load(path)

def train_model():
    pass

def predict_journey_outcome():
    pass 