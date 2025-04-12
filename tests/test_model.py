# Tests for model training and inference.

import pandas as pd
from sklearn.model_selection import train_test_split
from pathpilot.model import train, evaluate


def test_train_and_evaluate():
    # tiny synthetic dataset
    X = pd.DataFrame({"f1": [0, 1, 0, 1], "f2": [1, 0, 1, 0]})
    y = pd.Series([0, 1, 0, 1])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=1)
    model = train(X_tr, y_tr, n_estimators=10)
    metrics = evaluate(model, X_te, y_te)
    assert float(metrics["accuracy"]) >= 0.5

def test_train_model():
    pass

def test_predict_journey_outcome():
    pass 