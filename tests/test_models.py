import pytest
import pandas as pd
import numpy as np
from swing_trader.models.base import BaseModel, Signal
from swing_trader.models.random_forest import RandomForestModel

def test_signal_enum():
    assert Signal.BUY.value == 1
    assert Signal.HOLD.value == 0
    assert Signal.SELL.value == -1

def test_base_model_is_abstract():
    with pytest.raises(TypeError):
        BaseModel("test")

@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "rsi_14": np.random.uniform(20, 80, n),
        "macd": np.random.randn(n),
        "sma_20": np.random.uniform(90, 110, n),
        "volume_ratio": np.random.uniform(0.5, 2.0, n),
    })
    y = pd.Series(np.random.choice([-1, 0, 1], n))
    return X, y

def test_rf_fit_and_predict(sample_data):
    X, y = sample_data
    model = RandomForestModel()
    model.fit(X, y)

    assert model.is_fitted
    preds = model.predict(X)
    assert len(preds) == len(X)
    assert all(p in [-1, 0, 1] for p in preds)

def test_rf_predict_proba(sample_data):
    X, y = sample_data
    model = RandomForestModel()
    model.fit(X, y)

    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)

def test_rf_hyperparameters():
    model = RandomForestModel(n_estimators=50, max_depth=5)
    assert model.model.n_estimators == 50
    assert model.model.max_depth == 5
