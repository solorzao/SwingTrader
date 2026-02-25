import pytest
import pandas as pd
import numpy as np
from swing_trader.backtest.walk_forward import WalkForwardEngine, WalkForwardResult
from swing_trader.config import WalkForwardConfig, BacktestConfig
from swing_trader.models.random_forest import RandomForestModel


@pytest.fixture
def sample_data():
    """Create 500 days of data for walk-forward testing."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    X = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "f3": np.random.randn(n),
    }, index=dates)
    y = pd.Series(np.random.choice([-1, 0, 1], n), index=dates)
    prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(n) * 0.01)), index=dates)
    return X, y, prices


def rf_factory(X_train, y_train):
    model = RandomForestModel(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    return model


def test_walk_forward_expanding(sample_data):
    X, y, prices = sample_data
    engine = WalkForwardEngine(
        wf_config=WalkForwardConfig(
            mode="expanding", train_window_days=200, step_days=50
        ),
    )
    result = engine.run(X, y, prices, rf_factory)
    assert isinstance(result, WalkForwardResult)
    assert result.n_windows > 0
    assert len(result.oos_predictions) > 0
    assert result.backtest_result is not None


def test_walk_forward_rolling(sample_data):
    X, y, prices = sample_data
    engine = WalkForwardEngine(
        wf_config=WalkForwardConfig(
            mode="rolling", train_window_days=200, step_days=50
        ),
    )
    result = engine.run(X, y, prices, rf_factory)
    assert result.n_windows > 0
    # Rolling window: all training windows should be ~200 days
    for w in result.window_results:
        assert w.train_size <= 200


def test_walk_forward_per_window_metrics(sample_data):
    X, y, prices = sample_data
    engine = WalkForwardEngine(
        wf_config=WalkForwardConfig(train_window_days=200, step_days=50),
    )
    result = engine.run(X, y, prices, rf_factory)
    accs = result.per_window_accuracy
    assert len(accs) == result.n_windows
    assert all(0 <= a <= 1 for a in accs)


def test_walk_forward_too_little_data():
    X = pd.DataFrame({"f1": np.random.randn(50)}, index=pd.date_range("2024-01-01", periods=50))
    y = pd.Series(np.random.choice([-1, 0, 1], 50), index=X.index)
    prices = pd.Series(np.ones(50) * 100, index=X.index)

    engine = WalkForwardEngine(
        wf_config=WalkForwardConfig(train_window_days=200, step_days=50),
    )
    with pytest.raises(ValueError, match="train_window_days"):
        engine.run(X, y, prices, rf_factory)
