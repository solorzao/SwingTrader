import pytest
import pandas as pd
import numpy as np
from swing_trader.features.indicators import TechnicalIndicators

@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    return pd.DataFrame({
        "open": close + np.random.randn(n),
        "high": close + abs(np.random.randn(n)),
        "low": close - abs(np.random.randn(n)),
        "close": close,
        "volume": np.random.randint(1000000, 10000000, n)
    }, index=dates)

def test_add_momentum_indicators(sample_ohlcv):
    ti = TechnicalIndicators()
    df = ti.add_all(sample_ohlcv)
    momentum_cols = ["rsi_14", "macd", "macd_signal", "roc_10"]
    for col in momentum_cols:
        assert col in df.columns

def test_add_moving_averages(sample_ohlcv):
    ti = TechnicalIndicators()
    df = ti.add_all(sample_ohlcv)
    ma_cols = ["sma_20", "sma_50", "ema_12", "ema_26"]
    for col in ma_cols:
        assert col in df.columns

def test_add_volume_indicators(sample_ohlcv):
    ti = TechnicalIndicators()
    df = ti.add_all(sample_ohlcv)
    assert "obv" in df.columns
    assert "volume_sma_20" in df.columns

def test_dropna_removes_warmup_period(sample_ohlcv):
    ti = TechnicalIndicators()
    df = ti.add_all(sample_ohlcv, dropna=True)
    assert not df.isnull().any().any()
