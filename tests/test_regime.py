import pytest
import pandas as pd
import numpy as np
from swing_trader.evaluation.regime import RegimeAnalyzer


@pytest.fixture
def market_data():
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(n) * 0.02)), index=dates)
    signals = pd.Series(np.random.choice([-1, 0, 1], n), index=dates)
    return prices, signals


def test_detect_regimes(market_data):
    prices, _ = market_data
    analyzer = RegimeAnalyzer(vol_window=30)
    labels, vol, threshold = analyzer.detect_regimes(prices)
    assert set(labels.unique()).issubset({"low_vol", "high_vol"})
    assert threshold > 0
    assert len(vol) > 0


def test_analyze_returns_report(market_data):
    prices, signals = market_data
    analyzer = RegimeAnalyzer(vol_window=30)
    report = analyzer.analyze(prices, signals)
    assert report.threshold > 0
    assert len(report.regime_labels) > 0
    # Should have at least one regime with metrics
    assert len(report.regime_metrics) > 0


def test_regime_split_covers_data(market_data):
    prices, _ = market_data
    analyzer = RegimeAnalyzer(vol_window=30)
    labels, _, _ = analyzer.detect_regimes(prices)
    # All labels should be either high or low vol
    assert labels.isin(["low_vol", "high_vol"]).all()
