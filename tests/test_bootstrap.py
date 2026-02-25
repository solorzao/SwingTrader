import pytest
import numpy as np
import pandas as pd
from swing_trader.evaluation.bootstrap import (
    bootstrap_metric, bootstrap_sharpe, bootstrap_returns, bootstrap_win_rate,
)


def test_bootstrap_metric_basic():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = bootstrap_metric(data, np.mean, n_samples=500)
    assert 1.0 < result.point_estimate < 5.0
    assert result.ci_lower <= result.point_estimate <= result.ci_upper
    assert len(result.samples) == 500


def test_bootstrap_sharpe():
    np.random.seed(42)
    returns = pd.Series(np.random.randn(252) * 0.01 + 0.001)
    result = bootstrap_sharpe(returns, n_samples=500)
    assert result.metric_name == "sharpe_ratio"
    assert result.ci_lower < result.ci_upper


def test_bootstrap_returns():
    np.random.seed(42)
    returns = pd.Series(np.random.randn(100) * 0.01 + 0.002)
    result = bootstrap_returns(returns, n_samples=500)
    assert result.metric_name == "total_return"
    assert result.point_estimate != 0


def test_bootstrap_win_rate():
    pnls = np.array([100, -50, 200, -30, 150, 80, -20, 60])
    result = bootstrap_win_rate(pnls, n_samples=500)
    assert result.metric_name == "win_rate"
    assert 0 <= result.point_estimate <= 1
    assert 0 <= result.ci_lower <= result.ci_upper <= 1


def test_bootstrap_significance_positive():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = bootstrap_metric(data, np.mean, n_samples=1000)
    assert result.is_significant  # all positive data -> CI above zero


def test_bootstrap_significance_mixed():
    data = np.array([-2, -1, 0, 1, 2])
    result = bootstrap_metric(data, np.mean, n_samples=1000)
    # Mean is ~0, CI should cross zero
    assert not result.is_significant
