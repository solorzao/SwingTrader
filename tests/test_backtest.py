import pytest
import pandas as pd
import numpy as np
from swing_trader.backtest.engine import BacktestEngine, BacktestResult

@pytest.fixture
def sample_backtest_data():
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
    signals = np.random.choice([-1, 0, 1], n)
    return pd.DataFrame({"close": close, "signal": signals}, index=dates)

def test_backtest_returns_result(sample_backtest_data):
    engine = BacktestEngine(initial_capital=10000)
    result = engine.run(sample_backtest_data)
    assert isinstance(result, BacktestResult)
    assert hasattr(result, "total_return")
    assert hasattr(result, "sharpe_ratio")
    assert hasattr(result, "max_drawdown")

def test_backtest_trades_recorded(sample_backtest_data):
    engine = BacktestEngine(initial_capital=10000)
    result = engine.run(sample_backtest_data)
    assert len(result.trades) > 0
    assert "entry_date" in result.trades.columns
    assert "exit_date" in result.trades.columns
    assert "pnl" in result.trades.columns

def test_backtest_equity_curve(sample_backtest_data):
    engine = BacktestEngine(initial_capital=10000)
    result = engine.run(sample_backtest_data)
    assert len(result.equity_curve) == len(sample_backtest_data)
    assert result.equity_curve.iloc[0] == 10000
