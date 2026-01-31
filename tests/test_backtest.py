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


def test_backtest_mark_to_market_equity():
    """Test that equity curve reflects unrealized P&L during open positions."""
    # Create simple test data with predictable price movement
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    # Price rises steadily from 100 to 109
    close = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], dtype=float)
    # Buy on day 1, hold until end (exit forced on last day)
    signals = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    df = pd.DataFrame({"close": close, "signal": signals}, index=dates)

    engine = BacktestEngine(initial_capital=10000, position_size=1.0, commission=0, slippage=0)
    result = engine.run(df)

    # Verify equity curve changes during the open position (not flat staircase)
    # After entry at day 1 (price ~101), equity should increase each day as price rises
    equity_changes = result.equity_curve.diff().dropna()

    # During holding period (days 2-8), equity should change with price
    holding_period_changes = equity_changes.iloc[1:8]  # Days after entry, before forced exit

    # At least some days should show equity changes (mark-to-market)
    assert (holding_period_changes != 0).sum() > 0, "Equity should change during open position (mark-to-market)"

    # Since price is rising and we're long, most equity changes should be positive
    assert (holding_period_changes > 0).sum() >= (holding_period_changes < 0).sum(), \
        "Long position with rising prices should show positive equity changes"
