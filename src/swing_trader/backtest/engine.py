import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class BacktestResult:
    """Container for backtest results."""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    equity_curve: pd.Series
    trades: pd.DataFrame
    daily_returns: pd.Series

    def summary(self) -> str:
        return f"""
Backtest Results
================
Total Return:    {self.total_return:.2%}
Annual Return:   {self.annual_return:.2%}
Sharpe Ratio:    {self.sharpe_ratio:.2f}
Max Drawdown:    {self.max_drawdown:.2%}
Win Rate:        {self.win_rate:.2%}
Profit Factor:   {self.profit_factor:.2f}
Total Trades:    {self.total_trades}
"""


class BacktestEngine:
    """Walk-forward backtesting engine."""

    def __init__(
        self,
        initial_capital: float = 10000,
        position_size: float = 1.0,
        commission: float = 0.001,
        slippage: float = 0.0005
    ):
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.commission = commission
        self.slippage = slippage

    def run(
        self,
        data: pd.DataFrame,
        signal_col: str = "signal",
        price_col: str = "close"
    ) -> BacktestResult:
        df = data.copy()
        signals = df[signal_col].values
        prices = df[price_col].values
        dates = df.index

        capital = self.initial_capital
        position = 0
        entry_price = 0
        entry_date = None

        equity = [capital]
        trades = []

        for i in range(1, len(df)):
            signal = signals[i]
            price = prices[i]

            if position != 0:
                if (position == 1 and signal == -1) or \
                   (position == -1 and signal == 1) or \
                   i == len(df) - 1:

                    exit_price = price * (1 - self.slippage if position == 1 else 1 + self.slippage)

                    if position == 1:
                        pnl = (exit_price / entry_price - 1) * capital * self.position_size
                    else:
                        pnl = (entry_price / exit_price - 1) * capital * self.position_size

                    pnl -= capital * self.position_size * self.commission * 2
                    capital += pnl

                    trades.append({
                        "entry_date": entry_date,
                        "exit_date": dates[i],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "position": "long" if position == 1 else "short",
                        "pnl": pnl,
                        "return": pnl / (self.initial_capital * self.position_size)
                    })
                    position = 0

            if position == 0 and signal != 0:
                position = signal
                entry_price = price * (1 + self.slippage if signal == 1 else 1 - self.slippage)
                entry_date = dates[i]

            equity.append(capital)

        equity_curve = pd.Series(equity, index=dates)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
            columns=["entry_date", "exit_date", "entry_price", "exit_price", "position", "pnl", "return"]
        )

        return self._calculate_metrics(equity_curve, trades_df)

    def _calculate_metrics(self, equity_curve: pd.Series, trades: pd.DataFrame) -> BacktestResult:
        daily_returns = equity_curve.pct_change().dropna()
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        n_days = len(equity_curve)
        annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0

        if daily_returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        else:
            sharpe_ratio = 0

        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        if len(trades) > 0:
            wins = trades[trades["pnl"] > 0]
            losses = trades[trades["pnl"] <= 0]
            win_rate = len(wins) / len(trades)
            total_profit = wins["pnl"].sum() if len(wins) > 0 else 0
            total_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 1
            profit_factor = total_profit / total_loss if total_loss > 0 else 0
        else:
            win_rate = 0
            profit_factor = 0

        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            equity_curve=equity_curve,
            trades=trades,
            daily_returns=daily_returns
        )
