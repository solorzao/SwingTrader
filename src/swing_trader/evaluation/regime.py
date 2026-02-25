import pandas as pd
import numpy as np
from dataclasses import dataclass

from swing_trader.backtest.engine import BacktestEngine, BacktestResult


@dataclass
class RegimeReport:
    """Per-regime performance breakdown."""
    regime_labels: pd.Series  # "high_vol" or "low_vol" per date
    regime_metrics: dict[str, BacktestResult]  # regime_name -> BacktestResult
    volatility: pd.Series  # rolling volatility series
    threshold: float  # volatility threshold used for regime split


class RegimeAnalyzer:
    """Detect market regimes via rolling volatility and compute per-regime metrics."""

    def __init__(self, vol_window: int = 60, vol_quantile: float = 0.5):
        """
        Args:
            vol_window: Rolling window for volatility calculation (trading days)
            vol_quantile: Quantile threshold for high/low vol regime split
        """
        self.vol_window = vol_window
        self.vol_quantile = vol_quantile

    def detect_regimes(self, prices: pd.Series) -> tuple[pd.Series, pd.Series, float]:
        """Detect high/low volatility regimes.

        Returns:
            (regime_labels, rolling_volatility, threshold)
        """
        daily_returns = prices.pct_change().dropna()
        vol = daily_returns.rolling(self.vol_window).std().dropna() * np.sqrt(252)

        threshold = float(vol.quantile(self.vol_quantile))

        labels = pd.Series("low_vol", index=vol.index)
        labels[vol > threshold] = "high_vol"

        return labels, vol, threshold

    def analyze(
        self,
        prices: pd.Series,
        signals: pd.Series,
        initial_capital: float = 10_000,
        commission: float = 0.001,
        slippage: float = 0.0005,
    ) -> RegimeReport:
        """Run per-regime backtest analysis.

        Args:
            prices: Close price series
            signals: Trading signal series (-1, 0, 1)
            initial_capital: Starting capital
            commission: Trading commission
            slippage: Slippage per trade
        """
        regime_labels, volatility, threshold = self.detect_regimes(prices)

        # Align signals and prices to regime labels index
        common_idx = regime_labels.index.intersection(signals.index).intersection(prices.index)
        regime_labels = regime_labels.loc[common_idx]
        signals_aligned = signals.loc[common_idx]
        prices_aligned = prices.loc[common_idx]

        regime_metrics = {}
        for regime_name in ["low_vol", "high_vol"]:
            mask = regime_labels == regime_name
            if mask.sum() < 5:
                continue

            regime_prices = prices_aligned[mask]
            regime_signals = signals_aligned[mask]

            bt_data = pd.DataFrame({
                "close": regime_prices,
                "signal": regime_signals.astype(int),
            })

            engine = BacktestEngine(
                initial_capital=initial_capital,
                commission=commission,
                slippage=slippage,
            )
            try:
                regime_metrics[regime_name] = engine.run(bt_data)
            except Exception:
                pass

        return RegimeReport(
            regime_labels=regime_labels,
            regime_metrics=regime_metrics,
            volatility=volatility,
            threshold=threshold,
        )
