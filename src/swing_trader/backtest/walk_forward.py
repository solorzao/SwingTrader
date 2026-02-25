import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from swing_trader.models.base import BaseModel
from swing_trader.backtest.engine import BacktestEngine, BacktestResult
from swing_trader.config import WalkForwardConfig, BacktestConfig


@dataclass
class WindowResult:
    """Metrics for a single walk-forward window."""
    window_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_size: int
    test_size: int
    accuracy: float
    signals: pd.Series


@dataclass
class WalkForwardResult:
    """Container for walk-forward backtest results."""
    backtest_result: BacktestResult
    window_results: list[WindowResult]
    oos_predictions: pd.Series
    config: WalkForwardConfig

    @property
    def n_windows(self) -> int:
        return len(self.window_results)

    @property
    def per_window_accuracy(self) -> list[float]:
        return [w.accuracy for w in self.window_results]

    @property
    def mean_oos_accuracy(self) -> float:
        accs = self.per_window_accuracy
        return np.mean(accs) if accs else 0.0


ModelFactory = Callable[[pd.DataFrame, pd.Series], BaseModel]


class WalkForwardEngine:
    """Walk-forward backtesting with periodic retraining.

    For each window:
      1. Train model on [start..T]
      2. Predict on [T..T+step] (truly out-of-sample)
      3. Advance T by step_days
      4. Retrain if retrain_every steps reached

    Modes:
      - expanding: training window grows (more data over time)
      - rolling: fixed training window (adapts to regime changes)
    """

    def __init__(
        self,
        wf_config: WalkForwardConfig | None = None,
        bt_config: BacktestConfig | None = None,
    ):
        self.wf_config = wf_config or WalkForwardConfig()
        self.bt_config = bt_config or BacktestConfig()

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        prices: pd.Series,
        model_factory: ModelFactory,
    ) -> WalkForwardResult:
        """Run walk-forward backtest.

        Args:
            X: Feature DataFrame with DatetimeIndex
            y: Target labels
            prices: Close prices for backtest simulation
            model_factory: Callable(X_train, y_train) -> fitted BaseModel
        """
        cfg = self.wf_config
        n = len(X)
        step = cfg.step_days

        # Initial training window
        train_end_idx = cfg.train_window_days
        if train_end_idx >= n:
            raise ValueError(
                f"train_window_days ({cfg.train_window_days}) >= data length ({n}). "
                "Need more data or a smaller training window."
            )

        window_results = []
        all_signals = pd.Series(dtype=float, index=X.index)
        model = None
        steps_since_retrain = 0

        window_idx = 0
        while train_end_idx < n:
            test_end_idx = min(train_end_idx + step, n)

            # Determine training window
            if cfg.mode == "expanding":
                train_start_idx = 0
            else:  # rolling
                train_start_idx = max(0, train_end_idx - cfg.train_window_days)

            X_train = X.iloc[train_start_idx:train_end_idx]
            y_train = y.iloc[train_start_idx:train_end_idx]
            X_test = X.iloc[train_end_idx:test_end_idx]
            y_test = y.iloc[train_end_idx:test_end_idx]

            if len(X_train) < cfg.min_train_samples:
                train_end_idx += step
                continue

            if len(X_test) == 0:
                break

            # Retrain if needed
            if model is None or steps_since_retrain >= cfg.retrain_every:
                model = model_factory(X_train, y_train)
                steps_since_retrain = 0

            # Predict on test window (truly out-of-sample)
            try:
                preds = model.predict(X_test)

                # Handle LSTM shorter output
                if len(preds) < len(X_test):
                    pad = len(X_test) - len(preds)
                    preds = np.concatenate([[0] * pad, preds])

                signals = pd.Series(preds, index=X_test.index)
                all_signals.loc[X_test.index] = signals

                # Calculate accuracy for this window
                y_test_aligned = y_test.values[-len(preds):]
                accuracy = float((preds == y_test_aligned).mean())

            except Exception as e:
                print(f"Warning: Window {window_idx} prediction failed: {e}")
                signals = pd.Series(0, index=X_test.index)
                all_signals.loc[X_test.index] = 0
                accuracy = 0.0

            window_results.append(WindowResult(
                window_idx=window_idx,
                train_start=X_train.index[0],
                train_end=X_train.index[-1],
                test_start=X_test.index[0],
                test_end=X_test.index[-1],
                train_size=len(X_train),
                test_size=len(X_test),
                accuracy=accuracy,
                signals=signals,
            ))

            train_end_idx += step
            steps_since_retrain += 1
            window_idx += 1

        # Run backtest on all OOS predictions
        oos_predictions = all_signals.dropna()
        if len(oos_predictions) == 0:
            raise ValueError("Walk-forward produced no out-of-sample predictions")

        bt_data = pd.DataFrame({
            "close": prices.loc[oos_predictions.index],
            "signal": oos_predictions.astype(int),
        })

        engine = BacktestEngine(
            initial_capital=self.bt_config.initial_capital,
            position_size=self.bt_config.position_size,
            commission=self.bt_config.commission,
            slippage=self.bt_config.slippage,
        )
        backtest_result = engine.run(bt_data)

        return WalkForwardResult(
            backtest_result=backtest_result,
            window_results=window_results,
            oos_predictions=oos_predictions,
            config=self.wf_config,
        )
