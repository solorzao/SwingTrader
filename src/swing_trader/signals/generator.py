import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal

from swing_trader.data.fetcher import StockDataFetcher
from swing_trader.features.indicators import TechnicalIndicators
from swing_trader.models.base import BaseModel

class SignalGenerator:
    """Generate trading signals for stocks."""

    def __init__(
        self,
        models: dict[str, BaseModel] | None = None,
        model_dir: Path | None = None
    ):
        """
        Args:
            models: Dict of model_name -> fitted model
            model_dir: Directory to load saved models from
        """
        self.fetcher = StockDataFetcher()
        self.indicators = TechnicalIndicators()
        self.models = models or {}

        if model_dir and Path(model_dir).exists():
            self._load_models(Path(model_dir))

    def _load_models(self, model_dir: Path) -> None:
        """Load all models from directory."""
        from swing_trader.models import RandomForestModel, XGBoostModel, LSTMModel

        model_classes = {
            "rf": RandomForestModel,
            "xgb": XGBoostModel,
            "lstm": LSTMModel
        }

        for model_file in model_dir.glob("*.joblib"):
            name = model_file.stem
            model_type = name.split("_")[0]
            if model_type in model_classes:
                model = model_classes[model_type]()
                model.load(model_file)
                self.models[name] = model

    def generate(
        self,
        ticker: str,
        period: str = "6mo",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """
        Generate signals for a single ticker.

        Args:
            ticker: Stock symbol
            period: Data period (used if start/end not provided)
            start: Start date as YYYY-MM-DD (overrides period)
            end: End date as YYYY-MM-DD (overrides period)

        Returns DataFrame with columns:
            - date, open, high, low, close, volume
            - All technical indicators
            - signal: Consensus signal (-1, 0, 1)
            - confidence: Signal confidence (0-1)
            - model_signals: Individual model predictions
        """
        # Fetch data using date range if provided, otherwise period
        if start and end:
            df = self.fetcher.fetch_range(ticker, start=start, end=end)
        else:
            df = self.fetcher.fetch(ticker, period=period)

        # Add indicators
        df = self.indicators.add_all(df, dropna=True)

        return self._generate_for_data(df, ticker)

    def _generate_for_data(
        self,
        df: pd.DataFrame,
        ticker: str
    ) -> pd.DataFrame:
        """Generate signals for prepared data."""
        if not self.models:
            raise ValueError("No models loaded. Train or load models first.")

        result = df.copy()
        model_signals = {}
        model_probas = {}

        for name, model in self.models.items():
            if not model.is_fitted:
                continue

            try:
                signals = model.predict(df)
                probas = model.predict_proba(df)

                # Handle LSTM shorter output
                if len(signals) < len(df):
                    pad = len(df) - len(signals)
                    signals = np.concatenate([[np.nan] * pad, signals])
                    probas = np.vstack([np.full((pad, 3), np.nan), probas])

                model_signals[name] = signals
                model_probas[name] = probas

            except Exception as e:
                print(f"Warning: Model {name} failed: {e}")

        if not model_signals:
            raise ValueError("No models produced valid predictions")

        # Consensus signal (majority vote)
        signal_df = pd.DataFrame(model_signals)
        result["signal"] = signal_df.mode(axis=1).iloc[:, 0].fillna(0).astype(int)

        # Confidence (agreement among models)
        def calc_confidence(row):
            valid = row.dropna()
            if len(valid) == 0:
                return 0
            mode = valid.mode()
            if len(mode) == 0:
                return 0
            return (valid == mode.iloc[0]).sum() / len(valid)

        result["confidence"] = signal_df.apply(calc_confidence, axis=1)

        # Store individual model signals
        for name, signals in model_signals.items():
            result[f"signal_{name}"] = signals

        result["ticker"] = ticker

        return result

    def scan_universe(
        self,
        tickers: list[str],
        min_confidence: float = 0.6,
        signal_filter: Literal["buy", "sell", "all"] = "all"
    ) -> pd.DataFrame:
        """
        Scan multiple tickers and return latest signals.

        Returns DataFrame with one row per ticker showing latest signal.
        """
        results = []

        for ticker in tickers:
            try:
                df = self.generate(ticker)
                latest = df.iloc[-1].to_dict()
                latest["ticker"] = ticker
                results.append(latest)
            except Exception as e:
                print(f"Warning: Failed to process {ticker}: {e}")

        if not results:
            return pd.DataFrame()

        scan_df = pd.DataFrame(results)

        # Filter by confidence
        scan_df = scan_df[scan_df["confidence"] >= min_confidence]

        # Filter by signal type
        if signal_filter == "buy":
            scan_df = scan_df[scan_df["signal"] == 1]
        elif signal_filter == "sell":
            scan_df = scan_df[scan_df["signal"] == -1]

        return scan_df.sort_values("confidence", ascending=False)
