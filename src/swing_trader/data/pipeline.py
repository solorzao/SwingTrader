import pandas as pd
from pathlib import Path

from swing_trader.data.fetcher import StockDataFetcher
from swing_trader.data.cache import DataCache
from swing_trader.data.splits import TemporalSplitManager, TemporalSplit
from swing_trader.features.indicators import TechnicalIndicators
from swing_trader.features.labeler import SignalLabeler
from swing_trader.config import FeatureConfig, LabelConfig, SplitConfig


# Columns to exclude from features
EXCLUDE_COLS = frozenset(["target", "open", "high", "low", "close", "volume"])


class DataPipeline:
    """Unified data pipeline: fetch -> cache -> indicators -> labels -> split.

    Single entry point that replaces the 3 duplicated pipelines in cli.py,
    ui/app.py, and ui/views/training.py.
    """

    def __init__(
        self,
        feature_config: FeatureConfig | None = None,
        label_config: LabelConfig | None = None,
        split_config: SplitConfig | None = None,
        cache_dir: Path | None = None,
    ):
        self.feature_config = feature_config or FeatureConfig()
        self.label_config = label_config or LabelConfig()
        self.split_config = split_config or SplitConfig()

        self.fetcher = StockDataFetcher()
        self.indicators = TechnicalIndicators()
        self.labeler = SignalLabeler(
            forward_days=self.label_config.forward_days,
            threshold=self.label_config.threshold,
        )
        self.cache = DataCache(cache_dir=cache_dir) if cache_dir else None

    def fetch(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """Fetch OHLCV data with optional caching."""
        if self.cache:
            cached = self.cache.get(ticker, period)
            if cached is not None:
                return cached

        df = self.fetcher.fetch(ticker, period=period)

        if self.cache:
            self.cache.put(ticker, period, df)

        return df

    def prepare(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """Fetch data, add indicators and labels, drop NaNs.

        Returns DataFrame with all features, OHLCV, and 'target' column.
        """
        df = self.fetch(ticker, period)
        indicator_config = self.feature_config.to_indicator_config()
        df = self.indicators.add_all(df, dropna=False, config=indicator_config)
        df["target"] = self.labeler.create_labels(df)
        df = df.dropna()
        return df

    def prepare_multiple(
        self,
        tickers: list[str],
        period: str = "2y",
    ) -> pd.DataFrame:
        """Prepare data for multiple tickers and concatenate.

        Returns combined DataFrame preserving chronological order within
        each ticker's data.
        """
        all_data = []
        for ticker in tickers:
            try:
                df = self.prepare(ticker, period)
                all_data.append(df)
            except Exception as e:
                print(f"Warning: Failed to prepare {ticker}: {e}")

        if not all_data:
            raise ValueError("No data collected for any ticker")

        return pd.concat(all_data)

    def get_features_and_target(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Extract feature columns (X) and target (y) from prepared DataFrame."""
        feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
        X = df[feature_cols]
        y = df["target"]
        return X, y

    def prepare_and_split(
        self,
        tickers: list[str],
        period: str = "2y",
    ) -> TemporalSplit:
        """Full pipeline: fetch -> features -> labels -> split.

        This is the primary entry point for training.
        """
        combined = self.prepare_multiple(tickers, period)
        X, y = self.get_features_and_target(combined)

        splitter = TemporalSplitManager(
            train_ratio=self.split_config.train,
            val_ratio=self.split_config.val,
            test_ratio=self.split_config.test,
            holdout_ratio=self.split_config.holdout,
        )
        return splitter.split(X, y)
