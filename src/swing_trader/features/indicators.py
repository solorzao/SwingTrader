import pandas as pd
import numpy as np
from typing import Optional


class TechnicalIndicators:
    """Calculate technical indicators for swing trading."""

    # Default configuration
    DEFAULT_CONFIG = {
        "features": {
            "sma": True,
            "ema": True,
            "rsi": True,
            "macd": True,
            "bollinger": True,
            "atr": True,
            "obv": True,
            "stochastic": True,
        },
        "params": {
            "rsi_period": 14,
            "sma_periods": [10, 20, 50],
            "ema_periods": [12, 26],
            "atr_period": 14,
            "bb_period": 20,
            "stoch_period": 14,
        }
    }

    def __init__(self, config: Optional[dict] = None):
        """Initialize with optional custom configuration."""
        self.config = config or self.DEFAULT_CONFIG

    def add_all(self, df: pd.DataFrame, dropna: bool = True, config: Optional[dict] = None) -> pd.DataFrame:
        """Add technical indicators based on configuration."""
        df = df.copy()
        cfg = config or self.config
        features = cfg.get("features", self.DEFAULT_CONFIG["features"])
        params = cfg.get("params", self.DEFAULT_CONFIG["params"])

        # Always need sma_20 for some calculations
        sma_periods = params.get("sma_periods", [10, 20, 50])
        if 20 not in sma_periods:
            sma_periods = [20] + sma_periods

        if features.get("sma", True):
            df = self._add_sma(df, sma_periods)

        if features.get("ema", True):
            ema_periods = params.get("ema_periods", [12, 26])
            df = self._add_ema(df, ema_periods)

        if features.get("rsi", True):
            rsi_period = params.get("rsi_period", 14)
            df = self._add_rsi(df, rsi_period)

        if features.get("macd", True):
            df = self._add_macd(df)

        if features.get("bollinger", True):
            df = self._add_bollinger(df)

        if features.get("atr", True):
            atr_period = params.get("atr_period", 14)
            df = self._add_atr(df, atr_period)

        if features.get("obv", True):
            df = self._add_obv(df)

        if features.get("stochastic", True):
            stoch_period = params.get("stoch_period", 14)
            df = self._add_stochastic(df, stoch_period)

        # Add volume features
        df = self._add_volume(df)

        # Add rate of change
        df["roc_10"] = df["close"].pct_change(10) * 100

        if dropna:
            df = df.dropna()
        return df

    def _add_sma(self, df: pd.DataFrame, periods: list) -> pd.DataFrame:
        """Add Simple Moving Averages."""
        for period in periods:
            df[f"sma_{period}"] = df["close"].rolling(period).mean()
        return df

    def _add_ema(self, df: pd.DataFrame, periods: list) -> pd.DataFrame:
        """Add Exponential Moving Averages."""
        for period in periods:
            df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
        return df

    def _add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add RSI indicator."""
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
        return df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD indicators."""
        # Ensure EMAs exist
        if "ema_12" not in df.columns:
            df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
        if "ema_26" not in df.columns:
            df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        return df

    def _add_bollinger(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Bollinger Bands."""
        sma_col = f"sma_{period}"
        if sma_col not in df.columns:
            df[sma_col] = df["close"].rolling(period).mean()

        df["bb_mid"] = df[sma_col]
        bb_std = df["close"].rolling(period).std()
        df["bb_upper"] = df["bb_mid"] + (2 * bb_std)
        df["bb_lower"] = df["bb_mid"] - (2 * bb_std)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        return df

    def _add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range."""
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift())
        low_close = abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f"atr_{period}"] = tr.rolling(period).mean()
        return df

    def _add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add On-Balance Volume."""
        obv = [0]
        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i-1]:
                obv.append(obv[-1] + df["volume"].iloc[i])
            elif df["close"].iloc[i] < df["close"].iloc[i-1]:
                obv.append(obv[-1] - df["volume"].iloc[i])
            else:
                obv.append(obv[-1])
        df["obv"] = obv
        return df

    def _add_stochastic(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Stochastic Oscillator."""
        low_min = df["low"].rolling(period).min()
        high_max = df["high"].rolling(period).max()
        df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()
        return df

    def _add_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators."""
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
        return df

    def _add_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators."""
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

        # On-Balance Volume
        obv = [0]
        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i-1]:
                obv.append(obv[-1] + df["volume"].iloc[i])
            elif df["close"].iloc[i] < df["close"].iloc[i-1]:
                obv.append(obv[-1] - df["volume"].iloc[i])
            else:
                obv.append(obv[-1])
        df["obv"] = obv

        return df

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators: ATR, Bollinger Bands."""
        # ATR
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift())
        low_close = abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()

        # Bollinger Bands
        df["bb_mid"] = df["sma_20"]
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + (2 * bb_std)
        df["bb_lower"] = df["bb_mid"] - (2 * bb_std)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

        return df

    @staticmethod
    def infer_config_from_features(feature_columns: list[str]) -> dict:
        """Infer indicator configuration from feature column names.

        This allows reconstructing the config needed to generate features
        that match what a model was trained on, even if feature_config
        wasn't saved with the model.
        """
        import re

        config = {
            "features": {
                "sma": False,
                "ema": False,
                "rsi": False,
                "macd": False,
                "bollinger": False,
                "atr": False,
                "obv": False,
                "stochastic": False,
            },
            "params": {
                "rsi_period": 14,
                "sma_periods": [],
                "ema_periods": [],
                "atr_period": 14,
                "bb_period": 20,
                "stoch_period": 14,
            }
        }

        for col in feature_columns:
            # SMA: sma_10, sma_20, sma_50, etc.
            if match := re.match(r'sma_(\d+)', col):
                config["features"]["sma"] = True
                period = int(match.group(1))
                if period not in config["params"]["sma_periods"]:
                    config["params"]["sma_periods"].append(period)

            # EMA: ema_12, ema_26, etc.
            elif match := re.match(r'ema_(\d+)', col):
                config["features"]["ema"] = True
                period = int(match.group(1))
                if period not in config["params"]["ema_periods"]:
                    config["params"]["ema_periods"].append(period)

            # RSI: rsi_14, rsi_11, etc.
            elif match := re.match(r'rsi_(\d+)', col):
                config["features"]["rsi"] = True
                config["params"]["rsi_period"] = int(match.group(1))

            # MACD
            elif col in ['macd', 'macd_signal', 'macd_hist']:
                config["features"]["macd"] = True

            # Bollinger Bands
            elif col in ['bb_upper', 'bb_lower', 'bb_mid', 'bb_width']:
                config["features"]["bollinger"] = True

            # ATR: atr_14, etc.
            elif match := re.match(r'atr_(\d+)', col):
                config["features"]["atr"] = True
                config["params"]["atr_period"] = int(match.group(1))

            # OBV
            elif col == 'obv':
                config["features"]["obv"] = True

            # Stochastic
            elif col in ['stoch_k', 'stoch_d']:
                config["features"]["stochastic"] = True

        # Sort periods for consistency
        config["params"]["sma_periods"].sort()
        config["params"]["ema_periods"].sort()

        return config
