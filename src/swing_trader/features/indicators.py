import pandas as pd
import numpy as np

class TechnicalIndicators:
    """Calculate technical indicators for swing trading."""

    def add_all(self, df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
        """Add all technical indicators to OHLCV dataframe."""
        df = df.copy()
        df = self._add_moving_averages(df)
        df = self._add_momentum(df)
        df = self._add_volume(df)
        df = self._add_volatility(df)

        if dropna:
            df = df.dropna()
        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add SMA and EMA indicators."""
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
        return df

    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators: RSI, MACD, ROC."""
        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Rate of Change
        df["roc_10"] = df["close"].pct_change(10) * 100

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
