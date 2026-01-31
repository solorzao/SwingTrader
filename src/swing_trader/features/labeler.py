import pandas as pd
import numpy as np

class SignalLabeler:
    """Generate target labels for supervised learning."""

    def __init__(self, forward_days: int = 5, threshold: float = 0.02):
        """
        Args:
            forward_days: Number of days to look ahead for return calculation
            threshold: Minimum return threshold for BUY/SELL signal (e.g., 0.02 = 2%)
        """
        self.forward_days = forward_days
        self.threshold = threshold

    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Create signal labels based on forward returns.

        BUY (1): forward return > threshold
        SELL (-1): forward return < -threshold
        HOLD (0): otherwise
        """
        forward_return = df["close"].shift(-self.forward_days) / df["close"] - 1

        labels = pd.Series(0, index=df.index, dtype=int)
        labels[forward_return > self.threshold] = 1
        labels[forward_return < -self.threshold] = -1
        labels.iloc[-self.forward_days:] = np.nan

        return labels
