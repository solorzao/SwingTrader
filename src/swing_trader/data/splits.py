import pandas as pd
from dataclasses import dataclass


@dataclass
class TemporalSplit:
    """Container for chronologically split data."""
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    X_holdout: pd.DataFrame
    y_holdout: pd.Series
    train_end_date: pd.Timestamp
    val_end_date: pd.Timestamp
    test_end_date: pd.Timestamp
    holdout_start_date: pd.Timestamp

    @property
    def train_size(self) -> int:
        return len(self.X_train)

    @property
    def val_size(self) -> int:
        return len(self.X_val)

    @property
    def test_size(self) -> int:
        return len(self.X_test)

    @property
    def holdout_size(self) -> int:
        return len(self.X_holdout)


class TemporalSplitManager:
    """Enforces chronological train/val/test/holdout splits for time-series data.

    NEVER shuffles data. All splits are strictly chronological.
    """

    def __init__(
        self,
        train_ratio: float = 0.6,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        holdout_ratio: float = 0.2,
    ):
        total = train_ratio + val_ratio + test_ratio + holdout_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.holdout_ratio = holdout_ratio

    def split(self, X: pd.DataFrame, y: pd.Series) -> TemporalSplit:
        """Split data chronologically into train/val/test/holdout.

        Data MUST be sorted by date index. No shuffling occurs.

        Args:
            X: Feature DataFrame (should have DatetimeIndex)
            y: Target Series

        Returns:
            TemporalSplit with all four splits and boundary dates
        """
        n = len(X)
        if n < 20:
            raise ValueError(f"Need at least 20 samples for splitting, got {n}")

        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)
        test_end = val_end + int(n * self.test_ratio)

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_val = X.iloc[train_end:val_end]
        y_val = y.iloc[train_end:val_end]
        X_test = X.iloc[val_end:test_end]
        y_test = y.iloc[val_end:test_end]
        X_holdout = X.iloc[test_end:]
        y_holdout = y.iloc[test_end:]

        def _safe_timestamp(idx, pos):
            try:
                return pd.Timestamp(idx[pos])
            except Exception:
                return pd.Timestamp.now()

        train_end_date = _safe_timestamp(X_train.index, -1)
        val_end_date = _safe_timestamp(X_val.index, -1) if len(X_val) > 0 else train_end_date
        test_end_date = _safe_timestamp(X_test.index, -1) if len(X_test) > 0 else val_end_date
        holdout_start_date = _safe_timestamp(X_holdout.index, 0) if len(X_holdout) > 0 else test_end_date

        return TemporalSplit(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            X_holdout=X_holdout, y_holdout=y_holdout,
            train_end_date=train_end_date,
            val_end_date=val_end_date,
            test_end_date=test_end_date,
            holdout_start_date=holdout_start_date,
        )

    def split_train_test(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_ratio: float = 0.2,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Timestamp]:
        """Simple 2-way chronological split for backward compatibility.

        Returns (X_train, y_train, X_test, y_test, train_end_date)
        """
        n = len(X)
        split_idx = int(n * (1 - test_ratio))

        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]

        try:
            train_end_date = pd.Timestamp(X_train.index[-1])
        except Exception:
            train_end_date = pd.Timestamp.now()

        return X_train, y_train, X_test, y_test, train_end_date
