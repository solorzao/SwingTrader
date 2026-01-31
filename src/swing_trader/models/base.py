from abc import ABC, abstractmethod
from enum import IntEnum
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

class Signal(IntEnum):
    """Trading signal."""
    SELL = -1
    HOLD = 0
    BUY = 1

class BaseModel(ABC):
    """Abstract base class for all trading models."""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.feature_columns: list[str] = []
        self.is_fitted = False
        self.metrics: dict = {}  # Store training metrics

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs
    ) -> "BaseModel":
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict signals (BUY=1, HOLD=0, SELL=-1)."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for each class."""
        pass

    def save(self, path: Path, metrics: dict = None) -> None:
        """Save model to disk with optional metrics."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if metrics:
            self.metrics = metrics
        joblib.dump({
            "model": self.model,
            "feature_columns": self.feature_columns,
            "name": self.name,
            "metrics": self.metrics
        }, path)

    def load(self, path: Path) -> "BaseModel":
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_columns = data["feature_columns"]
        self.name = data["name"]
        self.metrics = data.get("metrics", {})
        self.is_fitted = True
        return self

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and validate feature columns."""
        if not self.feature_columns:
            exclude = ["signal", "target", "returns", "close", "open", "high", "low", "volume"]
            self.feature_columns = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c not in exclude
            ]
        return df[self.feature_columns]
