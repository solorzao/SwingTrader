import pandas as pd
import numpy as np
from typing import Literal
from .base import BaseModel, Signal

class EnsembleModel:
    """Combine multiple models for ensemble predictions."""

    def __init__(
        self,
        models: list[BaseModel],
        method: Literal["voting", "weighted"] = "voting",
        weights: list[float] | None = None
    ):
        """
        Args:
            models: List of fitted BaseModel instances
            method: Ensemble method ("voting" or "weighted")
            weights: Weights for each model (required for "weighted")
        """
        self.models = models
        self.method = method

        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions."""
        if self.method == "voting":
            return self._voting_predict(X)
        elif self.method == "weighted":
            return self._weighted_predict(X)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _voting_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Majority voting across models."""
        all_preds = np.array([m.predict(X) for m in self.models])
        weighted_votes = np.zeros((len(X), 3))
        for i, preds in enumerate(all_preds):
            for j, p in enumerate(preds):
                class_idx = int(p) + 1  # Map -1,0,1 to 0,1,2
                weighted_votes[j, class_idx] += self.weights[i]
        return np.argmax(weighted_votes, axis=1) - 1  # Map back to -1,0,1

    def _weighted_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted average of probabilities."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1) - 1  # Map 0,1,2 to -1,0,1

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted average of class probabilities."""
        all_proba = np.array([m.predict_proba(X) for m in self.models])
        weighted_proba = np.zeros_like(all_proba[0])
        for i, proba in enumerate(all_proba):
            weighted_proba += self.weights[i] * proba
        return weighted_proba

    def get_model_agreement(self, X: pd.DataFrame) -> pd.DataFrame:
        """Show how models agree/disagree."""
        preds = {m.name: m.predict(X) for m in self.models}
        df = pd.DataFrame(preds)
        df["ensemble"] = self.predict(X)
        df["agreement"] = (df.iloc[:, :-1].nunique(axis=1) == 1).astype(int)
        return df
