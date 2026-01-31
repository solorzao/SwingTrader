import pandas as pd
import numpy as np
import xgboost as xgb
from .base import BaseModel, Signal

class XGBoostModel(BaseModel):
    """XGBoost classifier for trading signals with GPU support."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        use_gpu: bool = False,
        **kwargs
    ):
        super().__init__(name="XGBoost")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.use_gpu = use_gpu

        model_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "objective": "multi:softprob",
            "num_class": 3,
            **kwargs
        }

        if use_gpu:
            # XGBoost 2.0+ uses device="cuda" for GPU acceleration
            model_params["device"] = "cuda"

        self.model = xgb.XGBClassifier(**model_params)
        self._label_map = {-1: 0, 0: 1, 1: 2}
        self._label_map_inv = {0: -1, 1: 0, 2: 1}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs
    ) -> "XGBoostModel":
        """Train the XGBoost model."""
        X_prep = self._prepare_features(X)
        y_mapped = y.map(self._label_map)

        self.model.fit(X_prep, y_mapped, **kwargs)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict signals."""
        X_prep = X[self.feature_columns]
        preds = self.model.predict(X_prep)
        return np.array([self._label_map_inv[int(p)] for p in preds])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        X_prep = X[self.feature_columns]
        return self.model.predict_proba(X_prep)

    def feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_columns
        ).sort_values(ascending=False)
