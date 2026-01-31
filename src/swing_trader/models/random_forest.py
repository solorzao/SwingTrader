import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .base import BaseModel, Signal

class RandomForestModel(BaseModel):
    """Random Forest classifier for trading signals."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 10,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(name="RandomForest")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
            **kwargs
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs
    ) -> "RandomForestModel":
        """Train the Random Forest model."""
        X_prep = self._prepare_features(X)
        self.model.fit(X_prep, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict signals."""
        X_prep = X[self.feature_columns]
        return self.model.predict(X_prep)

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
