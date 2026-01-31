import optuna
from optuna.integration import MLflowCallback
import pandas as pd
import numpy as np
from typing import Callable
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

from swing_trader.models import RandomForestModel, XGBoostModel, LSTMModel
from swing_trader.training.tracker import ExperimentTracker

class HyperparameterTuner:
    """Optuna-based hyperparameter optimization with MLflow logging."""

    def __init__(self, tracker: ExperimentTracker | None = None):
        self.tracker = tracker
        self.study = None
        self.best_params = None

    def tune_random_forest(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
        cv_splits: int = 5,
        timeout: int | None = None
    ) -> dict:
        """Tune RandomForest hyperparameters."""

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            }

            model = RandomForestModel(**params)
            tscv = TimeSeriesSplit(n_splits=cv_splits)

            scores = cross_val_score(
                model.model, X, y, cv=tscv, scoring="accuracy", n_jobs=-1
            )
            return scores.mean()

        return self._run_study("rf_tuning", objective, n_trials, timeout, "maximize")

    def tune_xgboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
        cv_splits: int = 5,
        timeout: int | None = None,
        use_gpu: bool = False
    ) -> dict:
        """Tune XGBoost hyperparameters."""

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }

            model = XGBoostModel(use_gpu=use_gpu, **params)
            y_mapped = y.map({-1: 0, 0: 1, 1: 2})
            tscv = TimeSeriesSplit(n_splits=cv_splits)

            scores = cross_val_score(
                model.model, X, y_mapped, cv=tscv, scoring="accuracy",
                n_jobs=1 if use_gpu else -1
            )
            return scores.mean()

        return self._run_study("xgb_tuning", objective, n_trials, timeout, "maximize")

    def tune_lstm(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 30,
        timeout: int | None = None
    ) -> dict:
        """Tune LSTM hyperparameters."""

        def objective(trial):
            params = {
                "sequence_length": trial.suggest_int("sequence_length", 10, 50),
                "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128, 256]),
                "num_layers": trial.suggest_int("num_layers", 1, 3),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
                "epochs": 30,  # Fixed for tuning speed
            }

            # Simple train/val split for LSTM
            split = int(len(X) * 0.8)
            X_train, X_val = X.iloc[:split], X.iloc[split:]
            y_train, y_val = y.iloc[:split], y.iloc[split:]

            model = LSTMModel(**params)
            model.fit(X_train, y_train)

            preds = model.predict(X_val)
            y_val_aligned = y_val.iloc[model.sequence_length - 1:].values
            accuracy = (preds == y_val_aligned).mean()

            return accuracy

        return self._run_study("lstm_tuning", objective, n_trials, timeout, "maximize")

    def _run_study(
        self,
        study_name: str,
        objective: Callable,
        n_trials: int,
        timeout: int | None,
        direction: str
    ) -> dict:
        """Run Optuna study with optional MLflow callback."""
        from datetime import datetime

        callbacks = []
        if self.tracker:
            callbacks.append(MLflowCallback(
                tracking_uri=self.tracker.tracking_uri,
                metric_name="cv_accuracy"
            ))

        # Use unique study name with timestamp to avoid reusing cached results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_study_name = f"{study_name}_{timestamp}"

        self.study = optuna.create_study(
            study_name=unique_study_name,
            direction=direction,
        )

        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=callbacks,
            show_progress_bar=True
        )

        self.best_params = self.study.best_params
        return {
            "best_params": self.best_params,
            "best_value": self.study.best_value,
            "n_trials": len(self.study.trials)
        }

    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame for plotting."""
        if self.study is None:
            return pd.DataFrame()

        return self.study.trials_dataframe()
