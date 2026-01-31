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
        cv_splits: int = 3,
        timeout: int | None = 180  # 3 minute default timeout
    ) -> dict:
        """Tune RandomForest hyperparameters."""

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
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
        cv_splits: int = 3,
        timeout: int | None = 180,  # 3 minute default timeout
        use_gpu: bool = False
    ) -> dict:
        """Tune XGBoost hyperparameters."""
        from sklearn.model_selection import train_test_split

        # For GPU, use simple train/val split (much faster than CV)
        # For CPU, use CV
        y_mapped = y.map({-1: 0, 0: 1, 1: 2})

        if use_gpu:
            # Simple 80/20 split for GPU (CV is too slow with GPU)
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_mapped, test_size=0.2, random_state=42
            )

            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                }

                model = XGBoostModel(use_gpu=True, **params)
                model.model.fit(X_train, y_train)
                preds = model.model.predict(X_val)
                accuracy = (preds == y_val).mean()
                return accuracy
        else:
            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                }

                model = XGBoostModel(use_gpu=False, **params)
                tscv = TimeSeriesSplit(n_splits=cv_splits)

                scores = cross_val_score(
                    model.model, X, y_mapped, cv=tscv, scoring="accuracy", n_jobs=-1
                )
                return scores.mean()

        return self._run_study("xgb_tuning", objective, n_trials, timeout, "maximize")

    def tune_lstm(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 20,
        timeout: int | None = 300  # 5 minute default timeout (LSTM is slower)
    ) -> dict:
        """Tune LSTM hyperparameters."""

        def objective(trial):
            params = {
                "sequence_length": trial.suggest_int("sequence_length", 10, 30),
                "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
                "num_layers": trial.suggest_int("num_layers", 1, 2),
                "dropout": trial.suggest_float("dropout", 0.1, 0.4),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
                "epochs": 15,  # Reduced for faster tuning
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
