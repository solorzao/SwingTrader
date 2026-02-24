import optuna
from optuna.integration import MLflowCallback
import pandas as pd
import numpy as np
import gc
from typing import Callable
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

from swing_trader.models import RandomForestModel, XGBoostModel, LSTMModel
from swing_trader.training.tracker import ExperimentTracker


def _cleanup_gpu_memory():
    """Force GPU memory cleanup between trials to prevent memory exhaustion."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass

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
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            }

            model = RandomForestModel(**params)
            tscv = TimeSeriesSplit(n_splits=cv_splits)

            # Use n_jobs=1 to avoid joblib parallel worker issues in Qt threads
            scores = cross_val_score(
                model.model, X, y, cv=tscv, scoring="accuracy", n_jobs=1
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
        """Tune XGBoost hyperparameters with early stopping to prevent hangs.

        Note: Tuning always uses CPU regardless of use_gpu parameter.
        GPU is problematic for repeated model creation in Optuna loops.
        The final model trained after tuning will use GPU if available.
        """
        y_mapped = y.map({-1: 0, 0: 1, 1: 2})

        # Temporal split: use chronological order (not random) for time-series data
        split = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y_mapped.iloc[:split], y_mapped.iloc[split:]

        def objective(trial):
            # Tune early_stopping_rounds as a hyperparameter (also prevents hanging)
            early_stopping_rounds = trial.suggest_int("early_stopping_rounds", 5, 50)

            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }

            # IMPORTANT: Always use CPU for tuning - GPU causes hangs in Optuna loops
            # due to CUDA context and memory management issues with repeated model creation.
            # Final training after tuning will use GPU if available.
            model = XGBoostModel(use_gpu=False, **params)

            try:
                # Fit with eval_set AND early_stopping_rounds passed to fit()
                model.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False
                )

                preds = model.model.predict(X_val)
                accuracy = (preds == y_val).mean()
                return accuracy
            finally:
                del model
                gc.collect()

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
                "sequence_length": trial.suggest_int("sequence_length", 5, 60),
                "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128, 256]),
                "num_layers": trial.suggest_int("num_layers", 1, 3),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
                "epochs": 20,  # Slightly more epochs for tuning
            }

            # Simple train/val split for LSTM
            split = int(len(X) * 0.8)
            X_train, X_val = X.iloc[:split], X.iloc[split:]
            y_train, y_val = y.iloc[:split], y.iloc[split:]

            model = LSTMModel(**params)
            try:
                model.fit(X_train, y_train)

                preds = model.predict(X_val)
                y_val_aligned = y_val.iloc[model.sequence_length - 1:].values
                accuracy = (preds == y_val_aligned).mean()

                return accuracy
            finally:
                # Critical: Free GPU memory after each trial to prevent exhaustion
                del model
                _cleanup_gpu_memory()

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

        try:
            # show_progress_bar=False to avoid tqdm deadlocks in Qt worker threads
            self.study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                callbacks=callbacks,
                show_progress_bar=False
            )

            self.best_params = self.study.best_params
            return {
                "best_params": self.best_params,
                "best_value": self.study.best_value,
                "n_trials": len(self.study.trials)
            }
        finally:
            # Final cleanup after all trials complete
            _cleanup_gpu_memory()

    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame for plotting."""
        if self.study is None:
            return pd.DataFrame()

        return self.study.trials_dataframe()
