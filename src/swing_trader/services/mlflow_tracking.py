"""MLflow experiment tracking for model training."""

import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
from typing import Any
import subprocess
import sys
import os


class MLflowTracker:
    """Handles MLflow experiment tracking and model registry."""

    def __init__(self, tracking_uri: str = None, experiment_name: str = "swing-trader"):
        # Use local mlruns directory by default
        if tracking_uri is None:
            tracking_uri = Path("mlruns").absolute().as_uri()

        mlflow.set_tracking_uri(tracking_uri)
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = experiment.experiment_id

        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()

    def start_run(self, run_name: str = None) -> str:
        """Start a new MLflow run."""
        run = mlflow.start_run(run_name=run_name)
        return run.info.run_id

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics."""
        for key, value in metrics.items():
            if value is not None:
                mlflow.log_metric(key, value)

    def log_model(self, model: Any, model_name: str, model_type: str) -> str:
        """Log model to MLflow and register it."""
        # Log the model
        if model_type == "LSTM":
            # PyTorch model
            mlflow.pytorch.log_model(model.model, "model")
        else:
            # Sklearn-compatible model
            mlflow.sklearn.log_model(model.model, "model")

        # Also save our custom wrapper
        artifact_path = Path("artifacts")
        artifact_path.mkdir(exist_ok=True)
        model_path = artifact_path / f"{model_name}.joblib"
        model.save(model_path)
        mlflow.log_artifact(str(model_path))

        return mlflow.active_run().info.run_id

    def log_feature_config(self, feature_config: dict) -> None:
        """Log feature configuration used for training."""
        mlflow.log_dict(feature_config, "feature_config.json")

    def log_figure(self, fig, filename: str) -> None:
        """Log a matplotlib figure."""
        mlflow.log_figure(fig, filename)

    def end_run(self) -> None:
        """End the current run."""
        mlflow.end_run()

    def get_best_run(self, metric: str = "f1_score") -> dict | None:
        """Get the best run by a metric."""
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1
        )
        if len(runs) > 0:
            return runs.iloc[0].to_dict()
        return None

    def list_runs(self, max_results: int = 20) -> list[dict]:
        """List recent runs."""
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=["start_time DESC"],
            max_results=max_results
        )
        return runs.to_dict('records')

    @staticmethod
    def launch_ui(port: int = 5000) -> subprocess.Popen:
        """Launch MLflow UI server."""
        # Get the mlruns directory
        mlruns_path = Path("mlruns").absolute()
        mlruns_path.mkdir(exist_ok=True)

        # Launch MLflow UI - use file:// URI format for backend
        backend_uri = mlruns_path.as_uri()

        # Build command as list (safe from injection)
        cmd = [sys.executable, "-m", "mlflow", "ui", "--port", str(port), "--backend-store-uri", backend_uri]

        # Launch process with proper flags for Windows
        if os.name == 'nt':
            # On Windows, use CREATE_NEW_CONSOLE so it runs independently
            process = subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        return process

    @staticmethod
    def get_ui_url(port: int = 5000) -> str:
        """Get the MLflow UI URL."""
        return f"http://localhost:{port}"
