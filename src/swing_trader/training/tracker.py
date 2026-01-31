import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
from typing import Any

class ExperimentTracker:
    """MLflow experiment tracking and model registry."""

    def __init__(self, experiment_name: str = "swing-trader", tracking_uri: str = "mlruns"):
        """
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: Path to MLflow tracking directory
        """
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

    def start_run(self, run_name: str, tags: dict | None = None) -> str:
        """Start a new MLflow run."""
        run = mlflow.start_run(run_name=run_name, tags=tags)
        return run.info.run_id

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics (can be called multiple times for training curves)."""
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model, model_name: str, registered_name: str | None = None):
        """Log model artifact and optionally register it."""
        if hasattr(model, 'model'):
            # Our custom model wrapper - log the underlying sklearn/xgboost model
            mlflow.sklearn.log_model(model.model, model_name)
        else:
            mlflow.sklearn.log_model(model, model_name)

        if registered_name:
            run_id = mlflow.active_run().info.run_id
            mlflow.register_model(f"runs:/{run_id}/{model_name}", registered_name)

    def log_artifact(self, local_path: str) -> None:
        """Log any file as artifact."""
        mlflow.log_artifact(local_path)

    def end_run(self) -> None:
        """End current run."""
        mlflow.end_run()

    def get_best_run(self, metric: str = "accuracy", maximize: bool = True) -> dict:
        """Get the best run based on a metric."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            return {}

        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        if runs.empty:
            return {}

        metric_col = f"metrics.{metric}"
        if metric_col in runs.columns:
            best_idx = runs[metric_col].idxmax() if maximize else runs[metric_col].idxmin()
            return runs.loc[best_idx].to_dict()
        return {}

    def list_registered_models(self) -> list[dict]:
        """List all registered models."""
        models = []
        for m in self.client.search_registered_models():
            models.append({
                "name": m.name,
                "versions": len(m.latest_versions) if m.latest_versions else 0
            })
        return models

    def load_model(self, model_name: str, version: str = "latest") -> Any:
        """Load a registered model."""
        if version == "latest":
            model_uri = f"models:/{model_name}/latest"
        else:
            model_uri = f"models:/{model_name}/{version}"
        return mlflow.sklearn.load_model(model_uri)

    def get_run_history(self, max_results: int = 100) -> list[dict]:
        """Get recent run history."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            return []

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=max_results,
            order_by=["start_time DESC"]
        )

        return runs.to_dict(orient="records") if not runs.empty else []
