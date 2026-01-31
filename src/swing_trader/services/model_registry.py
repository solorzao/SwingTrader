"""Model registry for managing trained ML models."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import joblib


@dataclass
class ModelInfo:
    """Information about a registered model."""
    name: str
    model_type: str
    path: Path
    created: datetime
    feature_count: int


class ModelRegistry:
    """Registry for managing trained ML models.

    Provides lazy loading and caching of models, with automatic
    detection of model types from filenames.
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self._models: dict[str, ModelInfo] = {}
        self._loaded_models: dict[str, Any] = {}
        self.refresh()

    def refresh(self) -> None:
        """Scan model directory and register all .joblib files."""
        self._models.clear()
        self._loaded_models.clear()

        if not self.model_dir.exists():
            return

        for path in self.model_dir.glob("*.joblib"):
            self._register_model(path)

    def _register_model(self, path: Path) -> None:
        """Load metadata from a model file without fully loading the model."""
        try:
            data = joblib.load(path)

            name = data.get("name", path.stem)
            feature_columns = data.get("feature_columns", [])
            feature_count = len(feature_columns)

            # Detect model type from filename
            model_type = self._detect_model_type(path.stem)

            # Get file creation time
            created = datetime.fromtimestamp(path.stat().st_mtime)

            info = ModelInfo(
                name=name,
                model_type=model_type,
                path=path,
                created=created,
                feature_count=feature_count
            )

            self._models[name] = info

        except Exception as e:
            # Skip files that can't be loaded
            print(f"Warning: Could not load model {path}: {e}")

    def _detect_model_type(self, filename: str) -> str:
        """Detect model type from filename."""
        filename_lower = filename.lower()

        if "random_forest" in filename_lower or "rf" in filename_lower:
            return "Random Forest"
        elif "xgboost" in filename_lower or "xgb" in filename_lower:
            return "XGBoost"
        elif "lstm" in filename_lower:
            return "LSTM"
        else:
            return "Unknown"

    def list_models(self) -> list[ModelInfo]:
        """Return list of all registered models."""
        return list(self._models.values())

    def get_model(self, name: str) -> Any:
        """Load and return a model by name (lazy loading, cached)."""
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found in registry")

        if name not in self._loaded_models:
            self._loaded_models[name] = self._load_model(self._models[name])

        return self._loaded_models[name]

    def _load_model(self, info: ModelInfo) -> Any:
        """Fully load a model using the appropriate model class."""
        from ..models.random_forest import RandomForestModel
        from ..models.xgboost_model import XGBoostModel
        from ..models.lstm import LSTMModel

        model_classes = {
            "Random Forest": RandomForestModel,
            "XGBoost": XGBoostModel,
            "LSTM": LSTMModel
        }

        model_class = model_classes.get(info.model_type)

        if model_class is None:
            # Fall back to loading raw joblib data for unknown types
            return joblib.load(info.path)

        model = model_class()
        model.load(info.path)
        return model

    def get_model_info(self, name: str) -> ModelInfo:
        """Return ModelInfo for a model by name."""
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found in registry")
        return self._models[name]

    def delete_model(self, name: str) -> None:
        """Delete a model file and remove from registry."""
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found in registry")

        info = self._models[name]

        # Delete the file
        if info.path.exists():
            info.path.unlink()

        # Remove from registry
        del self._models[name]

        # Remove from cache if loaded
        if name in self._loaded_models:
            del self._loaded_models[name]
