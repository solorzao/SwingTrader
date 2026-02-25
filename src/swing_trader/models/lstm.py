import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .base import BaseModel, Signal


class LSTMNetwork(nn.Module):
    """LSTM neural network for sequence classification."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 3
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


class LSTMModel(BaseModel):
    """LSTM classifier for trading signals with CUDA support."""

    def __init__(
        self,
        sequence_length: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        device: str | None = None
    ):
        super().__init__(name="LSTM")
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._label_map = {-1: 0, 0: 1, 1: 2}
        self._label_map_inv = {0: -1, 1: 0, 2: 1}
        self.scaler_mean = None
        self.scaler_std = None

    def _create_sequences(self, X: np.ndarray, y: np.ndarray | None = None):
        """Create sequences for LSTM input."""
        sequences = []
        labels = []
        for i in range(len(X) - self.sequence_length + 1):
            sequences.append(X[i:i + self.sequence_length])
            if y is not None:
                labels.append(y[i + self.sequence_length - 1])

        sequences = np.array(sequences)
        if y is not None:
            labels = np.array(labels)
            return sequences, labels
        return sequences

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        verbose: bool = False,
        scaler_X: pd.DataFrame | None = None,
        **kwargs
    ) -> "LSTMModel":
        """Train the LSTM model.

        Args:
            scaler_X: If provided, fit scaler on this data (should be train-only)
                to prevent data leakage. If None, fits scaler on X.
        """
        X_prep = self._prepare_features(X)

        # Fit scaler on scaler_X (train-only) to prevent data leakage
        if scaler_X is not None:
            scaler_data = scaler_X[self.feature_columns].values.astype(np.float32)
        else:
            scaler_data = X_prep.values.astype(np.float32)
        self.scaler_mean = scaler_data.mean(axis=0)
        self.scaler_std = scaler_data.std(axis=0) + 1e-8

        X_values = X_prep.values.astype(np.float32)
        X_scaled = (X_values - self.scaler_mean) / self.scaler_std

        y_mapped = y.map(self._label_map).values
        X_seq, y_seq = self._create_sequences(X_scaled, y_mapped)

        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.LongTensor(y_seq).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = LSTMNetwork(
            input_size=X_prep.shape[1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(dataloader):.4f}")

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict signals."""
        if self.scaler_mean is None or self.scaler_std is None:
            raise ValueError(
                "LSTM model missing scaler parameters. This model was saved with an older format. "
                "Please retrain the LSTM model to use it for predictions/backtesting."
            )

        X_prep = X[self.feature_columns].values.astype(np.float32)
        X_scaled = (X_prep - self.scaler_mean) / self.scaler_std
        X_seq = self._create_sequences(X_scaled)

        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, preds = torch.max(outputs, 1)

        return np.array([self._label_map_inv[p.item()] for p in preds])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        X_prep = X[self.feature_columns].values.astype(np.float32)
        X_scaled = (X_prep - self.scaler_mean) / self.scaler_std
        X_seq = self._create_sequences(X_scaled)

        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            proba = torch.softmax(outputs, dim=1)

        return proba.cpu().numpy()

    def save(self, path, metrics: dict = None, feature_config: dict = None) -> None:
        """Save LSTM model with all necessary attributes."""
        from pathlib import Path
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if metrics:
            self.metrics = metrics
        if feature_config:
            self.feature_config = feature_config

        # Save model state dict separately (CPU for portability)
        model_state = self.model.state_dict() if self.model else None

        joblib.dump({
            "model_state": model_state,
            "feature_columns": self.feature_columns,
            "name": self.name,
            "metrics": self.metrics,
            "feature_config": self.feature_config,
            # LSTM-specific attributes
            "sequence_length": self.sequence_length,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "input_size": len(self.feature_columns) if self.feature_columns else None,
        }, path)

    def load(self, path) -> "LSTMModel":
        """Load LSTM model with all necessary attributes."""
        from pathlib import Path
        import joblib

        data = joblib.load(path)

        self.feature_columns = data["feature_columns"]
        self.name = data["name"]
        self.metrics = data.get("metrics", {})
        self.feature_config = data.get("feature_config", None)

        # LSTM-specific attributes (with defaults for old models)
        self.sequence_length = data.get("sequence_length", 20)
        self.hidden_size = data.get("hidden_size", 64)
        self.num_layers = data.get("num_layers", 2)
        self.dropout = data.get("dropout", 0.2)
        self.scaler_mean = data.get("scaler_mean")
        self.scaler_std = data.get("scaler_std")

        # Handle both new format (model_state) and old format (model)
        model_state = data.get("model_state")
        old_model = data.get("model")
        input_size = data.get("input_size") or len(self.feature_columns)

        if model_state is not None:
            # New format: recreate network and load state dict
            self.model = LSTMNetwork(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(self.device)
            self.model.load_state_dict(model_state)
            self.model.eval()
        elif old_model is not None:
            # Old format: model was saved directly (may not work across devices)
            self.model = old_model
            if hasattr(self.model, 'eval'):
                self.model.eval()
            # Try to move to current device
            try:
                self.model = self.model.to(self.device)
            except:
                pass  # May fail if model structure changed

        self.is_fitted = True
        return self
