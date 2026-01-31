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
        **kwargs
    ) -> "LSTMModel":
        """Train the LSTM model."""
        X_prep = self._prepare_features(X)

        X_values = X_prep.values.astype(np.float32)
        self.scaler_mean = X_values.mean(axis=0)
        self.scaler_std = X_values.std(axis=0) + 1e-8
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
