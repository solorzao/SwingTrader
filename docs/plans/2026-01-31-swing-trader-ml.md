# Swing Trader ML Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a personal ML-powered swing trading signal generator for US equities with XGBoost, RandomForest, and LSTM models, featuring hyperparameter tuning, experiment tracking, and a web dashboard.

**Architecture:** Modular Python application with separate layers for data acquisition (Yahoo Finance), feature engineering (technical indicators), model training/inference with MLflow tracking, hyperparameter tuning with Optuna, backtesting, and a Streamlit dashboard for signal visualization. Models trained separately with option to ensemble.

**Tech Stack:** Python 3.14, yfinance, pandas, numpy, scikit-learn, xgboost (GPU), pytorch (LSTM + CUDA), MLflow (experiment tracking + model registry), Optuna (hyperparameter tuning), Dear PyGui (GPU-accelerated desktop UI), pytest

---

## Project Structure

```
swing-trader/
├── src/
│   └── swing_trader/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── fetcher.py          # Yahoo Finance data acquisition
│       │   └── cache.py            # SQLite caching layer
│       ├── features/
│       │   ├── __init__.py
│       │   ├── indicators.py       # Technical indicators
│       │   └── labeler.py          # Target label generation
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py             # Abstract base model
│       │   ├── random_forest.py
│       │   ├── xgboost_model.py
│       │   ├── lstm.py
│       │   └── ensemble.py         # Mix-and-match ensemble
│       ├── training/
│       │   ├── __init__.py
│       │   ├── tracker.py          # MLflow experiment tracking
│       │   └── tuner.py            # Optuna hyperparameter tuning
│       ├── backtest/
│       │   ├── __init__.py
│       │   └── engine.py           # Walk-forward backtesting
│       ├── signals/
│       │   ├── __init__.py
│       │   └── generator.py        # Signal generation
│       ├── ui/
│       │   ├── __init__.py
│       │   ├── app.py              # Dear PyGui main application
│       │   ├── views/
│       │   │   ├── __init__.py
│       │   │   ├── signals.py      # Signal analysis view
│       │   │   ├── training.py     # Training & tuning view
│       │   │   ├── backtest.py     # Backtesting view
│       │   │   └── models.py       # Model management view
│       │   └── components/
│       │       ├── __init__.py
│       │       ├── charts.py       # GPU-accelerated charts
│       │       └── tables.py       # Data tables
│       └── cli.py                  # Command-line interface
├── tests/
│   ├── __init__.py
│   ├── test_fetcher.py
│   ├── test_indicators.py
│   ├── test_models.py
│   ├── test_backtest.py
│   └── test_integration.py
├── mlruns/                         # MLflow experiment data
├── data/                           # Saved models + cache
├── docs/plans/
├── pyproject.toml
└── README.md
```

---

## Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `src/swing_trader/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "swing-trader"
version = "0.1.0"
description = "ML-powered swing trading signals"
requires-python = ">=3.11"
dependencies = [
    "yfinance>=0.2.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "xgboost>=2.0.0",
    "torch>=2.0.0",
    "mlflow>=2.10.0",
    "optuna>=3.5.0",
    "optuna-integration>=3.5.0",
    "dearpygui>=2.0.0",
    "pytest>=7.0.0",
    "joblib>=1.3.0",
]

[project.scripts]
swing-trader = "swing_trader.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

**Step 2: Create package init**

```python
# src/swing_trader/__init__.py
__version__ = "0.1.0"
```

**Step 3: Create tests init**

```python
# tests/__init__.py
```

**Step 4: Initialize git and install**

Run:
```bash
cd swing-trader
git init
pip install -e .
```

**Step 5: Commit**

```bash
git add .
git commit -m "chore: initial project setup with dependencies"
```

---

## Task 2: Data Fetcher

**Files:**
- Create: `src/swing_trader/data/__init__.py`
- Create: `src/swing_trader/data/fetcher.py`
- Create: `tests/test_fetcher.py`

**Step 1: Write the failing test**

```python
# tests/test_fetcher.py
import pytest
import pandas as pd
from swing_trader.data.fetcher import StockDataFetcher

def test_fetch_returns_dataframe():
    fetcher = StockDataFetcher()
    df = fetcher.fetch("AAPL", period="1mo")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "close" in df.columns

def test_fetch_has_required_columns():
    fetcher = StockDataFetcher()
    df = fetcher.fetch("MSFT", period="1mo")
    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        assert col in df.columns

def test_fetch_invalid_ticker_raises():
    fetcher = StockDataFetcher()
    with pytest.raises(ValueError, match="No data"):
        fetcher.fetch("INVALIDTICKER123", period="1mo")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_fetcher.py -v`
Expected: FAIL with "No module named 'swing_trader.data'"

**Step 3: Write minimal implementation**

```python
# src/swing_trader/data/__init__.py
from .fetcher import StockDataFetcher

__all__ = ["StockDataFetcher"]
```

```python
# src/swing_trader/data/fetcher.py
import yfinance as yf
import pandas as pd

class StockDataFetcher:
    """Fetch stock data from Yahoo Finance."""

    def fetch(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a ticker.

        Args:
            ticker: Stock symbol (e.g., "AAPL")
            period: Data period (e.g., "1mo", "1y", "2y", "max")
            interval: Data interval (e.g., "1d", "1wk")

        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            raise ValueError(f"No data found for ticker: {ticker}")

        df.columns = df.columns.str.lower()
        df = df[["open", "high", "low", "close", "volume"]]
        df.index.name = "date"

        return df

    def fetch_multiple(
        self,
        tickers: list[str],
        period: str = "2y",
        interval: str = "1d"
    ) -> dict[str, pd.DataFrame]:
        """Fetch data for multiple tickers."""
        return {t: self.fetch(t, period, interval) for t in tickers}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_fetcher.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/swing_trader/data/ tests/test_fetcher.py
git commit -m "feat: add Yahoo Finance data fetcher"
```

---

## Task 3: Technical Indicators (Features)

**Files:**
- Create: `src/swing_trader/features/__init__.py`
- Create: `src/swing_trader/features/indicators.py`
- Create: `tests/test_indicators.py`

**Step 1: Write the failing test**

```python
# tests/test_indicators.py
import pytest
import pandas as pd
import numpy as np
from swing_trader.features.indicators import TechnicalIndicators

@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    return pd.DataFrame({
        "open": close + np.random.randn(n),
        "high": close + abs(np.random.randn(n)),
        "low": close - abs(np.random.randn(n)),
        "close": close,
        "volume": np.random.randint(1000000, 10000000, n)
    }, index=dates)

def test_add_momentum_indicators(sample_ohlcv):
    ti = TechnicalIndicators()
    df = ti.add_all(sample_ohlcv)
    momentum_cols = ["rsi_14", "macd", "macd_signal", "roc_10"]
    for col in momentum_cols:
        assert col in df.columns

def test_add_moving_averages(sample_ohlcv):
    ti = TechnicalIndicators()
    df = ti.add_all(sample_ohlcv)
    ma_cols = ["sma_20", "sma_50", "ema_12", "ema_26"]
    for col in ma_cols:
        assert col in df.columns

def test_add_volume_indicators(sample_ohlcv):
    ti = TechnicalIndicators()
    df = ti.add_all(sample_ohlcv)
    assert "obv" in df.columns
    assert "volume_sma_20" in df.columns

def test_dropna_removes_warmup_period(sample_ohlcv):
    ti = TechnicalIndicators()
    df = ti.add_all(sample_ohlcv, dropna=True)
    assert not df.isnull().any().any()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_indicators.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/swing_trader/features/__init__.py
from .indicators import TechnicalIndicators

__all__ = ["TechnicalIndicators"]
```

```python
# src/swing_trader/features/indicators.py
import pandas as pd
import numpy as np

class TechnicalIndicators:
    """Calculate technical indicators for swing trading."""

    def add_all(self, df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
        """Add all technical indicators to OHLCV dataframe."""
        df = df.copy()
        df = self._add_moving_averages(df)
        df = self._add_momentum(df)
        df = self._add_volume(df)
        df = self._add_volatility(df)

        if dropna:
            df = df.dropna()
        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add SMA and EMA indicators."""
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
        return df

    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators: RSI, MACD, ROC."""
        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Rate of Change
        df["roc_10"] = df["close"].pct_change(10) * 100

        return df

    def _add_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators."""
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

        # On-Balance Volume
        obv = [0]
        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i-1]:
                obv.append(obv[-1] + df["volume"].iloc[i])
            elif df["close"].iloc[i] < df["close"].iloc[i-1]:
                obv.append(obv[-1] - df["volume"].iloc[i])
            else:
                obv.append(obv[-1])
        df["obv"] = obv

        return df

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators: ATR, Bollinger Bands."""
        # ATR
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift())
        low_close = abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()

        # Bollinger Bands
        df["bb_mid"] = df["sma_20"]
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + (2 * bb_std)
        df["bb_lower"] = df["bb_mid"] - (2 * bb_std)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

        return df
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_indicators.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/swing_trader/features/ tests/test_indicators.py
git commit -m "feat: add technical indicators for momentum/volume/volatility"
```

---

## Task 4: Base Model Interface

**Files:**
- Create: `src/swing_trader/models/__init__.py`
- Create: `src/swing_trader/models/base.py`
- Create: `tests/test_models.py`

**Step 1: Write the failing test**

```python
# tests/test_models.py
import pytest
import pandas as pd
import numpy as np
from swing_trader.models.base import BaseModel, Signal

def test_signal_enum():
    assert Signal.BUY.value == 1
    assert Signal.HOLD.value == 0
    assert Signal.SELL.value == -1

def test_base_model_is_abstract():
    with pytest.raises(TypeError):
        BaseModel()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py::test_signal_enum -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/swing_trader/models/__init__.py
from .base import BaseModel, Signal

__all__ = ["BaseModel", "Signal"]
```

```python
# src/swing_trader/models/base.py
from abc import ABC, abstractmethod
from enum import IntEnum
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

class Signal(IntEnum):
    """Trading signal."""
    SELL = -1
    HOLD = 0
    BUY = 1

class BaseModel(ABC):
    """Abstract base class for all trading models."""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.feature_columns: list[str] = []
        self.is_fitted = False

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs
    ) -> "BaseModel":
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict signals (BUY=1, HOLD=0, SELL=-1)."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for each class."""
        pass

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "feature_columns": self.feature_columns,
            "name": self.name
        }, path)

    def load(self, path: Path) -> "BaseModel":
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_columns = data["feature_columns"]
        self.name = data["name"]
        self.is_fitted = True
        return self

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and validate feature columns."""
        if not self.feature_columns:
            exclude = ["signal", "target", "returns", "close", "open", "high", "low", "volume"]
            self.feature_columns = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c not in exclude
            ]
        return df[self.feature_columns]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/swing_trader/models/ tests/test_models.py
git commit -m "feat: add base model interface with Signal enum"
```

---

## Task 5: Random Forest Model

**Files:**
- Modify: `src/swing_trader/models/__init__.py`
- Create: `src/swing_trader/models/random_forest.py`
- Modify: `tests/test_models.py`

**Step 1: Write the failing test**

Add to `tests/test_models.py`:

```python
from swing_trader.models.random_forest import RandomForestModel

@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "rsi_14": np.random.uniform(20, 80, n),
        "macd": np.random.randn(n),
        "sma_20": np.random.uniform(90, 110, n),
        "volume_ratio": np.random.uniform(0.5, 2.0, n),
    })
    y = pd.Series(np.random.choice([-1, 0, 1], n))
    return X, y

def test_rf_fit_and_predict(sample_data):
    X, y = sample_data
    model = RandomForestModel()
    model.fit(X, y)

    assert model.is_fitted
    preds = model.predict(X)
    assert len(preds) == len(X)
    assert all(p in [-1, 0, 1] for p in preds)

def test_rf_predict_proba(sample_data):
    X, y = sample_data
    model = RandomForestModel()
    model.fit(X, y)

    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)

def test_rf_hyperparameters():
    model = RandomForestModel(n_estimators=50, max_depth=5)
    assert model.model.n_estimators == 50
    assert model.model.max_depth == 5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py::test_rf_fit_and_predict -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/swing_trader/models/random_forest.py
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
```

Update `src/swing_trader/models/__init__.py`:

```python
from .base import BaseModel, Signal
from .random_forest import RandomForestModel

__all__ = ["BaseModel", "Signal", "RandomForestModel"]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/swing_trader/models/ tests/test_models.py
git commit -m "feat: add RandomForest model"
```

---

## Task 6: XGBoost Model

**Files:**
- Modify: `src/swing_trader/models/__init__.py`
- Create: `src/swing_trader/models/xgboost_model.py`
- Modify: `tests/test_models.py`

**Step 1: Write the failing test**

Add to `tests/test_models.py`:

```python
from swing_trader.models.xgboost_model import XGBoostModel

def test_xgb_fit_and_predict(sample_data):
    X, y = sample_data
    model = XGBoostModel()
    model.fit(X, y)

    assert model.is_fitted
    preds = model.predict(X)
    assert len(preds) == len(X)
    assert all(p in [-1, 0, 1] for p in preds)

def test_xgb_hyperparameters():
    model = XGBoostModel(n_estimators=50, max_depth=3, learning_rate=0.05)
    assert model.n_estimators == 50
    assert model.max_depth == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py::test_xgb_fit_and_predict -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/swing_trader/models/xgboost_model.py
import pandas as pd
import numpy as np
import xgboost as xgb
from .base import BaseModel, Signal

class XGBoostModel(BaseModel):
    """XGBoost classifier for trading signals."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(name="XGBoost")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            **kwargs
        )
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
        return np.array([self._label_map_inv[p] for p in preds])

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
```

Update `src/swing_trader/models/__init__.py`:

```python
from .base import BaseModel, Signal
from .random_forest import RandomForestModel
from .xgboost_model import XGBoostModel

__all__ = ["BaseModel", "Signal", "RandomForestModel", "XGBoostModel"]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/swing_trader/models/ tests/test_models.py
git commit -m "feat: add XGBoost model"
```

---

## Task 7: LSTM Model

**Files:**
- Modify: `src/swing_trader/models/__init__.py`
- Create: `src/swing_trader/models/lstm.py`
- Modify: `tests/test_models.py`

**Step 1: Write the failing test**

Add to `tests/test_models.py`:

```python
from swing_trader.models.lstm import LSTMModel

def test_lstm_fit_and_predict(sample_data):
    X, y = sample_data
    model = LSTMModel(sequence_length=10, epochs=2, batch_size=32)
    model.fit(X, y)

    assert model.is_fitted
    preds = model.predict(X)
    assert len(preds) == len(X) - model.sequence_length + 1
    assert all(p in [-1, 0, 1] for p in preds)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py::test_lstm_fit_and_predict -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/swing_trader/models/lstm.py
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
    """LSTM classifier for trading signals."""

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
```

Update `src/swing_trader/models/__init__.py`:

```python
from .base import BaseModel, Signal
from .random_forest import RandomForestModel
from .xgboost_model import XGBoostModel
from .lstm import LSTMModel

__all__ = ["BaseModel", "Signal", "RandomForestModel", "XGBoostModel", "LSTMModel"]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/swing_trader/models/ tests/test_models.py
git commit -m "feat: add LSTM model with PyTorch"
```

---

## Task 8: Target Label Generator

**Files:**
- Modify: `src/swing_trader/features/__init__.py`
- Create: `src/swing_trader/features/labeler.py`
- Create: `tests/test_labeler.py`

**Step 1: Write the failing test**

```python
# tests/test_labeler.py
import pytest
import pandas as pd
import numpy as np
from swing_trader.features.labeler import SignalLabeler

@pytest.fixture
def sample_prices():
    return pd.DataFrame({
        "close": [100, 105, 110, 115, 120, 115, 110, 105, 100, 95]
    })

def test_labeler_creates_labels(sample_prices):
    labeler = SignalLabeler(forward_days=3, threshold=0.03)
    labels = labeler.create_labels(sample_prices)
    assert len(labels) == len(sample_prices)
    assert labels.dtype == int
    assert set(labels.dropna().unique()).issubset({-1, 0, 1})

def test_labeler_buy_on_uptrend(sample_prices):
    labeler = SignalLabeler(forward_days=2, threshold=0.05)
    labels = labeler.create_labels(sample_prices)
    assert labels.iloc[0] == 1

def test_labeler_sell_on_downtrend(sample_prices):
    labeler = SignalLabeler(forward_days=2, threshold=0.05)
    labels = labeler.create_labels(sample_prices)
    assert labels.iloc[4] == -1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_labeler.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/swing_trader/features/labeler.py
import pandas as pd
import numpy as np

class SignalLabeler:
    """Generate target labels for supervised learning."""

    def __init__(self, forward_days: int = 5, threshold: float = 0.02):
        self.forward_days = forward_days
        self.threshold = threshold

    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """Create signal labels based on forward returns."""
        forward_return = df["close"].shift(-self.forward_days) / df["close"] - 1

        labels = pd.Series(0, index=df.index, dtype=int)
        labels[forward_return > self.threshold] = 1
        labels[forward_return < -self.threshold] = -1
        labels.iloc[-self.forward_days:] = np.nan

        return labels
```

Update `src/swing_trader/features/__init__.py`:

```python
from .indicators import TechnicalIndicators
from .labeler import SignalLabeler

__all__ = ["TechnicalIndicators", "SignalLabeler"]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_labeler.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/swing_trader/features/ tests/test_labeler.py
git commit -m "feat: add signal labeler for target generation"
```

---

## Task 9: Ensemble Model

**Files:**
- Modify: `src/swing_trader/models/__init__.py`
- Create: `src/swing_trader/models/ensemble.py`
- Modify: `tests/test_models.py`

**Step 1: Write the failing test**

Add to `tests/test_models.py`:

```python
from swing_trader.models.ensemble import EnsembleModel

def test_ensemble_voting(sample_data):
    X, y = sample_data
    rf = RandomForestModel(n_estimators=10)
    xgb_model = XGBoostModel(n_estimators=10)
    rf.fit(X, y)
    xgb_model.fit(X, y)

    ensemble = EnsembleModel(models=[rf, xgb_model], method="voting")
    preds = ensemble.predict(X)

    assert len(preds) == len(X)
    assert all(p in [-1, 0, 1] for p in preds)

def test_ensemble_weighted_average(sample_data):
    X, y = sample_data
    rf = RandomForestModel(n_estimators=10)
    xgb_model = XGBoostModel(n_estimators=10)
    rf.fit(X, y)
    xgb_model.fit(X, y)

    ensemble = EnsembleModel(
        models=[rf, xgb_model],
        method="weighted",
        weights=[0.6, 0.4]
    )
    proba = ensemble.predict_proba(X)

    assert proba.shape == (len(X), 3)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py::test_ensemble_voting -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/swing_trader/models/ensemble.py
import pandas as pd
import numpy as np
from typing import Literal
from .base import BaseModel, Signal

class EnsembleModel:
    """Combine multiple models for ensemble predictions."""

    def __init__(
        self,
        models: list[BaseModel],
        method: Literal["voting", "weighted"] = "voting",
        weights: list[float] | None = None
    ):
        self.models = models
        self.method = method

        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions."""
        if self.method == "voting":
            return self._voting_predict(X)
        elif self.method == "weighted":
            return self._weighted_predict(X)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _voting_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Majority voting across models."""
        all_preds = np.array([m.predict(X) for m in self.models])
        weighted_votes = np.zeros((len(X), 3))
        for i, preds in enumerate(all_preds):
            for j, p in enumerate(preds):
                class_idx = p + 1
                weighted_votes[j, class_idx] += self.weights[i]
        return np.argmax(weighted_votes, axis=1) - 1

    def _weighted_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted average of probabilities."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1) - 1

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted average of class probabilities."""
        all_proba = np.array([m.predict_proba(X) for m in self.models])
        weighted_proba = np.zeros_like(all_proba[0])
        for i, proba in enumerate(all_proba):
            weighted_proba += self.weights[i] * proba
        return weighted_proba

    def get_model_agreement(self, X: pd.DataFrame) -> pd.DataFrame:
        """Show how models agree/disagree."""
        preds = {m.name: m.predict(X) for m in self.models}
        df = pd.DataFrame(preds)
        df["ensemble"] = self.predict(X)
        df["agreement"] = (df.iloc[:, :-1].nunique(axis=1) == 1).astype(int)
        return df
```

Update `src/swing_trader/models/__init__.py`:

```python
from .base import BaseModel, Signal
from .random_forest import RandomForestModel
from .xgboost_model import XGBoostModel
from .lstm import LSTMModel
from .ensemble import EnsembleModel

__all__ = [
    "BaseModel", "Signal", "RandomForestModel",
    "XGBoostModel", "LSTMModel", "EnsembleModel"
]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/swing_trader/models/ tests/test_models.py
git commit -m "feat: add ensemble model with voting/weighted methods"
```

---

## Task 10: Backtesting Engine

**Files:**
- Create: `src/swing_trader/backtest/__init__.py`
- Create: `src/swing_trader/backtest/engine.py`
- Create: `tests/test_backtest.py`

**Step 1: Write the failing test**

```python
# tests/test_backtest.py
import pytest
import pandas as pd
import numpy as np
from swing_trader.backtest.engine import BacktestEngine, BacktestResult

@pytest.fixture
def sample_backtest_data():
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
    signals = np.random.choice([-1, 0, 1], n)
    return pd.DataFrame({"close": close, "signal": signals}, index=dates)

def test_backtest_returns_result(sample_backtest_data):
    engine = BacktestEngine(initial_capital=10000)
    result = engine.run(sample_backtest_data)
    assert isinstance(result, BacktestResult)
    assert hasattr(result, "total_return")
    assert hasattr(result, "sharpe_ratio")
    assert hasattr(result, "max_drawdown")

def test_backtest_trades_recorded(sample_backtest_data):
    engine = BacktestEngine(initial_capital=10000)
    result = engine.run(sample_backtest_data)
    assert len(result.trades) > 0
    assert "entry_date" in result.trades.columns
    assert "exit_date" in result.trades.columns
    assert "pnl" in result.trades.columns

def test_backtest_equity_curve(sample_backtest_data):
    engine = BacktestEngine(initial_capital=10000)
    result = engine.run(sample_backtest_data)
    assert len(result.equity_curve) == len(sample_backtest_data)
    assert result.equity_curve.iloc[0] == 10000
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_backtest.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/swing_trader/backtest/__init__.py
from .engine import BacktestEngine, BacktestResult

__all__ = ["BacktestEngine", "BacktestResult"]
```

```python
# src/swing_trader/backtest/engine.py
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class BacktestResult:
    """Container for backtest results."""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    equity_curve: pd.Series
    trades: pd.DataFrame
    daily_returns: pd.Series

    def summary(self) -> str:
        return f"""
Backtest Results
================
Total Return:    {self.total_return:.2%}
Annual Return:   {self.annual_return:.2%}
Sharpe Ratio:    {self.sharpe_ratio:.2f}
Max Drawdown:    {self.max_drawdown:.2%}
Win Rate:        {self.win_rate:.2%}
Profit Factor:   {self.profit_factor:.2f}
Total Trades:    {self.total_trades}
"""


class BacktestEngine:
    """Walk-forward backtesting engine."""

    def __init__(
        self,
        initial_capital: float = 10000,
        position_size: float = 1.0,
        commission: float = 0.001,
        slippage: float = 0.0005
    ):
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.commission = commission
        self.slippage = slippage

    def run(
        self,
        data: pd.DataFrame,
        signal_col: str = "signal",
        price_col: str = "close"
    ) -> BacktestResult:
        df = data.copy()
        signals = df[signal_col].values
        prices = df[price_col].values
        dates = df.index

        capital = self.initial_capital
        position = 0
        entry_price = 0
        entry_date = None

        equity = [capital]
        trades = []

        for i in range(1, len(df)):
            signal = signals[i]
            price = prices[i]

            if position != 0:
                if (position == 1 and signal == -1) or \
                   (position == -1 and signal == 1) or \
                   i == len(df) - 1:

                    exit_price = price * (1 - self.slippage if position == 1 else 1 + self.slippage)

                    if position == 1:
                        pnl = (exit_price / entry_price - 1) * capital * self.position_size
                    else:
                        pnl = (entry_price / exit_price - 1) * capital * self.position_size

                    pnl -= capital * self.position_size * self.commission * 2
                    capital += pnl

                    trades.append({
                        "entry_date": entry_date,
                        "exit_date": dates[i],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "position": "long" if position == 1 else "short",
                        "pnl": pnl,
                        "return": pnl / (self.initial_capital * self.position_size)
                    })
                    position = 0

            if position == 0 and signal != 0:
                position = signal
                entry_price = price * (1 + self.slippage if signal == 1 else 1 - self.slippage)
                entry_date = dates[i]

            equity.append(capital)

        equity_curve = pd.Series(equity, index=dates)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
            columns=["entry_date", "exit_date", "entry_price", "exit_price", "position", "pnl", "return"]
        )

        return self._calculate_metrics(equity_curve, trades_df)

    def _calculate_metrics(self, equity_curve: pd.Series, trades: pd.DataFrame) -> BacktestResult:
        daily_returns = equity_curve.pct_change().dropna()
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        n_days = len(equity_curve)
        annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0

        if daily_returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        else:
            sharpe_ratio = 0

        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        if len(trades) > 0:
            wins = trades[trades["pnl"] > 0]
            losses = trades[trades["pnl"] <= 0]
            win_rate = len(wins) / len(trades)
            total_profit = wins["pnl"].sum() if len(wins) > 0 else 0
            total_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 1
            profit_factor = total_profit / total_loss if total_loss > 0 else 0
        else:
            win_rate = 0
            profit_factor = 0

        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            equity_curve=equity_curve,
            trades=trades,
            daily_returns=daily_returns
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_backtest.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/swing_trader/backtest/ tests/test_backtest.py
git commit -m "feat: add backtesting engine with performance metrics"
```

---

## Task 11: MLflow Integration

**Files:**
- Create: `src/swing_trader/training/__init__.py`
- Create: `src/swing_trader/training/tracker.py`

**Implementation:**

```python
# src/swing_trader/training/tracker.py
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
import pandas as pd
from typing import Any

class ExperimentTracker:
    """MLflow experiment tracking and model registry."""

    def __init__(self, experiment_name: str = "swing-trader", tracking_uri: str = "mlruns"):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
        self.experiment_name = experiment_name

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
            # Our custom model wrapper
            mlflow.sklearn.log_model(model.model, model_name)
        else:
            mlflow.sklearn.log_model(model, model_name)

        if registered_name:
            mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/{model_name}", registered_name)

    def log_artifact(self, local_path: str) -> None:
        """Log any file as artifact."""
        mlflow.log_artifact(local_path)

    def end_run(self) -> None:
        """End current run."""
        mlflow.end_run()

    def get_best_run(self, metric: str = "accuracy", maximize: bool = True) -> dict:
        """Get the best run based on a metric."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
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
        return [{"name": m.name, "versions": len(m.latest_versions)}
                for m in self.client.search_registered_models()]

    def load_model(self, model_name: str, version: str = "latest") -> Any:
        """Load a registered model."""
        if version == "latest":
            model_uri = f"models:/{model_name}/latest"
        else:
            model_uri = f"models:/{model_name}/{version}"
        return mlflow.sklearn.load_model(model_uri)
```

**Commit:** `git commit -m "feat: add MLflow experiment tracking and model registry"`

---

## Task 12: Optuna Hyperparameter Tuning

**Files:**
- Create: `src/swing_trader/training/tuner.py`

**Implementation:**

```python
# src/swing_trader/training/tuner.py
import optuna
from optuna.integration import MLflowCallback
import pandas as pd
import numpy as np
from typing import Callable, Any
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
        use_gpu: bool = True
    ) -> dict:
        """Tune XGBoost hyperparameters."""

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }

            if use_gpu:
                params["tree_method"] = "gpu_hist"
                params["device"] = "cuda"

            model = XGBoostModel(**params)
            y_mapped = y.map({-1: 0, 0: 1, 1: 2})
            tscv = TimeSeriesSplit(n_splits=cv_splits)

            scores = cross_val_score(
                model.model, X, y_mapped, cv=tscv, scoring="accuracy", n_jobs=1 if use_gpu else -1
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
        callbacks = []
        if self.tracker:
            callbacks.append(MLflowCallback(
                tracking_uri=self.tracker.client.tracking_uri,
                metric_name="cv_accuracy"
            ))

        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            load_if_exists=True
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
```

**Commit:** `git commit -m "feat: add Optuna hyperparameter tuning with GPU support"`

---

## Task 13: Signal Generator

**Files:**
- Create: `src/swing_trader/signals/__init__.py`
- Create: `src/swing_trader/signals/generator.py`
- Create: `tests/test_signals.py`

See earlier plan for implementation. Key functionality:
- Generate signals for single ticker
- Scan universe (S&P 500)
- Aggregate predictions from multiple models
- Calculate confidence scores

**Commit:** `git commit -m "feat: add signal generator with universe scanning"`

---

## Task 14: CLI Interface

**Files:**
- Create: `src/swing_trader/cli.py`

Commands: train, tune, signal, scan, backtest, ui (launches desktop app)

**Commit:** `git commit -m "feat: add CLI interface"`

---

## Task 15: Dear PyGui Desktop App - Core

**Files:**
- Create: `src/swing_trader/ui/__init__.py`
- Create: `src/swing_trader/ui/app.py`
- Create: `src/swing_trader/ui/theme.py`

**Implementation:**

```python
# src/swing_trader/ui/app.py
import dearpygui.dearpygui as dpg
from pathlib import Path

class SwingTraderApp:
    """Main Dear PyGui application."""

    def __init__(self):
        self.width = 1600
        self.height = 900

    def setup(self):
        """Initialize Dear PyGui context and viewport."""
        dpg.create_context()
        dpg.create_viewport(
            title="Swing Trader ML",
            width=self.width,
            height=self.height
        )
        dpg.setup_dearpygui()

        # Apply theme
        self._setup_theme()

        # Create main window
        self._create_main_window()

    def _setup_theme(self):
        """Setup dark trading theme."""
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (15, 15, 15))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (30, 30, 30))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (40, 80, 120))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (50, 100, 150))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6)
        dpg.bind_theme(global_theme)

    def _create_main_window(self):
        """Create main application window with navigation."""
        with dpg.window(tag="main_window", label="Swing Trader ML"):
            # Navigation tabs
            with dpg.tab_bar(tag="main_tabs"):
                with dpg.tab(label="Signals", tag="tab_signals"):
                    dpg.add_text("Signal Analysis", tag="signals_placeholder")

                with dpg.tab(label="Training", tag="tab_training"):
                    dpg.add_text("Model Training & Tuning", tag="training_placeholder")

                with dpg.tab(label="Backtest", tag="tab_backtest"):
                    dpg.add_text("Backtesting", tag="backtest_placeholder")

                with dpg.tab(label="Models", tag="tab_models"):
                    dpg.add_text("Model Registry", tag="models_placeholder")

        dpg.set_primary_window("main_window", True)

    def run(self):
        """Start the application."""
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

def main():
    app = SwingTraderApp()
    app.setup()
    app.run()

if __name__ == "__main__":
    main()
```

**Commit:** `git commit -m "feat: add Dear PyGui desktop app core structure"`

---

## Task 16: Desktop App - Charts Component

**Files:**
- Create: `src/swing_trader/ui/components/__init__.py`
- Create: `src/swing_trader/ui/components/charts.py`

GPU-accelerated candlestick charts, line plots for equity curves, signal overlays.

**Commit:** `git commit -m "feat: add GPU-accelerated chart components"`

---

## Task 17: Desktop App - Signals View

**Files:**
- Create: `src/swing_trader/ui/views/__init__.py`
- Create: `src/swing_trader/ui/views/signals.py`

Signal analysis view with:
- Ticker input and search
- Candlestick chart with signal overlays (BUY/SELL markers)
- Model confidence display
- Real-time signal updates

**Commit:** `git commit -m "feat: add signals analysis view"`

---

## Task 18: Desktop App - Training View

**Files:**
- Create: `src/swing_trader/ui/views/training.py`

Training view with:
- Model selection (RF, XGBoost, LSTM)
- Hyperparameter sliders/inputs
- "Auto-tune" button (Optuna)
- Real-time training metrics charts
- GPU utilization display
- MLflow run history table

**Commit:** `git commit -m "feat: add training and tuning view"`

---

## Task 19: Desktop App - Backtest View

**Files:**
- Create: `src/swing_trader/ui/views/backtest.py`

Backtest view with:
- Model/date range selection
- Equity curve chart
- Performance metrics display
- Trade log table
- Drawdown visualization

**Commit:** `git commit -m "feat: add backtesting view"`

---

## Task 20: Desktop App - Models View

**Files:**
- Create: `src/swing_trader/ui/views/models.py`

Model management view with:
- MLflow model registry browser
- Version comparison
- Model deployment (set active)
- Performance comparison charts

**Commit:** `git commit -m "feat: add model registry view"`

---

## Task 21: Integration Tests

**Files:**
- Create: `tests/test_integration.py`

Full pipeline tests: fetch -> features -> train -> tune -> track -> backtest -> signal

**Commit:** `git commit -m "test: add integration tests for full pipeline"`

---

## Summary

| Task | Component | Description |
|------|-----------|-------------|
| 1 | Setup | Project structure, dependencies |
| 2 | Data | Yahoo Finance fetcher |
| 3 | Features | Technical indicators |
| 4 | Models | Base interface |
| 5 | Models | RandomForest |
| 6 | Models | XGBoost (GPU) |
| 7 | Models | LSTM (CUDA) |
| 8 | Features | Signal labeler |
| 9 | Models | Ensemble |
| 10 | Backtest | Engine + metrics |
| 11 | Training | MLflow experiment tracking |
| 12 | Training | Optuna hyperparameter tuning |
| 13 | Signals | Generator + scanner |
| 14 | CLI | Command interface |
| 15 | UI | Dear PyGui app core |
| 16 | UI | Chart components |
| 17 | UI | Signals view |
| 18 | UI | Training view |
| 19 | UI | Backtest view |
| 20 | UI | Models view |
| 21 | Tests | Integration tests |

**Future enhancements:**
- SQLite caching for data
- Real-time alerts / notifications
- More sophisticated position sizing
- Multi-timeframe analysis
- Paper trading mode with live data
