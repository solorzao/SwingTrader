"""End-to-end integration tests for the swing trading ML application."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from swing_trader.data import StockDataFetcher
from swing_trader.features import TechnicalIndicators, SignalLabeler
from swing_trader.models import RandomForestModel, XGBoostModel, EnsembleModel
from swing_trader.backtest import BacktestEngine


class TestDataPipeline:
    """Test the data fetching and feature engineering pipeline."""

    def test_fetch_and_add_indicators(self):
        """Test fetching data and adding technical indicators."""
        fetcher = StockDataFetcher()
        df = fetcher.fetch("AAPL", period="3mo")

        assert not df.empty
        assert all(col in df.columns for col in ["open", "high", "low", "close", "volume"])

        indicators = TechnicalIndicators()
        df_with_indicators = indicators.add_all(df)

        # Check key indicators are present
        expected_indicators = ["sma_20", "rsi_14", "macd", "atr_14", "obv"]
        for ind in expected_indicators:
            assert ind in df_with_indicators.columns

        # Verify no NaN after dropna
        assert not df_with_indicators.isnull().any().any()

    def test_label_generation(self):
        """Test signal label generation."""
        fetcher = StockDataFetcher()
        df = fetcher.fetch("MSFT", period="3mo")

        labeler = SignalLabeler(forward_days=5, threshold=0.02)
        labels = labeler.create_labels(df)

        # Labels should be -1, 0, or 1 (plus NaN at the end)
        valid_labels = labels.dropna()
        assert set(valid_labels.unique()).issubset({-1, 0, 1})


class TestModelTraining:
    """Test model training and prediction."""

    @pytest.fixture
    def training_data(self):
        """Create training data from real stock data."""
        fetcher = StockDataFetcher()
        indicators = TechnicalIndicators()
        labeler = SignalLabeler()

        # Fetch and process data
        df = fetcher.fetch("AAPL", period="1y")
        df = indicators.add_all(df)
        df["target"] = labeler.create_labels(df)
        df = df.dropna()

        # Prepare features and labels
        exclude = ["target", "open", "high", "low", "close", "volume"]
        feature_cols = [c for c in df.columns if c not in exclude]
        X = df[feature_cols]
        y = df["target"]

        return X, y

    def test_random_forest_training(self, training_data):
        """Test RandomForest model training."""
        X, y = training_data

        # Split data
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = RandomForestModel(n_estimators=50)
        model.fit(X_train, y_train)

        assert model.is_fitted
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)
        assert all(p in [-1, 0, 1] for p in preds)

    def test_xgboost_training(self, training_data):
        """Test XGBoost model training."""
        X, y = training_data

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = XGBoostModel(n_estimators=50)
        model.fit(X_train, y_train)

        assert model.is_fitted
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)
        assert all(p in [-1, 0, 1] for p in preds)

    def test_ensemble_prediction(self, training_data):
        """Test ensemble model combining RF and XGBoost."""
        X, y = training_data

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        rf = RandomForestModel(n_estimators=20)
        xgb = XGBoostModel(n_estimators=20)

        rf.fit(X_train, y_train)
        xgb.fit(X_train, y_train)

        ensemble = EnsembleModel(models=[rf, xgb], method="voting")
        preds = ensemble.predict(X_test)

        assert len(preds) == len(X_test)
        assert all(p in [-1, 0, 1] for p in preds)


class TestModelPersistence:
    """Test model save/load functionality."""

    def test_save_and_load_model(self):
        """Test saving and loading a trained model."""
        # Create and train model
        np.random.seed(42)
        X = pd.DataFrame({
            "f1": np.random.randn(100),
            "f2": np.random.randn(100),
            "f3": np.random.randn(100),
        })
        y = pd.Series(np.random.choice([-1, 0, 1], 100))

        model = RandomForestModel(n_estimators=10)
        model.fit(X, y)
        preds_before = model.predict(X)

        # Save and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model.joblib"
            model.save(path)

            loaded = RandomForestModel()
            loaded.load(path)
            preds_after = loaded.predict(X)

        np.testing.assert_array_equal(preds_before, preds_after)


class TestBacktesting:
    """Test backtesting functionality."""

    def test_backtest_with_signals(self):
        """Test backtesting with generated signals."""
        # Create sample data with signals
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2024-01-01", periods=n)
        close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
        signals = np.random.choice([-1, 0, 1], n, p=[0.2, 0.6, 0.2])

        df = pd.DataFrame({
            "close": close,
            "signal": signals
        }, index=dates)

        engine = BacktestEngine(initial_capital=10000)
        result = engine.run(df)

        # Verify result attributes
        assert hasattr(result, "total_return")
        assert hasattr(result, "sharpe_ratio")
        assert hasattr(result, "max_drawdown")
        assert hasattr(result, "equity_curve")
        assert hasattr(result, "trades")

        # Equity curve should start at initial capital
        assert result.equity_curve.iloc[0] == 10000

        # Max drawdown should be negative or zero
        assert result.max_drawdown <= 0


class TestFullPipeline:
    """Test the complete pipeline end-to-end."""

    @pytest.mark.integration
    def test_full_pipeline(self):
        """
        Test complete pipeline: fetch -> features -> train -> predict -> backtest
        """
        # 1. Fetch data
        fetcher = StockDataFetcher()
        df = fetcher.fetch("AAPL", period="1y")
        assert len(df) > 100

        # 2. Add features
        indicators = TechnicalIndicators()
        df = indicators.add_all(df)

        # 3. Create labels
        labeler = SignalLabeler(forward_days=5, threshold=0.02)
        df["target"] = labeler.create_labels(df)
        df = df.dropna()

        # 4. Train model
        exclude = ["target", "open", "high", "low", "close", "volume"]
        feature_cols = [c for c in df.columns if c not in exclude]
        X = df[feature_cols]
        y = df["target"]

        split = int(len(X) * 0.7)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = RandomForestModel(n_estimators=50)
        model.fit(X_train, y_train)

        # 5. Generate predictions
        preds = model.predict(X_test)
        accuracy = (preds == y_test.values).mean()
        print(f"Model accuracy: {accuracy:.2%}")

        # 6. Backtest
        test_df = df.iloc[split:].copy()
        test_df["signal"] = preds

        engine = BacktestEngine(initial_capital=10000)
        result = engine.run(test_df)

        print(result.summary())

        # Assertions
        assert result.total_trades >= 0
        assert -1 <= result.max_drawdown <= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
