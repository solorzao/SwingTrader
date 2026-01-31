import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock
from swing_trader.signals.generator import SignalGenerator

@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.predict.return_value = np.array([1, 0, -1, 1, 0])
    model.predict_proba.return_value = np.array([
        [0.1, 0.2, 0.7],
        [0.3, 0.4, 0.3],
        [0.7, 0.2, 0.1],
        [0.1, 0.2, 0.7],
        [0.3, 0.4, 0.3]
    ])
    model.is_fitted = True
    model.name = "test_model"
    return model

@pytest.fixture
def sample_indicator_data():
    """Sample data with indicators."""
    np.random.seed(42)
    n = 5
    return pd.DataFrame({
        "open": [100, 101, 102, 103, 104],
        "high": [101, 102, 103, 104, 105],
        "low": [99, 100, 101, 102, 103],
        "close": [100.5, 101.5, 102.5, 103.5, 104.5],
        "volume": [1000000] * n,
        "sma_20": [100] * n,
        "rsi_14": [50] * n,
        "macd": [0.1] * n,
    })

def test_generator_with_mock_model(mock_model, sample_indicator_data):
    generator = SignalGenerator(models={"test": mock_model})
    result = generator._generate_for_data(sample_indicator_data, "TEST")

    assert "signal" in result.columns
    assert "confidence" in result.columns
    assert len(result) == len(sample_indicator_data)

def test_generator_no_models_raises():
    generator = SignalGenerator()
    df = pd.DataFrame({"close": [100, 101, 102]})

    with pytest.raises(ValueError, match="No models loaded"):
        generator._generate_for_data(df, "TEST")
