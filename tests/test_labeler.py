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
    # Note: dtype becomes float64 due to NaN values at the end
    assert np.issubdtype(labels.dtype, np.floating) or np.issubdtype(labels.dtype, np.integer)
    assert set(labels.dropna().unique()).issubset({-1, 0, 1})

def test_labeler_buy_on_uptrend(sample_prices):
    labeler = SignalLabeler(forward_days=2, threshold=0.05)
    labels = labeler.create_labels(sample_prices)
    assert labels.iloc[0] == 1  # 100 -> 110 = 10% gain

def test_labeler_sell_on_downtrend(sample_prices):
    labeler = SignalLabeler(forward_days=2, threshold=0.05)
    labels = labeler.create_labels(sample_prices)
    assert labels.iloc[4] == -1  # 120 -> 110 = -8% loss
