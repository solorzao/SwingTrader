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
        BaseModel("test")
