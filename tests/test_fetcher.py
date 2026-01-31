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
