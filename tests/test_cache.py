import pytest
import pandas as pd
import os
import time
from swing_trader.data.cache import DataCache


@pytest.fixture
def sample_ohlcv():
    df = pd.DataFrame({
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
        "volume": [1000, 1100],
    }, index=pd.date_range("2024-01-01", periods=2, freq="D"))
    df.index.name = "date"
    return df


@pytest.fixture
def cache(tmp_path):
    return DataCache(cache_dir=tmp_path, ttl_hours=24)


def test_put_and_get_roundtrip(cache, sample_ohlcv):
    cache.put("AAPL", "1y", sample_ohlcv)
    result = cache.get("AAPL", "1y")
    assert result is not None
    # check_freq=False because parquet doesn't preserve DatetimeIndex freq
    pd.testing.assert_frame_equal(result, sample_ohlcv, check_freq=False)


def test_cache_miss_returns_none(cache):
    result = cache.get("AAPL", "1y")
    assert result is None


def test_ttl_expiration_returns_none(tmp_path, sample_ohlcv):
    cache = DataCache(cache_dir=tmp_path, ttl_hours=0)
    cache.put("AAPL", "1y", sample_ohlcv)
    # TTL of 0 hours means any file is immediately expired
    time.sleep(0.1)
    result = cache.get("AAPL", "1y")
    assert result is None


def test_invalidate_removes_file(cache, sample_ohlcv):
    cache.put("AAPL", "1y", sample_ohlcv)
    cache.invalidate("AAPL", "1y")
    assert cache.get("AAPL", "1y") is None
    assert not cache._cache_path("AAPL", "1y").exists()


def test_clear_removes_all_files(cache, sample_ohlcv):
    cache.put("AAPL", "1y", sample_ohlcv)
    cache.put("MSFT", "2y", sample_ohlcv)
    cache.clear()
    assert cache.get("AAPL", "1y") is None
    assert cache.get("MSFT", "2y") is None


def test_cache_path_normalizes_ticker(cache):
    assert cache._cache_path("aapl", "1y").name == "AAPL_1y.parquet"
    assert cache._cache_path("Aapl", "1y").name == "AAPL_1y.parquet"
