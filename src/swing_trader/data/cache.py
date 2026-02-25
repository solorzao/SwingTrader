import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


class DataCache:
    """Parquet-based disk cache for OHLCV data.

    Stores fetched data as parquet files to avoid repeated Yahoo Finance calls.
    Files are keyed by ticker and period, with a configurable TTL.
    """

    def __init__(self, cache_dir: Path = Path("data/cache"), ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.ttl = timedelta(hours=ttl_hours)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, ticker: str, period: str) -> Path:
        """Get the cache file path for a ticker+period combo."""
        return self.cache_dir / f"{ticker.upper()}_{period}.parquet"

    def get(self, ticker: str, period: str) -> pd.DataFrame | None:
        """Get cached data if it exists and is not expired.

        Returns None if cache miss or expired.
        """
        path = self._cache_path(ticker, period)
        if not path.exists():
            return None

        # Check TTL based on file modification time
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        if datetime.now() - mtime > self.ttl:
            return None

        return pd.read_parquet(path)

    def put(self, ticker: str, period: str, data: pd.DataFrame) -> None:
        """Store data in cache."""
        path = self._cache_path(ticker, period)
        data.to_parquet(path)

    def invalidate(self, ticker: str, period: str) -> None:
        """Remove cached data for a ticker+period."""
        path = self._cache_path(ticker, period)
        if path.exists():
            path.unlink()

    def clear(self) -> None:
        """Clear all cached data."""
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink()
