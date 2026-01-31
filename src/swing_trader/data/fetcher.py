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
