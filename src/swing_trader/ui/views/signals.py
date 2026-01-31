import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
import threading
from datetime import datetime
from ..theme import COLORS
from ...data.fetcher import StockDataFetcher
from ...features.indicators import TechnicalIndicators
from ...signals.generator import SignalGenerator
from ...models.base import Signal


class SignalsView:
    """Signal analysis view with ticker search and charts."""

    def __init__(self, parent: str | int):
        self.parent = parent
        self.fetcher = StockDataFetcher()
        self.indicators = TechnicalIndicators()
        self.generator = None
        self._current_data = None
        self._setup()

    def _setup(self):
        """Build the signals view UI."""
        with dpg.group(horizontal=True, parent=self.parent):
            # Left sidebar
            self._create_sidebar()
            dpg.add_spacer(width=10)
            # Main content
            self._create_main_content()

    def _create_sidebar(self):
        """Create the left sidebar with controls."""
        with dpg.child_window(width=280, height=-1, border=True):
            dpg.add_text("TICKER ANALYSIS", color=COLORS["accent"])
            dpg.add_spacer(height=15)

            dpg.add_text("Symbol", color=COLORS["text_muted"])
            dpg.add_input_text(
                tag="signal_ticker_input",
                hint="AAPL",
                width=-1,
                on_enter=True,
                callback=self._on_analyze,
                uppercase=True
            )

            dpg.add_spacer(height=10)
            dpg.add_text("Period", color=COLORS["text_muted"])
            dpg.add_combo(
                tag="signal_period",
                items=["1mo", "3mo", "6mo", "1y", "2y"],
                default_value="6mo",
                width=-1
            )

            dpg.add_spacer(height=15)
            dpg.add_button(
                label="Analyze",
                tag="analyze_btn",
                width=-1,
                callback=self._on_analyze
            )

            dpg.add_spacer(height=10)
            dpg.add_text("", tag="signal_status", color=COLORS["text_muted"])

            dpg.add_spacer(height=25)
            dpg.add_separator()
            dpg.add_spacer(height=15)

            dpg.add_text("QUICK SCAN", color=COLORS["accent"])
            dpg.add_spacer(height=10)

            dpg.add_text("Filter", color=COLORS["text_muted"])
            dpg.add_combo(
                tag="scan_filter",
                items=["All Signals", "BUY Only", "SELL Only"],
                default_value="All Signals",
                width=-1
            )

            dpg.add_spacer(height=10)
            dpg.add_text("Min Confidence", color=COLORS["text_muted"])
            dpg.add_slider_float(
                tag="scan_confidence",
                default_value=0.6,
                min_value=0.0,
                max_value=1.0,
                format="%.0f%%",
                width=-1
            )

            dpg.add_spacer(height=15)
            dpg.add_button(label="Scan Top Stocks", width=-1, callback=self._on_scan)

            dpg.add_spacer(height=10)
            dpg.add_text("", tag="scan_status", color=COLORS["text_muted"], wrap=260)

    def _create_main_content(self):
        """Create the main content area."""
        with dpg.child_window(width=-1, height=-1, border=True):
            # Chart area
            dpg.add_text("PRICE CHART", color=COLORS["text_secondary"])
            dpg.add_spacer(height=10)

            with dpg.plot(
                tag="signals_chart",
                height=350,
                width=-1,
                anti_aliased=True
            ):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="", tag="sig_x_axis")
                dpg.add_plot_axis(dpg.mvYAxis, label="Price ($)", tag="sig_y_axis")

            dpg.add_spacer(height=20)

            # Signal summary cards
            dpg.add_text("SIGNAL SUMMARY", color=COLORS["text_secondary"])
            dpg.add_spacer(height=10)

            with dpg.group(horizontal=True):
                self._create_signal_card("Current Signal", "sig_current", "--")
                dpg.add_spacer(width=10)
                self._create_signal_card("Confidence", "sig_confidence", "--")
                dpg.add_spacer(width=10)
                self._create_signal_card("RF Signal", "sig_rf", "--")
                dpg.add_spacer(width=10)
                self._create_signal_card("XGB Signal", "sig_xgb", "--")
                dpg.add_spacer(width=10)
                self._create_signal_card("LSTM Signal", "sig_lstm", "--")

            dpg.add_spacer(height=20)

            # Scan results table
            dpg.add_text("SCAN RESULTS", color=COLORS["text_secondary"])
            dpg.add_spacer(height=10)
            self._scan_results_container = dpg.add_child_window(height=200, border=True, tag="scan_results_container")
            with dpg.group(parent=self._scan_results_container):
                dpg.add_text("Run a scan to see results", color=COLORS["text_muted"])

    def _create_signal_card(self, title: str, tag: str, default: str):
        """Create a metric card."""
        with dpg.child_window(width=150, height=80, border=True):
            dpg.add_text(title.upper(), color=COLORS["text_secondary"])
            dpg.add_spacer(height=5)
            dpg.add_text(default, tag=tag, color=COLORS["text_muted"])

    def _on_analyze(self, sender=None, data=None):
        """Handle analyze button click."""
        ticker = dpg.get_value("signal_ticker_input").strip().upper()
        if not ticker:
            dpg.set_value("signal_status", "Enter a ticker symbol")
            return

        dpg.set_value("signal_status", f"Analyzing {ticker}...")
        dpg.configure_item("analyze_btn", enabled=False)

        def run_analysis():
            try:
                period = dpg.get_value("signal_period")

                # Fetch data
                data = self.fetcher.fetch(ticker, period=period)
                if data is None or data.empty:
                    dpg.set_value("signal_status", f"No data found for {ticker}")
                    dpg.configure_item("analyze_btn", enabled=True)
                    return

                # Add indicators
                data = self.indicators.add_all(data)
                self._current_data = data

                # Update chart
                self._update_chart(data, ticker)

                # Try to generate signals if models exist
                self._generate_signals(ticker, data)

                dpg.set_value("signal_status", f"Analysis complete for {ticker}")

            except Exception as e:
                dpg.set_value("signal_status", f"Error: {str(e)[:50]}")
            finally:
                dpg.configure_item("analyze_btn", enabled=True)

        thread = threading.Thread(target=run_analysis, daemon=True)
        thread.start()

    def _update_chart(self, data: pd.DataFrame, ticker: str):
        """Update the price chart with data."""
        try:
            # Clear existing series
            if dpg.does_item_exist("price_line"):
                dpg.delete_item("price_line")
            if dpg.does_item_exist("sma20_line"):
                dpg.delete_item("sma20_line")
            if dpg.does_item_exist("sma50_line"):
                dpg.delete_item("sma50_line")

            # Convert dates to numeric for plotting
            dates = list(range(len(data)))
            prices = data['Close'].values.tolist()

            # Add price line
            dpg.add_line_series(
                dates, prices,
                label=f"{ticker} Close",
                parent="sig_y_axis",
                tag="price_line"
            )

            # Add SMA lines if available
            if 'SMA_20' in data.columns:
                sma20 = data['SMA_20'].fillna(method='bfill').values.tolist()
                dpg.add_line_series(
                    dates, sma20,
                    label="SMA 20",
                    parent="sig_y_axis",
                    tag="sma20_line"
                )

            if 'SMA_50' in data.columns:
                sma50 = data['SMA_50'].fillna(method='bfill').values.tolist()
                dpg.add_line_series(
                    dates, sma50,
                    label="SMA 50",
                    parent="sig_y_axis",
                    tag="sma50_line"
                )

            # Fit axes
            dpg.fit_axis_data("sig_x_axis")
            dpg.fit_axis_data("sig_y_axis")

        except Exception as e:
            print(f"Chart update error: {e}")

    def _generate_signals(self, ticker: str, data: pd.DataFrame):
        """Generate signals using available models."""
        try:
            from pathlib import Path
            import joblib

            models_dir = Path("models")
            signals = {}

            # Check for each model type
            model_files = {
                "rf": ("random_forest.joblib", "sig_rf"),
                "xgb": ("xgboost.joblib", "sig_xgb"),
                "lstm": ("lstm.joblib", "sig_lstm")
            }

            for model_key, (filename, tag) in model_files.items():
                model_path = models_dir / filename
                if model_path.exists():
                    try:
                        model = joblib.load(model_path)
                        # Prepare features
                        feature_cols = [c for c in data.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
                        if feature_cols:
                            X = data[feature_cols].dropna().iloc[-1:].values
                            pred = model.predict(X)[0]
                            signal_name = {1: "BUY", 0: "HOLD", -1: "SELL"}.get(pred, "HOLD")
                            signals[model_key] = pred
                            color = COLORS["buy"] if pred == 1 else COLORS["sell"] if pred == -1 else COLORS["text_muted"]
                            dpg.set_value(tag, signal_name)
                            dpg.configure_item(tag, color=color)
                    except Exception as e:
                        dpg.set_value(tag, "Error")
                else:
                    dpg.set_value(tag, "Not trained")

            # Calculate consensus
            if signals:
                avg_signal = sum(signals.values()) / len(signals)
                if avg_signal > 0.3:
                    consensus = "BUY"
                    color = COLORS["buy"]
                elif avg_signal < -0.3:
                    consensus = "SELL"
                    color = COLORS["sell"]
                else:
                    consensus = "HOLD"
                    color = COLORS["warning"]

                dpg.set_value("sig_current", consensus)
                dpg.configure_item("sig_current", color=color)

                confidence = abs(avg_signal)
                dpg.set_value("sig_confidence", f"{confidence:.0%}")
            else:
                dpg.set_value("sig_current", "No models")
                dpg.set_value("sig_confidence", "--")

        except Exception as e:
            dpg.set_value("sig_current", "Error")
            print(f"Signal generation error: {e}")

    def _on_scan(self, sender=None, data=None):
        """Handle scan button click."""
        dpg.set_value("scan_status", "Scanning...")

        def run_scan():
            try:
                # Top stocks to scan
                tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "WMT"]
                filter_type = dpg.get_value("scan_filter")
                min_conf = dpg.get_value("scan_confidence")

                results = []

                for i, ticker in enumerate(tickers):
                    dpg.set_value("scan_status", f"Scanning {ticker} ({i+1}/{len(tickers)})...")
                    try:
                        data = self.fetcher.fetch(ticker, period="6mo")
                        if data is None or data.empty:
                            continue

                        data = self.indicators.add_all(data)

                        # Simple signal based on RSI and MACD
                        if 'RSI' in data.columns and 'MACD' in data.columns:
                            latest = data.iloc[-1]
                            rsi = latest.get('RSI', 50)
                            macd = latest.get('MACD', 0)
                            macd_signal = latest.get('MACD_Signal', 0)

                            # Simple signal logic
                            if rsi < 30 and macd > macd_signal:
                                signal = "BUY"
                                confidence = min(1.0, (30 - rsi) / 30 + 0.5)
                            elif rsi > 70 and macd < macd_signal:
                                signal = "SELL"
                                confidence = min(1.0, (rsi - 70) / 30 + 0.5)
                            else:
                                signal = "HOLD"
                                confidence = 0.5

                            # Apply filters
                            if filter_type == "BUY Only" and signal != "BUY":
                                continue
                            if filter_type == "SELL Only" and signal != "SELL":
                                continue
                            if confidence < min_conf:
                                continue

                            results.append({
                                "ticker": ticker,
                                "signal": signal,
                                "confidence": confidence,
                                "price": latest['Close'],
                                "rsi": rsi
                            })
                    except Exception as e:
                        continue

                # Update results table
                self._update_scan_results(results)
                dpg.set_value("scan_status", f"Found {len(results)} signals")

            except Exception as e:
                dpg.set_value("scan_status", f"Scan error: {str(e)[:40]}")

        thread = threading.Thread(target=run_scan, daemon=True)
        thread.start()

    def _update_scan_results(self, results: list):
        """Update the scan results table."""
        dpg.delete_item("scan_results_container", children_only=True)

        if not results:
            with dpg.group(parent="scan_results_container"):
                dpg.add_text("No signals found matching criteria", color=COLORS["text_muted"])
            return

        with dpg.group(parent="scan_results_container"):
            with dpg.table(header_row=True, borders_innerH=True, borders_outerH=True,
                          borders_innerV=True, borders_outerV=True, row_background=True):
                dpg.add_table_column(label="Ticker", width_fixed=True, init_width_or_weight=80)
                dpg.add_table_column(label="Signal", width_fixed=True, init_width_or_weight=80)
                dpg.add_table_column(label="Confidence", width_fixed=True, init_width_or_weight=100)
                dpg.add_table_column(label="Price", width_fixed=True, init_width_or_weight=100)
                dpg.add_table_column(label="RSI", width_fixed=True, init_width_or_weight=80)

                for r in sorted(results, key=lambda x: x['confidence'], reverse=True):
                    with dpg.table_row():
                        dpg.add_text(r['ticker'])

                        sig_color = COLORS["buy"] if r['signal'] == "BUY" else COLORS["sell"] if r['signal'] == "SELL" else COLORS["text_muted"]
                        dpg.add_text(r['signal'], color=sig_color)

                        dpg.add_text(f"{r['confidence']:.0%}")
                        dpg.add_text(f"${r['price']:.2f}")
                        dpg.add_text(f"{r['rsi']:.1f}")
