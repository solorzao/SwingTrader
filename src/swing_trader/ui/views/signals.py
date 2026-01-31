import dearpygui.dearpygui as dpg
import pandas as pd
from pathlib import Path
from ..theme import COLORS

class SignalsView:
    """Signal analysis view with ticker search and charts."""

    def __init__(self, parent: str | int):
        self.parent = parent
        self.generator = None
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
                callback=self._on_analyze
            )

            dpg.add_spacer(height=10)
            dpg.add_text("Period", color=COLORS["text_muted"])
            dpg.add_combo(
                tag="signal_period",
                items=["1mo", "3mo", "6mo", "1y"],
                default_value="6mo",
                width=-1
            )

            dpg.add_spacer(height=15)
            dpg.add_button(
                label="Analyze",
                width=-1,
                callback=self._on_analyze
            )

            dpg.add_spacer(height=25)
            dpg.add_separator()
            dpg.add_spacer(height=15)

            dpg.add_text("UNIVERSE SCAN", color=COLORS["accent"])
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
            dpg.add_button(label="Scan S&P 500", width=-1, callback=self._on_scan)

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
                dpg.add_plot_axis(dpg.mvXAxis, label="", tag="sig_x_axis", time=True)
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

    def _create_signal_card(self, title: str, tag: str, default: str):
        """Create a metric card."""
        with dpg.child_window(width=150, height=80, border=True):
            dpg.add_text(title.upper(), color=COLORS["text_secondary"])
            dpg.add_spacer(height=5)
            dpg.add_text(default, tag=tag, color=COLORS["text_muted"])

    def _on_analyze(self, sender=None, data=None):
        """Handle analyze button click."""
        ticker = dpg.get_value("signal_ticker_input")
        if not ticker:
            return

        dpg.set_value("sig_current", "Loading...")
        # In production, this would call SignalGenerator
        print(f"Analyzing {ticker}...")

    def _on_scan(self, sender=None, data=None):
        """Handle scan button click."""
        print("Scanning universe...")
