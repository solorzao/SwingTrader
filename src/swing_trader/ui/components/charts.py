import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from typing import Callable

class CandlestickChart:
    """GPU-accelerated candlestick chart component."""

    def __init__(self, tag: str, parent: str | int):
        self.tag = tag
        self.parent = parent
        self.x_axis_tag = f"{tag}_x_axis"
        self.y_axis_tag = f"{tag}_y_axis"
        self._setup()

    def _setup(self):
        """Create the plot widget."""
        with dpg.plot(
            label="",
            height=400,
            width=-1,
            parent=self.parent,
            tag=self.tag,
            anti_aliased=True
        ):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="", tag=self.x_axis_tag, time=True)
            dpg.add_plot_axis(dpg.mvYAxis, label="Price ($)", tag=self.y_axis_tag)

    def update(self, df: pd.DataFrame):
        """Update chart with OHLC data."""
        if df.empty:
            return

        # Clear existing series
        if dpg.does_item_exist(f"{self.tag}_candles"):
            dpg.delete_item(f"{self.tag}_candles")

        # Convert dates to timestamps for time axis
        dates = df.index.astype(np.int64) // 10**9  # Convert to Unix timestamp

        # Add candlestick series
        dpg.add_candle_series(
            dates.tolist(),
            df["open"].tolist(),
            df["close"].tolist(),
            df["low"].tolist(),
            df["high"].tolist(),
            label="Price",
            parent=self.y_axis_tag,
            tag=f"{self.tag}_candles"
        )

        # Auto-fit axes
        dpg.fit_axis_data(self.x_axis_tag)
        dpg.fit_axis_data(self.y_axis_tag)


class LineChart:
    """Line chart for equity curves and metrics."""

    def __init__(self, tag: str, parent: str | int, height: int = 300):
        self.tag = tag
        self.parent = parent
        self.height = height
        self.x_axis_tag = f"{tag}_x_axis"
        self.y_axis_tag = f"{tag}_y_axis"
        self._setup()

    def _setup(self):
        """Create the plot widget."""
        with dpg.plot(
            label="",
            height=self.height,
            width=-1,
            parent=self.parent,
            tag=self.tag,
            anti_aliased=True
        ):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="", tag=self.x_axis_tag)
            dpg.add_plot_axis(dpg.mvYAxis, label="", tag=self.y_axis_tag)

    def add_series(
        self,
        x: list,
        y: list,
        label: str,
        color: tuple = (88, 166, 255, 255)
    ):
        """Add a line series to the chart."""
        series_tag = f"{self.tag}_{label.replace(' ', '_')}"

        if dpg.does_item_exist(series_tag):
            dpg.delete_item(series_tag)

        dpg.add_line_series(
            x, y,
            label=label,
            parent=self.y_axis_tag,
            tag=series_tag
        )

        # Apply color theme
        with dpg.theme() as series_theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, color)
        dpg.bind_item_theme(series_tag, series_theme)

    def clear(self):
        """Clear all series."""
        for child in dpg.get_item_children(self.y_axis_tag, 1) or []:
            dpg.delete_item(child)

    def fit(self):
        """Auto-fit axes to data."""
        dpg.fit_axis_data(self.x_axis_tag)
        dpg.fit_axis_data(self.y_axis_tag)


class SignalOverlay:
    """Overlay buy/sell signals on a chart."""

    BUY_COLOR = (0, 212, 170, 200)   # Cyan
    SELL_COLOR = (255, 107, 107, 200)  # Coral red

    def __init__(self, chart_tag: str, y_axis_tag: str):
        self.chart_tag = chart_tag
        self.y_axis_tag = y_axis_tag

    def add_signals(self, df: pd.DataFrame, signal_col: str = "signal"):
        """Add signal markers to chart."""
        if df.empty or signal_col not in df.columns:
            return

        # Clear existing markers
        for marker in ["_buy_markers", "_sell_markers"]:
            tag = f"{self.chart_tag}{marker}"
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)

        dates = df.index.astype(np.int64) // 10**9

        # Buy signals
        buy_mask = df[signal_col] == 1
        if buy_mask.any():
            buy_x = dates[buy_mask].tolist()
            buy_y = df.loc[buy_mask, "close"].tolist()
            dpg.add_scatter_series(
                buy_x, buy_y,
                label="BUY",
                parent=self.y_axis_tag,
                tag=f"{self.chart_tag}_buy_markers"
            )
            with dpg.theme() as buy_theme:
                with dpg.theme_component(dpg.mvScatterSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, self.BUY_COLOR)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Up)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 8)
            dpg.bind_item_theme(f"{self.chart_tag}_buy_markers", buy_theme)

        # Sell signals
        sell_mask = df[signal_col] == -1
        if sell_mask.any():
            sell_x = dates[sell_mask].tolist()
            sell_y = df.loc[sell_mask, "close"].tolist()
            dpg.add_scatter_series(
                sell_x, sell_y,
                label="SELL",
                parent=self.y_axis_tag,
                tag=f"{self.chart_tag}_sell_markers"
            )
            with dpg.theme() as sell_theme:
                with dpg.theme_component(dpg.mvScatterSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, self.SELL_COLOR)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Down)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 8)
            dpg.bind_item_theme(f"{self.chart_tag}_sell_markers", sell_theme)
