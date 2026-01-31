import dearpygui.dearpygui as dpg
from ..theme import COLORS

class BacktestView:
    """Backtesting results view."""

    def __init__(self, parent: str | int):
        self.parent = parent
        self._setup()

    def _setup(self):
        """Build the backtest view UI."""
        with dpg.group(horizontal=True, parent=self.parent):
            self._create_controls()
            dpg.add_spacer(width=10)
            self._create_results()

    def _create_controls(self):
        """Create backtest configuration panel."""
        with dpg.child_window(width=280, height=-1, border=True):
            dpg.add_text("BACKTEST CONFIG", color=COLORS["accent"])
            dpg.add_spacer(height=15)

            dpg.add_text("Symbol", color=COLORS["text_muted"])
            dpg.add_input_text(tag="bt_ticker", hint="AAPL", width=-1)

            dpg.add_spacer(height=10)
            dpg.add_text("Backtest Period", color=COLORS["text_muted"])
            dpg.add_combo(
                tag="bt_period",
                items=["3mo", "6mo", "1y", "2y"],
                default_value="1y",
                width=-1
            )

            dpg.add_spacer(height=10)
            dpg.add_text("Initial Capital", color=COLORS["text_muted"])
            dpg.add_input_float(tag="bt_capital", default_value=10000, width=-1, format="$%.2f")

            dpg.add_spacer(height=10)
            dpg.add_text("Commission (%)", color=COLORS["text_muted"])
            dpg.add_input_float(tag="bt_commission", default_value=0.1, width=-1, format="%.2f%%")

            dpg.add_spacer(height=20)
            dpg.add_button(
                label="Run Backtest",
                width=-1,
                callback=self._on_run
            )

            dpg.add_spacer(height=25)
            dpg.add_separator()
            dpg.add_spacer(height=15)

            dpg.add_text("TRADE LOG", color=COLORS["accent"])
            dpg.add_spacer(height=10)
            dpg.add_text("No trades yet", tag="bt_trades", color=COLORS["text_muted"])

    def _create_results(self):
        """Create results visualization panel."""
        with dpg.child_window(width=-1, height=-1, border=True):
            dpg.add_text("EQUITY CURVE", color=COLORS["text_secondary"])
            dpg.add_spacer(height=10)

            with dpg.plot(tag="equity_chart", height=300, width=-1):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="", tag="eq_x", time=True)
                dpg.add_plot_axis(dpg.mvYAxis, label="Portfolio Value ($)", tag="eq_y")

            dpg.add_spacer(height=20)

            dpg.add_text("PERFORMANCE METRICS", color=COLORS["text_secondary"])
            dpg.add_spacer(height=10)

            with dpg.group(horizontal=True):
                self._create_metric_card("Total Return", "bt_return", "--")
                dpg.add_spacer(width=10)
                self._create_metric_card("Sharpe Ratio", "bt_sharpe", "--")
                dpg.add_spacer(width=10)
                self._create_metric_card("Max Drawdown", "bt_drawdown", "--")
                dpg.add_spacer(width=10)
                self._create_metric_card("Win Rate", "bt_winrate", "--")
                dpg.add_spacer(width=10)
                self._create_metric_card("Total Trades", "bt_trades_count", "--")

    def _create_metric_card(self, title: str, tag: str, default: str):
        """Create a metric display card."""
        with dpg.child_window(width=140, height=80, border=True):
            dpg.add_text(title.upper(), color=COLORS["text_secondary"])
            dpg.add_spacer(height=5)
            dpg.add_text(default, tag=tag, color=COLORS["text_muted"])

    def _on_run(self, sender=None, data=None):
        """Handle backtest run."""
        print("Running backtest...")
