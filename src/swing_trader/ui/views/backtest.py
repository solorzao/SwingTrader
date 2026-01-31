import dearpygui.dearpygui as dpg
import threading
import numpy as np
from ..theme import COLORS
from ...data.fetcher import StockDataFetcher
from ...features.indicators import TechnicalIndicators
from ...backtest.engine import BacktestEngine


class BacktestView:
    """Backtesting results view."""

    def __init__(self, parent: str | int):
        self.parent = parent
        self.fetcher = StockDataFetcher()
        self.indicators = TechnicalIndicators()
        self.engine = BacktestEngine()
        self._is_running = False
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
            dpg.add_input_text(tag="bt_ticker", default_value="AAPL", width=-1, uppercase=True)

            dpg.add_spacer(height=10)
            dpg.add_text("Backtest Period", color=COLORS["text_muted"])
            dpg.add_combo(
                tag="bt_period",
                items=["3mo", "6mo", "1y", "2y", "5y"],
                default_value="1y",
                width=-1
            )

            dpg.add_spacer(height=10)
            dpg.add_text("Initial Capital", color=COLORS["text_muted"])
            dpg.add_input_float(tag="bt_capital", default_value=10000, width=-1, format="$%.2f")

            dpg.add_spacer(height=10)
            dpg.add_text("Commission (%)", color=COLORS["text_muted"])
            dpg.add_input_float(tag="bt_commission", default_value=0.1, width=-1, format="%.2f%%")

            dpg.add_spacer(height=10)
            dpg.add_text("Signal Threshold", color=COLORS["text_muted"])
            dpg.add_slider_float(
                tag="bt_threshold",
                default_value=0.02,
                min_value=0.01,
                max_value=0.10,
                format="%.1f%%",
                width=-1
            )

            dpg.add_spacer(height=20)
            dpg.add_button(
                label="Run Backtest",
                tag="bt_run_btn",
                width=-1,
                callback=self._on_run
            )

            dpg.add_spacer(height=10)
            dpg.add_text("", tag="bt_status", color=COLORS["text_muted"], wrap=260)

            dpg.add_spacer(height=25)
            dpg.add_separator()
            dpg.add_spacer(height=15)

            dpg.add_text("TRADE LOG", color=COLORS["accent"])
            dpg.add_spacer(height=10)
            self._trades_container = dpg.add_child_window(height=200, border=True, tag="bt_trades_container")
            with dpg.group(parent=self._trades_container):
                dpg.add_text("Run a backtest to see trades", color=COLORS["text_muted"])

    def _create_results(self):
        """Create results visualization panel."""
        with dpg.child_window(width=-1, height=-1, border=True):
            dpg.add_text("EQUITY CURVE", color=COLORS["text_secondary"])
            dpg.add_spacer(height=10)

            with dpg.plot(tag="equity_chart", height=300, width=-1):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="Day", tag="eq_x")
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

            dpg.add_spacer(height=20)

            # Drawdown chart
            dpg.add_text("DRAWDOWN", color=COLORS["text_secondary"])
            dpg.add_spacer(height=10)

            with dpg.plot(tag="drawdown_chart", height=150, width=-1):
                dpg.add_plot_axis(dpg.mvXAxis, label="Day", tag="dd_x")
                dpg.add_plot_axis(dpg.mvYAxis, label="Drawdown %", tag="dd_y")

    def _create_metric_card(self, title: str, tag: str, default: str):
        """Create a metric display card."""
        with dpg.child_window(width=140, height=80, border=True):
            dpg.add_text(title.upper(), color=COLORS["text_secondary"])
            dpg.add_spacer(height=5)
            dpg.add_text(default, tag=tag, color=COLORS["text_muted"])

    def _on_run(self, sender=None, data=None):
        """Handle backtest run."""
        if self._is_running:
            return

        self._is_running = True
        dpg.configure_item("bt_run_btn", enabled=False)
        dpg.set_value("bt_status", "Running backtest...")

        def run_backtest():
            try:
                ticker = dpg.get_value("bt_ticker").strip().upper()
                period = dpg.get_value("bt_period")
                initial_capital = dpg.get_value("bt_capital")
                commission = dpg.get_value("bt_commission") / 100  # Convert to decimal
                threshold = dpg.get_value("bt_threshold")

                if not ticker:
                    dpg.set_value("bt_status", "Enter a ticker symbol")
                    return

                # Fetch data
                dpg.set_value("bt_status", f"Fetching {ticker} data...")
                data = self.fetcher.fetch(ticker, period=period)

                if data is None or data.empty:
                    dpg.set_value("bt_status", f"No data found for {ticker}")
                    return

                # Add indicators
                data = self.indicators.add_all(data)

                # Generate signals for backtest
                dpg.set_value("bt_status", "Generating signals...")
                signals = self._generate_backtest_signals(data, threshold)

                # Run backtest
                dpg.set_value("bt_status", "Running backtest simulation...")
                result = self.engine.run(
                    data=data,
                    signals=signals,
                    initial_capital=initial_capital,
                    commission=commission
                )

                # Update UI with results
                self._update_results(result, data, ticker)
                dpg.set_value("bt_status", f"Backtest complete for {ticker}")

            except Exception as e:
                dpg.set_value("bt_status", f"Error: {str(e)[:50]}")
                import traceback
                traceback.print_exc()
            finally:
                self._is_running = False
                dpg.configure_item("bt_run_btn", enabled=True)

        thread = threading.Thread(target=run_backtest, daemon=True)
        thread.start()

    def _generate_backtest_signals(self, data, threshold):
        """Generate signals for backtesting based on technical indicators."""
        import pandas as pd
        signals = pd.Series(index=data.index, data=0)  # Default HOLD

        # Check for trained models first
        from pathlib import Path
        import joblib
        models_dir = Path("models")

        model_loaded = False
        for model_file in ["random_forest.joblib", "xgboost.joblib"]:
            model_path = models_dir / model_file
            if model_path.exists():
                try:
                    model = joblib.load(model_path)
                    feature_cols = [c for c in data.columns if c not in
                                   ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
                    if feature_cols:
                        X = data[feature_cols].dropna()
                        predictions = model.predict(X.values)
                        signals.loc[X.index] = predictions
                        model_loaded = True
                        break
                except Exception:
                    continue

        # Fallback to indicator-based signals if no model
        if not model_loaded:
            if 'RSI' in data.columns and 'MACD' in data.columns:
                for i in range(len(data)):
                    rsi = data['RSI'].iloc[i]
                    macd = data['MACD'].iloc[i] if 'MACD' in data.columns else 0
                    macd_signal = data['MACD_Signal'].iloc[i] if 'MACD_Signal' in data.columns else 0

                    if pd.notna(rsi) and pd.notna(macd):
                        # Buy signal: RSI oversold and MACD crossover
                        if rsi < 30 and macd > macd_signal:
                            signals.iloc[i] = 1
                        # Sell signal: RSI overbought and MACD crossunder
                        elif rsi > 70 and macd < macd_signal:
                            signals.iloc[i] = -1

        return signals

    def _update_results(self, result, data, ticker):
        """Update the UI with backtest results."""
        try:
            # Update metric cards
            return_color = COLORS["buy"] if result.total_return > 0 else COLORS["sell"]
            dpg.set_value("bt_return", f"{result.total_return:.1%}")
            dpg.configure_item("bt_return", color=return_color)

            dpg.set_value("bt_sharpe", f"{result.sharpe_ratio:.2f}")
            sharpe_color = COLORS["buy"] if result.sharpe_ratio > 1 else COLORS["warning"] if result.sharpe_ratio > 0 else COLORS["sell"]
            dpg.configure_item("bt_sharpe", color=sharpe_color)

            dpg.set_value("bt_drawdown", f"{result.max_drawdown:.1%}")
            dpg.configure_item("bt_drawdown", color=COLORS["sell"])

            dpg.set_value("bt_winrate", f"{result.win_rate:.1%}")
            winrate_color = COLORS["buy"] if result.win_rate > 0.5 else COLORS["warning"]
            dpg.configure_item("bt_winrate", color=winrate_color)

            dpg.set_value("bt_trades_count", str(result.total_trades))

            # Update equity curve
            self._update_equity_chart(result.equity_curve, ticker)

            # Update drawdown chart
            self._update_drawdown_chart(result.equity_curve)

            # Update trade log
            self._update_trade_log(result.trades)

        except Exception as e:
            print(f"Results update error: {e}")

    def _update_equity_chart(self, equity_curve, ticker):
        """Update the equity curve chart."""
        try:
            # Clear existing series
            if dpg.does_item_exist("equity_line"):
                dpg.delete_item("equity_line")

            x_data = list(range(len(equity_curve)))
            y_data = equity_curve.tolist() if hasattr(equity_curve, 'tolist') else list(equity_curve)

            dpg.add_line_series(
                x_data, y_data,
                label=f"{ticker} Portfolio",
                parent="eq_y",
                tag="equity_line"
            )

            dpg.fit_axis_data("eq_x")
            dpg.fit_axis_data("eq_y")

        except Exception as e:
            print(f"Equity chart error: {e}")

    def _update_drawdown_chart(self, equity_curve):
        """Update the drawdown chart."""
        try:
            # Clear existing series
            if dpg.does_item_exist("drawdown_line"):
                dpg.delete_item("drawdown_line")

            # Calculate drawdown
            equity = np.array(equity_curve)
            running_max = np.maximum.accumulate(equity)
            drawdown = (equity - running_max) / running_max * 100

            x_data = list(range(len(drawdown)))
            y_data = drawdown.tolist()

            dpg.add_line_series(
                x_data, y_data,
                label="Drawdown",
                parent="dd_y",
                tag="drawdown_line"
            )

            dpg.fit_axis_data("dd_x")
            dpg.fit_axis_data("dd_y")

        except Exception as e:
            print(f"Drawdown chart error: {e}")

    def _update_trade_log(self, trades):
        """Update the trade log."""
        dpg.delete_item("bt_trades_container", children_only=True)

        if not trades:
            with dpg.group(parent="bt_trades_container"):
                dpg.add_text("No trades executed", color=COLORS["text_muted"])
            return

        with dpg.group(parent="bt_trades_container"):
            with dpg.table(header_row=True, borders_innerH=True, borders_outerH=True,
                          borders_innerV=True, borders_outerV=True, row_background=True,
                          scrollY=True):
                dpg.add_table_column(label="Date", width_fixed=True, init_width_or_weight=80)
                dpg.add_table_column(label="Type", width_fixed=True, init_width_or_weight=50)
                dpg.add_table_column(label="Price", width_fixed=True, init_width_or_weight=70)
                dpg.add_table_column(label="P/L", width_fixed=True, init_width_or_weight=60)

                for trade in trades[-20:]:  # Show last 20 trades
                    with dpg.table_row():
                        date_str = trade.get('date', '--')
                        if hasattr(date_str, 'strftime'):
                            date_str = date_str.strftime('%m/%d')
                        dpg.add_text(str(date_str)[:10])

                        trade_type = trade.get('type', '--')
                        type_color = COLORS["buy"] if trade_type == "BUY" else COLORS["sell"]
                        dpg.add_text(trade_type, color=type_color)

                        price = trade.get('price', 0)
                        dpg.add_text(f"${price:.2f}")

                        pnl = trade.get('pnl', 0)
                        pnl_color = COLORS["buy"] if pnl > 0 else COLORS["sell"] if pnl < 0 else COLORS["text_muted"]
                        pnl_str = f"+${pnl:.0f}" if pnl > 0 else f"-${abs(pnl):.0f}" if pnl < 0 else "--"
                        dpg.add_text(pnl_str, color=pnl_color)
