import sys
import numpy as np

# Pre-import torch in main thread to avoid DLL loading issues in worker threads
try:
    import torch
except ImportError:
    pass

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QLineEdit, QPushButton, QComboBox, QSlider,
    QCheckBox, QProgressBar, QTableWidget, QTableWidgetItem, QGroupBox,
    QSpinBox, QDoubleSpinBox, QHeaderView, QSplitter, QFrame, QScrollArea,
    QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette


class Colors:
    BG_DARK = "#0D1117"
    BG_MEDIUM = "#161B22"
    BG_LIGHT = "#21262D"
    TEXT_PRIMARY = "#E6EDF3"
    TEXT_SECONDARY = "#8B949E"
    TEXT_MUTED = "#6E7681"
    BUY = "#00D4AA"
    SELL = "#FF6B6B"
    ACCENT = "#FFD93D"
    INTERACTIVE = "#58A6FF"
    WARNING = "#FFC107"
    BORDER = "#30363D"


STYLESHEET = """
    QMainWindow, QWidget {
        background-color: #0D1117;
        color: #E6EDF3;
    }
    QTabWidget::pane {
        border: 1px solid #30363D;
        background-color: #0D1117;
    }
    QTabBar::tab {
        background-color: #161B22;
        color: #8B949E;
        padding: 10px 20px;
        border: 1px solid #30363D;
        border-bottom: none;
    }
    QTabBar::tab:selected {
        background-color: #21262D;
        color: #E6EDF3;
    }
    QGroupBox {
        border: 1px solid #30363D;
        border-radius: 4px;
        margin-top: 10px;
        padding-top: 10px;
        background-color: #161B22;
    }
    QGroupBox::title {
        color: #FFD93D;
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
    }
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
        background-color: #21262D;
        border: 1px solid #30363D;
        border-radius: 4px;
        padding: 8px;
        color: #E6EDF3;
    }
    QPushButton {
        background-color: #238636;
        border: none;
        border-radius: 4px;
        padding: 10px 20px;
        color: white;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #2EA043;
    }
    QPushButton:disabled {
        background-color: #21262D;
        color: #6E7681;
    }
    QProgressBar {
        border: 1px solid #30363D;
        border-radius: 4px;
        background-color: #21262D;
        text-align: center;
    }
    QProgressBar::chunk {
        background-color: #58A6FF;
        border-radius: 3px;
    }
    QTableWidget {
        background-color: #161B22;
        border: 1px solid #30363D;
        gridline-color: #30363D;
    }
    QTableWidget::item {
        padding: 5px;
    }
    QHeaderView::section {
        background-color: #21262D;
        color: #8B949E;
        padding: 8px;
        border: none;
        border-right: 1px solid #30363D;
        border-bottom: 1px solid #30363D;
    }
    QSlider::groove:horizontal {
        border: 1px solid #30363D;
        height: 8px;
        background: #21262D;
        border-radius: 4px;
    }
    QSlider::handle:horizontal {
        background: #58A6FF;
        width: 18px;
        margin: -5px 0;
        border-radius: 9px;
    }
    QCheckBox {
        color: #E6EDF3;
    }
    QLabel {
        color: #E6EDF3;
    }
"""


def create_chart_canvas(parent=None, width=5, height=4):
    """Create a matplotlib chart canvas with lazy loading to avoid circular imports."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib
    matplotlib.use('QtAgg')

    fig = Figure(figsize=(width, height), facecolor='#0D1117')
    axes = fig.add_subplot(111)
    axes.set_facecolor('#0D1117')
    axes.tick_params(colors='#8B949E')
    for spine in axes.spines.values():
        spine.set_color('#30363D')

    canvas = FigureCanvas(fig)
    canvas.fig = fig
    canvas.axes = axes
    canvas.setMinimumHeight(250)
    return canvas


class WorkerThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str, int)  # status message, progress percentage

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            # Pass progress callback if the function accepts it
            result = self.func(*self.args, progress_callback=self.progress.emit, **self.kwargs)
            self.finished.emit(result)
        except TypeError:
            # Function doesn't accept progress_callback
            try:
                result = self.func(*self.args, **self.kwargs)
                self.finished.emit(result)
            except Exception as e:
                import traceback
                error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                self.error.emit(error_msg)
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)


class SignalsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self._current_model = None  # Set by global selector
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)

        left_panel = QWidget()
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout(left_panel)

        ticker_group = QGroupBox("TICKER ANALYSIS")
        ticker_layout = QVBoxLayout(ticker_group)

        ticker_layout.addWidget(QLabel("Symbol"))
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("AAPL")
        ticker_layout.addWidget(self.ticker_input)

        ticker_layout.addWidget(QLabel("Period"))
        self.period_combo = QComboBox()
        self.period_combo.addItems(["1mo", "3mo", "6mo", "1y", "2y"])
        self.period_combo.setCurrentText("6mo")
        ticker_layout.addWidget(self.period_combo)

        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self.on_analyze)
        ticker_layout.addWidget(self.analyze_btn)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #6E7681;")
        ticker_layout.addWidget(self.status_label)

        left_layout.addWidget(ticker_group)

        scan_group = QGroupBox("QUICK SCAN")
        scan_layout = QVBoxLayout(scan_group)

        scan_layout.addWidget(QLabel("Stock List"))
        self.stocklist_combo = QComboBox()
        self.stocklist_combo.addItems(["Top 10 Tech", "S&P 50", "S&P 100"])
        scan_layout.addWidget(self.stocklist_combo)

        scan_layout.addWidget(QLabel("Filter"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All Signals", "BUY Only", "SELL Only"])
        scan_layout.addWidget(self.filter_combo)

        self.scan_btn = QPushButton("Scan")
        self.scan_btn.clicked.connect(self.on_scan)
        scan_layout.addWidget(self.scan_btn)

        self.scan_progress = QProgressBar()
        self.scan_progress.setVisible(False)
        scan_layout.addWidget(self.scan_progress)

        left_layout.addWidget(scan_group)
        left_layout.addStretch()

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        chart_group = QGroupBox("PRICE CHART")
        chart_layout = QVBoxLayout(chart_group)
        self.chart = create_chart_canvas(self, width=8, height=4)
        chart_layout.addWidget(self.chart)
        right_layout.addWidget(chart_group)

        results_group = QGroupBox("SCAN RESULTS")
        results_layout = QVBoxLayout(results_group)
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(9)
        self.results_table.setHorizontalHeaderLabels([
            "Ticker", "Signal", "BUY%", "HOLD%", "SELL%", "Price", "RSI", "MACD", "Trend"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setStyleSheet("""
            QTableWidget {
                font-size: 11px;
            }
            QHeaderView::section {
                font-size: 10px;
            }
        """)
        results_layout.addWidget(self.results_table)
        right_layout.addWidget(results_group)

        layout.addWidget(left_panel)
        layout.addWidget(right_panel, 1)

    def on_analyze(self):
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            self.status_label.setText("Enter a ticker symbol")
            return

        self.status_label.setText(f"Analyzing {ticker}...")
        self.analyze_btn.setEnabled(False)

        # Use model from global selector
        model_name = self._current_model
        period = self.period_combo.currentText()

        def analyze():
            from ..data.fetcher import StockDataFetcher
            from ..features.indicators import TechnicalIndicators
            from ..services import ModelRegistry

            fetcher = StockDataFetcher()
            indicators = TechnicalIndicators()
            data = fetcher.fetch(ticker, period=period)
            if data is not None and not data.empty:
                data = indicators.add_all(data)

            signal_info = None
            if model_name and data is not None and not data.empty:
                # Load model and generate predictions
                registry = ModelRegistry()
                try:
                    model = registry.get_model(model_name)

                    # Prepare features (exclude non-feature columns)
                    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'label']
                    feature_cols = [c for c in data.columns if c not in exclude_cols]
                    X = data[feature_cols].dropna()

                    if not X.empty:
                        # Generate predictions
                        predictions = model.predict(X)

                        # Handle LSTM shorter output due to sequence creation
                        if hasattr(model, 'sequence_length') and len(predictions) < len(X):
                            seq_len = model.sequence_length
                            # Align predictions with data index
                            pred_index = X.index[seq_len-1:]
                            data.loc[pred_index, 'signal'] = predictions
                        else:
                            data.loc[X.index, 'signal'] = predictions

                        # Get latest prediction info
                        latest_pred = predictions[-1]
                        signal_map = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
                        signal = signal_map.get(latest_pred, 'HOLD')

                        # Try to get probabilities for confidence
                        try:
                            probas = model.predict_proba(X)
                            if probas is not None and len(probas) > 0:
                                latest_proba = probas[-1]
                                confidence = max(latest_proba)
                                probabilities = {'SELL': latest_proba[0], 'HOLD': latest_proba[1], 'BUY': latest_proba[2]}
                            else:
                                confidence = 0.5
                                probabilities = None
                        except:
                            confidence = 0.5
                            probabilities = None

                        signal_info = {
                            'signal': signal,
                            'confidence': confidence,
                            'probabilities': probabilities
                        }
                except Exception as e:
                    print(f"Model prediction error: {e}")

            return data, ticker, signal_info

        self.worker = WorkerThread(analyze)
        self.worker.finished.connect(self.on_analyze_complete)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_analyze_complete(self, result):
        data, ticker, signal_info = result
        self.analyze_btn.setEnabled(True)
        if data is None or data.empty:
            self.status_label.setText(f"No data for {ticker}")
            return

        # Update status with signal info
        if signal_info:
            signal = signal_info['signal']
            confidence = signal_info['confidence']
            color_map = {'BUY': '#00D4AA', 'SELL': '#FF6B6B', 'HOLD': '#8B949E'}
            color = color_map.get(signal, '#8B949E')
            self.status_label.setText(f"{ticker}: {signal} ({confidence:.0%})")
            self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        else:
            self.status_label.setText(f"Analysis complete for {ticker}")
            self.status_label.setStyleSheet("color: #6E7681;")

        self.chart.axes.clear()
        self.chart.axes.plot(data.index, data['close'], color='#58A6FF', label='Close')
        if 'sma_20' in data.columns:
            self.chart.axes.plot(data.index, data['sma_20'], color='#FFD93D', label='SMA 20', alpha=0.7)

        # Add signal markers if available
        if 'signal' in data.columns:
            buy_mask = data['signal'] == 1
            sell_mask = data['signal'] == -1
            if buy_mask.any():
                self.chart.axes.scatter(data.index[buy_mask], data.loc[buy_mask, 'close'],
                                       marker='^', color='#00D4AA', s=100, label='BUY', zorder=5)
            if sell_mask.any():
                self.chart.axes.scatter(data.index[sell_mask], data.loc[sell_mask, 'close'],
                                       marker='v', color='#FF6B6B', s=100, label='SELL', zorder=5)

        self.chart.axes.legend(facecolor='#0D1117', edgecolor='#30363D', labelcolor='#E6EDF3')
        self.chart.axes.set_facecolor('#0D1117')
        self.chart.draw()

    # S&P 500 stock lists by market cap (as of 2024)
    SP500_TOP_10 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V"]

    SP500_TOP_50 = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V",
        "UNH", "XOM", "JNJ", "WMT", "MA", "PG", "CVX", "HD", "ABBV", "MRK",
        "LLY", "KO", "PEP", "COST", "BAC", "AVGO", "TMO", "MCD", "CSCO", "PFE",
        "ACN", "ABT", "CRM", "DHR", "CMCSA", "NKE", "ORCL", "VZ", "TXN", "ADBE",
        "WFC", "NEE", "PM", "RTX", "UPS", "MS", "INTC", "BMY", "QCOM", "UNP"
    ]

    SP500_TOP_100 = SP500_TOP_50 + [
        "HON", "LOW", "T", "SPGI", "IBM", "SCHW", "DE", "GS", "ELV", "AMD",
        "AMAT", "CAT", "AXP", "LMT", "INTU", "BKNG", "GILD", "BLK", "MDLZ", "SYK",
        "ADI", "ISRG", "TJX", "AMGN", "ADP", "VRTX", "CI", "MMC", "CB", "PLD",
        "MO", "ZTS", "TMUS", "SO", "DUK", "REGN", "CME", "CL", "SLB", "BDX",
        "NOC", "PGR", "EOG", "ITW", "BSX", "FISV", "CSX", "AON", "WM", "ICE"
    ]

    def on_scan(self):
        self.status_label.setText("Scanning...")
        self.scan_btn.setEnabled(False)
        self.scan_progress.setVisible(True)
        self.scan_progress.setValue(0)

        model_name = self._current_model
        filter_type = self.filter_combo.currentText()
        stock_list = self.stocklist_combo.currentText()

        def scan(progress_callback=None):
            from ..data.fetcher import StockDataFetcher
            from ..features.indicators import TechnicalIndicators
            from ..services import ModelRegistry

            # Select ticker list based on choice
            if stock_list == "S&P 50":
                tickers = SignalsTab.SP500_TOP_50
            elif stock_list == "S&P 100":
                tickers = SignalsTab.SP500_TOP_100
            else:  # Top 10 Tech
                tickers = SignalsTab.SP500_TOP_10
            fetcher = StockDataFetcher()
            indicators = TechnicalIndicators()

            model = None
            if model_name:
                registry = ModelRegistry()
                model = registry.get_model(model_name)

            results = []
            for i, ticker in enumerate(tickers):
                if progress_callback:
                    progress_callback(f"Scanning {ticker}...", int((i / len(tickers)) * 100))

                try:
                    data = fetcher.fetch(ticker, period="6mo")
                    if data is None or data.empty:
                        continue
                    data = indicators.add_all(data)
                    latest = data.iloc[-1]

                    # Get indicator values
                    rsi = latest.get('rsi_14', 50)
                    macd = latest.get('macd', 0)
                    macd_signal_val = latest.get('macd_signal', 0)
                    sma_20 = latest.get('sma_20', 0)
                    price = latest['close']

                    # Determine trend (price vs SMA)
                    trend = "↑ Up" if price > sma_20 else "↓ Down"

                    if model:
                        # Use ML model
                        predictions = model.predict(data)
                        probas = model.predict_proba(data)

                        # Get latest prediction
                        latest_pred = predictions[-1]
                        latest_proba = probas[-1]

                        signal = "BUY" if latest_pred == 1 else "SELL" if latest_pred == -1 else "HOLD"
                        # Probabilities: [SELL, HOLD, BUY] based on label mapping
                        prob_sell = float(latest_proba[0])
                        prob_hold = float(latest_proba[1])
                        prob_buy = float(latest_proba[2])
                    else:
                        # Fallback to RSI/MACD
                        if rsi < 30 and macd > macd_signal_val:
                            signal = "BUY"
                            prob_buy, prob_hold, prob_sell = 0.7, 0.2, 0.1
                        elif rsi > 70 and macd < macd_signal_val:
                            signal = "SELL"
                            prob_buy, prob_hold, prob_sell = 0.1, 0.2, 0.7
                        else:
                            signal = "HOLD"
                            prob_buy, prob_hold, prob_sell = 0.2, 0.6, 0.2

                    # Apply filter
                    if filter_type == "BUY Only" and signal != "BUY":
                        continue
                    if filter_type == "SELL Only" and signal != "SELL":
                        continue

                    results.append({
                        "ticker": ticker,
                        "signal": signal,
                        "prob_buy": prob_buy,
                        "prob_hold": prob_hold,
                        "prob_sell": prob_sell,
                        "price": price,
                        "rsi": rsi,
                        "macd": macd - macd_signal_val,  # MACD histogram value
                        "trend": trend
                    })
                except Exception as e:
                    print(f"Error scanning {ticker}: {e}")
                    continue

            return results

        self.worker = WorkerThread(scan)
        self.worker.finished.connect(self.on_scan_complete)
        self.worker.error.connect(self.on_error)
        self.worker.progress.connect(self._on_scan_progress)
        self.worker.start()

    def _on_scan_progress(self, msg, pct):
        self.status_label.setText(msg)
        self.scan_progress.setValue(pct)

    def on_scan_complete(self, results):
        self.scan_btn.setEnabled(True)
        self.scan_progress.setVisible(False)
        self.status_label.setText(f"Found {len(results)} signals")
        self.results_table.setRowCount(len(results))

        # Sort by highest relevant probability
        def sort_key(r):
            if r['signal'] == 'BUY':
                return r['prob_buy']
            elif r['signal'] == 'SELL':
                return r['prob_sell']
            return r['prob_hold']

        for i, r in enumerate(sorted(results, key=sort_key, reverse=True)):
            # Ticker
            self.results_table.setItem(i, 0, QTableWidgetItem(r['ticker']))

            # Signal with color
            signal_item = QTableWidgetItem(r['signal'])
            color = "#3FB950" if r['signal'] == "BUY" else "#F85149" if r['signal'] == "SELL" else "#8B949E"
            signal_item.setForeground(QColor(color))
            self.results_table.setItem(i, 1, signal_item)

            # Probability breakdown with color intensity
            buy_item = QTableWidgetItem(f"{r['prob_buy']:.0%}")
            buy_item.setForeground(QColor("#3FB950"))
            self.results_table.setItem(i, 2, buy_item)

            hold_item = QTableWidgetItem(f"{r['prob_hold']:.0%}")
            hold_item.setForeground(QColor("#8B949E"))
            self.results_table.setItem(i, 3, hold_item)

            sell_item = QTableWidgetItem(f"{r['prob_sell']:.0%}")
            sell_item.setForeground(QColor("#F85149"))
            self.results_table.setItem(i, 4, sell_item)

            # Price
            self.results_table.setItem(i, 5, QTableWidgetItem(f"${r['price']:.2f}"))

            # RSI with color (oversold < 30, overbought > 70)
            rsi_item = QTableWidgetItem(f"{r['rsi']:.1f}")
            if r['rsi'] < 30:
                rsi_item.setForeground(QColor("#3FB950"))  # Oversold = potential buy
            elif r['rsi'] > 70:
                rsi_item.setForeground(QColor("#F85149"))  # Overbought = potential sell
            self.results_table.setItem(i, 6, rsi_item)

            # MACD histogram
            macd_item = QTableWidgetItem(f"{r['macd']:.3f}")
            macd_color = "#3FB950" if r['macd'] > 0 else "#F85149"
            macd_item.setForeground(QColor(macd_color))
            self.results_table.setItem(i, 7, macd_item)

            # Trend
            trend_item = QTableWidgetItem(r['trend'])
            trend_color = "#3FB950" if "Up" in r['trend'] else "#F85149"
            trend_item.setForeground(QColor(trend_color))
            self.results_table.setItem(i, 8, trend_item)

    def on_error(self, error):
        self.analyze_btn.setEnabled(True)
        self.scan_btn.setEnabled(True)
        self.scan_progress.setVisible(False)
        self.status_label.setText(f"Error: {error[:40]}")

    def set_model(self, model_name: str):
        """Set the active model (called from global selector)."""
        self._current_model = model_name if model_name else None


class TrainingTab(QWidget):
    training_complete = pyqtSignal()  # Emitted when training finishes successfully

    def __init__(self):
        super().__init__()
        self.worker = None
        self.mlflow_process = None
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)

        # Left panel with scroll area for all controls
        left_scroll = QScrollArea()
        left_scroll.setFixedWidth(400)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        train_group = QGroupBox("MODEL TRAINING")
        train_layout = QVBoxLayout(train_group)

        train_layout.addWidget(QLabel("Model Type"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Random Forest", "XGBoost", "LSTM"])
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        train_layout.addWidget(self.model_combo)

        train_layout.addWidget(QLabel("Training Symbols"))
        self.tickers_input = QLineEdit("AAPL, MSFT, GOOGL")
        train_layout.addWidget(self.tickers_input)

        train_layout.addWidget(QLabel("Data Period"))
        self.period_combo = QComboBox()
        self.period_combo.addItems(["3mo", "6mo", "1y", "2y", "5y"])
        self.period_combo.setCurrentText("1y")
        train_layout.addWidget(self.period_combo)

        left_layout.addWidget(train_group)

        # Feature Configuration group
        features_group = QGroupBox("FEATURES")
        features_layout = QVBoxLayout(features_group)

        # Feature checkboxes
        self.feature_checks = {}
        feature_row1 = QHBoxLayout()
        for name in ["SMA", "EMA", "RSI"]:
            cb = QCheckBox(name)
            cb.setChecked(True)
            self.feature_checks[name] = cb
            feature_row1.addWidget(cb)
        features_layout.addLayout(feature_row1)

        feature_row2 = QHBoxLayout()
        for name in ["MACD", "Bollinger", "ATR"]:
            cb = QCheckBox(name)
            cb.setChecked(True)
            self.feature_checks[name] = cb
            feature_row2.addWidget(cb)
        features_layout.addLayout(feature_row2)

        feature_row3 = QHBoxLayout()
        for name in ["OBV", "Stochastic"]:
            cb = QCheckBox(name)
            cb.setChecked(True)
            self.feature_checks[name] = cb
            feature_row3.addWidget(cb)
        features_layout.addLayout(feature_row3)

        # Feature parameters
        features_layout.addWidget(QLabel("RSI Period"))
        self.rsi_period_spin = QSpinBox()
        self.rsi_period_spin.setRange(5, 30)
        self.rsi_period_spin.setValue(14)
        features_layout.addWidget(self.rsi_period_spin)

        features_layout.addWidget(QLabel("SMA Periods (comma-sep)"))
        self.sma_periods_input = QLineEdit("10, 20, 50")
        features_layout.addWidget(self.sma_periods_input)

        features_layout.addWidget(QLabel("EMA Periods (comma-sep)"))
        self.ema_periods_input = QLineEdit("12, 26")
        features_layout.addWidget(self.ema_periods_input)

        left_layout.addWidget(features_group)

        # Hyperparameters group
        self.hyperparam_group = QGroupBox("HYPERPARAMETERS")
        self.hyperparam_layout = QVBoxLayout(self.hyperparam_group)
        self.setup_rf_hyperparams()
        left_layout.addWidget(self.hyperparam_group)

        # Auto-tune group
        tune_group = QGroupBox("AUTO-TUNE (OPTUNA)")
        tune_layout = QVBoxLayout(tune_group)

        self.autotune_check = QCheckBox("Auto-tune hyperparameters")
        self.autotune_check.setToolTip("Use Optuna to automatically find optimal hyperparameters")
        self.autotune_check.toggled.connect(self.on_autotune_toggled)
        tune_layout.addWidget(self.autotune_check)

        trials_layout = QHBoxLayout()
        trials_layout.addWidget(QLabel("Trials:"))
        self.n_trials_spin = QSpinBox()
        self.n_trials_spin.setRange(10, 200)
        self.n_trials_spin.setValue(30)
        self.n_trials_spin.setSingleStep(10)
        self.n_trials_spin.setEnabled(False)
        trials_layout.addWidget(self.n_trials_spin)
        trials_layout.addStretch()
        tune_layout.addLayout(trials_layout)

        self.tune_status = QLabel("Manual hyperparameters will be used")
        self.tune_status.setStyleSheet("color: #6E7681; font-size: 11px;")
        self.tune_status.setWordWrap(True)
        tune_layout.addWidget(self.tune_status)

        left_layout.addWidget(tune_group)

        # Training buttons
        self.train_btn = QPushButton("Start Training")
        self.train_btn.clicked.connect(self.on_train)
        left_layout.addWidget(self.train_btn)

        # MLflow button
        mlflow_layout = QHBoxLayout()
        self.mlflow_btn = QPushButton("Launch MLflow UI")
        self.mlflow_btn.setStyleSheet("""
            QPushButton {
                background-color: #0D47A1;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
        """)
        self.mlflow_btn.clicked.connect(self.on_launch_mlflow)
        mlflow_layout.addWidget(self.mlflow_btn)
        left_layout.addLayout(mlflow_layout)

        status_group = QGroupBox("STATUS")
        status_layout = QVBoxLayout(status_group)
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #6E7681;")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        status_layout.addWidget(self.progress_bar)
        left_layout.addWidget(status_group)

        left_layout.addStretch()

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Metrics summary panel
        metrics_group = QGroupBox("CLASSIFICATION METRICS")
        metrics_layout = QHBoxLayout(metrics_group)
        self.metric_displays = {}
        for name in ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]:
            frame = QGroupBox(name.upper())
            frame.setFixedHeight(80)
            frame_layout = QVBoxLayout(frame)
            label = QLabel("--")
            label.setStyleSheet("color: #6E7681; font-size: 16px; font-weight: bold;")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            frame_layout.addWidget(label)
            self.metric_displays[name] = label
            metrics_layout.addWidget(frame)
        right_layout.addWidget(metrics_group)

        # Charts in horizontal layout
        charts_layout = QHBoxLayout()

        # Feature importance chart
        importance_group = QGroupBox("FEATURE IMPORTANCE")
        importance_layout = QVBoxLayout(importance_group)
        self.importance_chart = create_chart_canvas(self, width=5, height=4)
        importance_layout.addWidget(self.importance_chart)
        charts_layout.addWidget(importance_group)

        # Confusion matrix chart
        confusion_group = QGroupBox("CONFUSION MATRIX")
        confusion_layout = QVBoxLayout(confusion_group)
        self.confusion_chart = create_chart_canvas(self, width=4, height=4)
        confusion_layout.addWidget(self.confusion_chart)
        charts_layout.addWidget(confusion_group)

        right_layout.addLayout(charts_layout)

        # Add left panel to scroll area
        left_scroll.setWidget(left_panel)
        layout.addWidget(left_scroll)
        layout.addWidget(right_panel, 1)

    def clear_hyperparam_layout(self):
        # Remove Python references to widgets before deleting them
        for attr in ['n_estimators_spin', 'max_depth_spin', 'min_samples_spin',
                     'learning_rate_spin', 'hidden_size_spin', 'num_layers_spin', 'epochs_spin']:
            if hasattr(self, attr):
                delattr(self, attr)

        while self.hyperparam_layout.count():
            item = self.hyperparam_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def setup_rf_hyperparams(self):
        self.clear_hyperparam_layout()
        self.hyperparam_layout.addWidget(QLabel("Number of Trees"))
        self.n_estimators_spin = QSpinBox()
        self.n_estimators_spin.setRange(10, 500)
        self.n_estimators_spin.setValue(100)
        self.n_estimators_spin.setSingleStep(10)
        self.hyperparam_layout.addWidget(self.n_estimators_spin)

        self.hyperparam_layout.addWidget(QLabel("Max Depth (0 = unlimited)"))
        self.max_depth_spin = QSpinBox()
        self.max_depth_spin.setRange(0, 50)
        self.max_depth_spin.setValue(10)
        self.hyperparam_layout.addWidget(self.max_depth_spin)

        self.hyperparam_layout.addWidget(QLabel("Min Samples Split"))
        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setRange(2, 20)
        self.min_samples_spin.setValue(2)
        self.hyperparam_layout.addWidget(self.min_samples_spin)

    def setup_xgb_hyperparams(self):
        self.clear_hyperparam_layout()
        self.hyperparam_layout.addWidget(QLabel("Number of Rounds"))
        self.n_estimators_spin = QSpinBox()
        self.n_estimators_spin.setRange(10, 500)
        self.n_estimators_spin.setValue(100)
        self.n_estimators_spin.setSingleStep(10)
        self.hyperparam_layout.addWidget(self.n_estimators_spin)

        self.hyperparam_layout.addWidget(QLabel("Max Depth"))
        self.max_depth_spin = QSpinBox()
        self.max_depth_spin.setRange(1, 15)
        self.max_depth_spin.setValue(6)
        self.hyperparam_layout.addWidget(self.max_depth_spin)

        self.hyperparam_layout.addWidget(QLabel("Learning Rate"))
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.01, 1.0)
        self.learning_rate_spin.setValue(0.1)
        self.learning_rate_spin.setSingleStep(0.01)
        self.hyperparam_layout.addWidget(self.learning_rate_spin)

    def setup_lstm_hyperparams(self):
        self.clear_hyperparam_layout()
        self.hyperparam_layout.addWidget(QLabel("Hidden Size"))
        self.hidden_size_spin = QSpinBox()
        self.hidden_size_spin.setRange(16, 256)
        self.hidden_size_spin.setValue(64)
        self.hidden_size_spin.setSingleStep(16)
        self.hyperparam_layout.addWidget(self.hidden_size_spin)

        self.hyperparam_layout.addWidget(QLabel("Number of Layers"))
        self.num_layers_spin = QSpinBox()
        self.num_layers_spin.setRange(1, 4)
        self.num_layers_spin.setValue(2)
        self.hyperparam_layout.addWidget(self.num_layers_spin)

        self.hyperparam_layout.addWidget(QLabel("Epochs"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(5, 100)
        self.epochs_spin.setValue(30)
        self.epochs_spin.setSingleStep(5)
        self.hyperparam_layout.addWidget(self.epochs_spin)

        self.hyperparam_layout.addWidget(QLabel("Learning Rate"))
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.0001, 0.1)
        self.learning_rate_spin.setValue(0.001)
        self.learning_rate_spin.setSingleStep(0.0001)
        self.learning_rate_spin.setDecimals(4)
        self.hyperparam_layout.addWidget(self.learning_rate_spin)

    def on_model_changed(self, model_name):
        if model_name == "Random Forest":
            self.setup_rf_hyperparams()
        elif model_name == "XGBoost":
            self.setup_xgb_hyperparams()
        elif model_name == "LSTM":
            self.setup_lstm_hyperparams()

    def on_autotune_toggled(self, checked):
        """Enable/disable auto-tune controls."""
        self.n_trials_spin.setEnabled(checked)
        # Disable manual hyperparams when auto-tuning
        self.hyperparam_group.setEnabled(not checked)
        if checked:
            self.tune_status.setText("Optuna will search for optimal hyperparameters")
            self.tune_status.setStyleSheet("color: #58A6FF; font-size: 11px;")
        else:
            self.tune_status.setText("Manual hyperparameters will be used")
            self.tune_status.setStyleSheet("color: #6E7681; font-size: 11px;")

    def on_launch_mlflow(self):
        """Launch MLflow UI in browser."""
        import webbrowser
        from ..services import MLflowTracker
        from PyQt6.QtCore import QTimer

        url = MLflowTracker.get_ui_url(5000)

        if self.mlflow_process is None or self.mlflow_process.poll() is not None:
            # Not running or terminated - start it
            self.status_label.setText("Starting MLflow UI server...")
            try:
                self.mlflow_process = MLflowTracker.launch_ui(port=5000)
                self.mlflow_btn.setText("Open MLflow UI")
                self.status_label.setText(f"MLflow UI starting at {url}")

                # Open browser after a delay to let server start
                QTimer.singleShot(2000, lambda: webbrowser.open(url))
            except Exception as e:
                self.status_label.setText(f"Failed to start MLflow: {str(e)[:50]}")
                self.mlflow_process = None
        else:
            # Already running, just open browser
            webbrowser.open(url)
            self.status_label.setText(f"Opening {url}")

    def get_feature_config(self) -> dict:
        """Get current feature configuration from UI."""
        return {
            "features": {
                "sma": self.feature_checks["SMA"].isChecked(),
                "ema": self.feature_checks["EMA"].isChecked(),
                "rsi": self.feature_checks["RSI"].isChecked(),
                "macd": self.feature_checks["MACD"].isChecked(),
                "bollinger": self.feature_checks["Bollinger"].isChecked(),
                "atr": self.feature_checks["ATR"].isChecked(),
                "obv": self.feature_checks["OBV"].isChecked(),
                "stochastic": self.feature_checks["Stochastic"].isChecked(),
            },
            "params": {
                "rsi_period": self.rsi_period_spin.value(),
                "sma_periods": [int(x.strip()) for x in self.sma_periods_input.text().split(",") if x.strip()],
                "ema_periods": [int(x.strip()) for x in self.ema_periods_input.text().split(",") if x.strip()],
            }
        }

    def on_train(self):
        self.status_label.setText("Starting training...")
        self.train_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        # Capture ALL UI values before starting thread (Qt widgets can't be accessed from threads)
        model_type = self.model_combo.currentText()
        tickers_text = self.tickers_input.text()
        period = self.period_combo.currentText()
        feature_config = self.get_feature_config()

        # Auto-tune settings
        use_autotune = self.autotune_check.isChecked()
        n_trials = self.n_trials_spin.value()

        # Common params (may not exist depending on model type)
        n_estimators_spin = getattr(self, 'n_estimators_spin', None)
        n_estimators = n_estimators_spin.value() if n_estimators_spin else 100
        max_depth_spin = getattr(self, 'max_depth_spin', None)
        max_depth = max_depth_spin.value() if max_depth_spin else 10
        learning_rate_spin = getattr(self, 'learning_rate_spin', None)
        learning_rate = learning_rate_spin.value() if learning_rate_spin else 0.1

        # LSTM params (may not exist depending on model type)
        hidden_size_spin = getattr(self, 'hidden_size_spin', None)
        hidden_size = hidden_size_spin.value() if hidden_size_spin else 64
        num_layers_spin = getattr(self, 'num_layers_spin', None)
        num_layers = num_layers_spin.value() if num_layers_spin else 2
        epochs_spin = getattr(self, 'epochs_spin', None)
        epochs = epochs_spin.value() if epochs_spin else 30

        def train(progress_callback=None):
            def update_progress(msg, pct):
                if progress_callback:
                    progress_callback(msg, pct)

            from ..data.fetcher import StockDataFetcher
            from ..features.indicators import TechnicalIndicators
            from ..features.labeler import SignalLabeler
            import pandas as pd
            from pathlib import Path

            tickers = [t.strip().upper() for t in tickers_text.split(",")]
            fetcher = StockDataFetcher()
            indicators = TechnicalIndicators()
            labeler = SignalLabeler()

            update_progress("Fetching market data...", 10)
            all_data = []
            for i, ticker in enumerate(tickers):
                update_progress(f"Fetching {ticker}...", 10 + (i * 20 // len(tickers)))
                data = fetcher.fetch(ticker, period=period)
                if data is not None and not data.empty:
                    data = indicators.add_all(data, config=feature_config)
                    data['label'] = labeler.create_labels(data)
                    all_data.append(data)

            if not all_data:
                return None

            update_progress("Preparing features...", 30)
            combined = pd.concat(all_data, ignore_index=True).dropna()
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'label']
            feature_cols = [c for c in combined.columns if c not in exclude_cols]
            X = combined[feature_cols]
            y = combined['label']

            # Split data for training and validation
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Auto-tune hyperparameters if enabled
            tuned_params = {}
            if use_autotune:
                update_progress(f"Auto-tuning {model_type} ({n_trials} trials)... This may take a while", 35)
                from ..training.tuner import HyperparameterTuner
                import optuna
                optuna.logging.set_verbosity(optuna.logging.WARNING)  # Reduce Optuna console spam

                tuner = HyperparameterTuner()

                try:
                    if model_type == "Random Forest":
                        result = tuner.tune_random_forest(X_train, y_train, n_trials=n_trials)
                    elif model_type == "XGBoost":
                        use_gpu = torch.cuda.is_available()
                        result = tuner.tune_xgboost(X_train, y_train, n_trials=n_trials, use_gpu=use_gpu)
                    elif model_type == "LSTM":
                        result = tuner.tune_lstm(X_train, y_train, n_trials=n_trials)
                    else:
                        result = tuner.tune_random_forest(X_train, y_train, n_trials=n_trials)

                    tuned_params = result.get("best_params", {})
                    actual_trials = result.get("n_trials", n_trials)
                    best_score = result.get("best_value", 0)
                    update_progress(f"Tuning complete! {actual_trials} trials, best score: {best_score:.3f}", 50)
                except Exception as e:
                    import traceback
                    print(f"Tuning error: {traceback.format_exc()}")  # Log full error to console
                    update_progress(f"Tuning failed: {str(e)[:50]}. Using defaults.", 40)

            update_progress(f"Training {model_type}...", 55)

            # Create model based on selection (use tuned params if available)
            is_lstm = False
            if model_type == "Random Forest":
                from ..models.random_forest import RandomForestModel
                params = {
                    "n_estimators": tuned_params.get("n_estimators", n_estimators),
                    "max_depth": tuned_params.get("max_depth", max_depth if max_depth > 0 else None),
                    "min_samples_split": tuned_params.get("min_samples_split", 2),
                    "min_samples_leaf": tuned_params.get("min_samples_leaf", 1),
                }
                if params["max_depth"] == 0:
                    params["max_depth"] = None
                model = RandomForestModel(**params)
            elif model_type == "XGBoost":
                from ..models.xgboost_model import XGBoostModel
                use_gpu = torch.cuda.is_available()
                params = {
                    "n_estimators": tuned_params.get("n_estimators", n_estimators),
                    "max_depth": tuned_params.get("max_depth", max_depth),
                    "learning_rate": tuned_params.get("learning_rate", learning_rate),
                    "subsample": tuned_params.get("subsample", 1.0),
                    "colsample_bytree": tuned_params.get("colsample_bytree", 1.0),
                    "use_gpu": use_gpu,
                }
                model = XGBoostModel(**params)
            elif model_type == "LSTM":
                from ..models.lstm import LSTMModel
                device = "cuda" if torch.cuda.is_available() else "cpu"
                params = {
                    "hidden_size": tuned_params.get("hidden_size", hidden_size),
                    "num_layers": tuned_params.get("num_layers", num_layers),
                    "epochs": tuned_params.get("epochs", epochs),
                    "learning_rate": tuned_params.get("learning_rate", learning_rate),
                    "dropout": tuned_params.get("dropout", 0.2),
                    "device": device,
                }
                if "sequence_length" in tuned_params:
                    params["sequence_length"] = tuned_params["sequence_length"]
                if "batch_size" in tuned_params:
                    params["batch_size"] = tuned_params["batch_size"]
                model = LSTMModel(**params)
                is_lstm = True
            else:  # All Models - use Random Forest as default
                from ..models.random_forest import RandomForestModel
                md = max_depth if max_depth > 0 else None
                model = RandomForestModel(n_estimators=n_estimators, max_depth=md)

            model.fit(X_train, y_train)

            update_progress("Calculating metrics...", 80)

            # Calculate comprehensive metrics
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                confusion_matrix, roc_auc_score
            )
            import numpy as np

            predictions = model.predict(X_test)

            # LSTM predictions are shorter due to sequence creation
            if is_lstm:
                seq_len = model.sequence_length
                y_true = y_test.values[seq_len-1:]
            else:
                y_true = y_test.values

            accuracy = accuracy_score(y_true, predictions)
            precision = precision_score(y_true, predictions, average='weighted', zero_division=0)
            recall = recall_score(y_true, predictions, average='weighted', zero_division=0)
            f1 = f1_score(y_true, predictions, average='weighted', zero_division=0)

            # AUC requires probability predictions
            try:
                y_proba = model.predict_proba(X_test)
                auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            except:
                auc = None

            # Confusion matrix
            cm = confusion_matrix(y_true, predictions, labels=[-1, 0, 1])

            # Get feature importance (if available)
            try:
                importance = model.feature_importance()
                top_features = importance.head(10)
                feature_names = top_features.index.tolist()
                feature_importance = top_features.values.tolist()
            except (AttributeError, NotImplementedError):
                feature_names = feature_cols[:10]
                feature_importance = None

            update_progress("Saving model & logging to MLflow...", 95)

            # Save model with timestamp for versioning
            from datetime import datetime
            from ..services import MLflowTracker

            Path("models").mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{model_type.lower().replace(' ', '_')}_{timestamp}"
            model_path = Path(f"models/{model_filename}.joblib")

            # Save model with metrics
            model_metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auc": auc
            }
            model.save(model_path, metrics=model_metrics)

            # Log to MLflow
            try:
                tracker = MLflowTracker()
                run_name = f"{model_type}_{timestamp}"
                tracker.start_run(run_name=run_name)

                # Log parameters (use actual params from model, which may be tuned)
                params = {
                    "model_type": model_type,
                    "tickers": tickers_text,
                    "period": period,
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "n_features": len(feature_cols),
                    "auto_tuned": use_autotune,
                }
                if use_autotune:
                    params["n_trials"] = n_trials

                # Log the actual hyperparameters used (tuned or manual)
                if tuned_params:
                    for k, v in tuned_params.items():
                        params[k] = v
                else:
                    if model_type == "Random Forest":
                        params["n_estimators"] = n_estimators
                        params["max_depth"] = max_depth
                    elif model_type == "XGBoost":
                        params["n_estimators"] = n_estimators
                        params["max_depth"] = max_depth
                        params["learning_rate"] = learning_rate
                    elif model_type == "LSTM":
                        params["hidden_size"] = hidden_size
                        params["num_layers"] = num_layers
                        params["epochs"] = epochs
                        params["learning_rate"] = learning_rate

                tracker.log_params(params)

                # Log feature config
                tracker.log_feature_config(feature_config)

                # Log metrics
                metrics = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                }
                if auc is not None:
                    metrics["auc"] = auc
                tracker.log_metrics(metrics)

                # Log model artifact path
                import mlflow
                mlflow.log_artifact(str(model_path))

                tracker.end_run()
            except Exception as e:
                print(f"MLflow logging error (non-fatal): {e}")

            return {
                'samples': len(combined),
                'model_type': model_type,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'confusion_matrix': cm.tolist(),
                'feature_names': feature_names,
                'feature_importance': feature_importance,
            }

        self.worker = WorkerThread(train)
        self.worker.finished.connect(self.on_train_complete)
        self.worker.error.connect(self.on_error)
        self.worker.progress.connect(self.on_train_progress)
        self.worker.start()

    def on_train_progress(self, message, percent):
        self.status_label.setText(message)
        self.progress_bar.setValue(percent)

    def on_train_complete(self, result):
        self.train_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        if result is None or (isinstance(result, dict) and result.get('samples') is None):
            self.status_label.setText("No data collected")
            return

        samples = result['samples']
        self.status_label.setText(f"Training complete! {samples} samples processed")

        # Signal that training is complete so model list can refresh
        self.training_complete.emit()

        # Update metric displays
        self.metric_displays["Accuracy"].setText(f"{result['accuracy']:.1%}")
        self.metric_displays["Accuracy"].setStyleSheet("color: #00D4AA; font-size: 16px; font-weight: bold;")

        self.metric_displays["Precision"].setText(f"{result['precision']:.1%}")
        self.metric_displays["Precision"].setStyleSheet("color: #58A6FF; font-size: 16px; font-weight: bold;")

        self.metric_displays["Recall"].setText(f"{result['recall']:.1%}")
        self.metric_displays["Recall"].setStyleSheet("color: #58A6FF; font-size: 16px; font-weight: bold;")

        self.metric_displays["F1 Score"].setText(f"{result['f1']:.1%}")
        self.metric_displays["F1 Score"].setStyleSheet("color: #FFD93D; font-size: 16px; font-weight: bold;")

        if result['auc'] is not None:
            self.metric_displays["AUC"].setText(f"{result['auc']:.3f}")
            self.metric_displays["AUC"].setStyleSheet("color: #58A6FF; font-size: 16px; font-weight: bold;")
        else:
            self.metric_displays["AUC"].setText("N/A")

        # Plot feature importance (or show message if not available)
        self.importance_chart.axes.clear()
        feature_names = result['feature_names']
        importance = result['feature_importance']

        if importance is not None:
            y_pos = range(len(feature_names))
            self.importance_chart.axes.barh(y_pos, importance, color='#58A6FF')
            self.importance_chart.axes.set_yticks(y_pos)
            self.importance_chart.axes.set_yticklabels(feature_names, fontsize=8)
            self.importance_chart.axes.set_xlabel('Importance', color='#8B949E', fontsize=8)
            self.importance_chart.axes.invert_yaxis()
        else:
            # LSTM doesn't have feature importance
            self.importance_chart.axes.text(0.5, 0.5, f"Feature importance\nnot available for\n{result.get('model_type', 'this model')}",
                ha='center', va='center', color='#6E7681', fontsize=12, transform=self.importance_chart.axes.transAxes)
            self.importance_chart.axes.set_xticks([])
            self.importance_chart.axes.set_yticks([])

        self.importance_chart.axes.set_facecolor('#0D1117')
        self.importance_chart.axes.tick_params(colors='#8B949E', labelsize=7)
        self.importance_chart.fig.tight_layout()
        self.importance_chart.draw()

        # Plot confusion matrix
        import numpy as np
        self.confusion_chart.axes.clear()
        cm = np.array(result['confusion_matrix'])
        labels = ['SELL', 'HOLD', 'BUY']

        im = self.confusion_chart.axes.imshow(cm, cmap='Blues', aspect='auto')
        self.confusion_chart.axes.set_xticks(range(3))
        self.confusion_chart.axes.set_yticks(range(3))
        self.confusion_chart.axes.set_xticklabels(labels, fontsize=8)
        self.confusion_chart.axes.set_yticklabels(labels, fontsize=8)
        self.confusion_chart.axes.set_xlabel('Predicted', color='#8B949E', fontsize=8)
        self.confusion_chart.axes.set_ylabel('Actual', color='#8B949E', fontsize=8)
        self.confusion_chart.axes.tick_params(colors='#8B949E', labelsize=7)

        # Add text annotations
        for i in range(3):
            for j in range(3):
                color = 'white' if cm[i, j] > cm.max() / 2 else '#E6EDF3'
                self.confusion_chart.axes.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=10)

        self.confusion_chart.fig.tight_layout()
        self.confusion_chart.draw()

    def on_error(self, error):
        self.train_btn.setEnabled(True)
        # Show first line of error in status, print full error to console
        first_line = error.split('\n')[0] if '\n' in error else error
        self.status_label.setText(f"Error: {first_line[:60]}")
        print(f"Training error:\n{error}")


class BacktestTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self._current_model = None  # Set by global selector
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)

        left_panel = QWidget()
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout(left_panel)

        config_group = QGroupBox("BACKTEST CONFIG")
        config_layout = QVBoxLayout(config_group)

        config_layout.addWidget(QLabel("Symbol"))
        self.ticker_input = QLineEdit("AAPL")
        config_layout.addWidget(self.ticker_input)

        config_layout.addWidget(QLabel("Period"))
        self.period_combo = QComboBox()
        self.period_combo.addItems(["3mo", "6mo", "1y", "2y"])
        self.period_combo.setCurrentText("1y")
        config_layout.addWidget(self.period_combo)

        config_layout.addWidget(QLabel("Initial Capital"))
        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setRange(1000, 1000000)
        self.capital_spin.setValue(10000)
        self.capital_spin.setPrefix("$")
        config_layout.addWidget(self.capital_spin)

        self.run_btn = QPushButton("Run Backtest")
        self.run_btn.clicked.connect(self.on_run)
        config_layout.addWidget(self.run_btn)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #6E7681;")
        config_layout.addWidget(self.status_label)

        left_layout.addWidget(config_group)
        left_layout.addStretch()

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Performance metrics at top
        metrics_group = QGroupBox("PERFORMANCE METRICS")
        metrics_layout = QHBoxLayout(metrics_group)
        self.metric_labels = {}
        for name in ["Return", "Sharpe", "Drawdown", "Win Rate", "Trades"]:
            frame = QGroupBox(name.upper())
            frame.setFixedHeight(75)
            frame_layout = QVBoxLayout(frame)
            label = QLabel("--")
            label.setStyleSheet("color: #6E7681; font-size: 14px; font-weight: bold;")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            frame_layout.addWidget(label)
            self.metric_labels[name] = label
            metrics_layout.addWidget(frame)
        right_layout.addWidget(metrics_group)

        # Middle section: equity curve and trade history side by side
        middle_layout = QHBoxLayout()

        # Equity curve (left)
        equity_group = QGroupBox("EQUITY CURVE")
        equity_layout = QVBoxLayout(equity_group)
        self.equity_chart = create_chart_canvas(self, width=5, height=3)
        equity_layout.addWidget(self.equity_chart)
        middle_layout.addWidget(equity_group, 3)

        # Trade history (right)
        trades_group = QGroupBox("TRADE HISTORY")
        trades_layout = QVBoxLayout(trades_group)

        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(6)
        self.trades_table.setHorizontalHeaderLabels(["Type", "Entry", "Exit", "Entry $", "Exit $", "P&L"])
        self.trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.trades_table.verticalHeader().setVisible(False)
        self.trades_table.setStyleSheet("""
            QTableWidget {
                background-color: #161B22;
                border: 1px solid #30363D;
                font-size: 11px;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QHeaderView::section {
                background-color: #21262D;
                color: #8B949E;
                padding: 6px;
                border: none;
                font-size: 10px;
            }
        """)
        trades_layout.addWidget(self.trades_table)
        middle_layout.addWidget(trades_group, 2)

        right_layout.addLayout(middle_layout)

        layout.addWidget(left_panel)
        layout.addWidget(right_panel, 1)

    def set_model(self, model_name: str):
        """Set the active model (called from global selector)."""
        self._current_model = model_name if model_name else None

    def on_run(self):
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            self.status_label.setText("Enter a ticker")
            return
        self.status_label.setText("Running backtest...")
        self.run_btn.setEnabled(False)

        model_name = self._current_model
        period = self.period_combo.currentText()
        capital = self.capital_spin.value()

        def backtest(progress_callback=None):
            from ..data.fetcher import StockDataFetcher
            from ..features.indicators import TechnicalIndicators
            from ..backtest.engine import BacktestEngine
            from ..services import ModelRegistry
            import pandas as pd

            if progress_callback:
                progress_callback("Fetching data...", 20)

            fetcher = StockDataFetcher()
            indicators = TechnicalIndicators()
            data = fetcher.fetch(ticker, period=period)

            if data is None or data.empty:
                return None

            data = indicators.add_all(data)

            if progress_callback:
                progress_callback("Generating signals...", 50)

            if model_name:
                # Use ML model
                registry = ModelRegistry()
                model = registry.get_model(model_name)
                predictions = model.predict(data)

                # Handle LSTM shorter output
                if len(predictions) < len(data):
                    pad = len(data) - len(predictions)
                    predictions = np.concatenate([[0] * pad, predictions])

                data['signal'] = predictions
            else:
                # Fallback to RSI strategy
                data['signal'] = 0
                for i in range(len(data)):
                    rsi = data['rsi_14'].iloc[i] if 'rsi_14' in data.columns else 50
                    if pd.notna(rsi):
                        if rsi < 30:
                            data.iloc[i, data.columns.get_loc('signal')] = 1
                        elif rsi > 70:
                            data.iloc[i, data.columns.get_loc('signal')] = -1

            if progress_callback:
                progress_callback("Running backtest...", 80)

            engine = BacktestEngine(initial_capital=capital)
            result = engine.run(data, signal_col='signal')

            return result

        self.worker = WorkerThread(backtest)
        self.worker.finished.connect(self.on_backtest_complete)
        self.worker.error.connect(self.on_error)
        self.worker.progress.connect(lambda msg, pct: self.status_label.setText(msg))
        self.worker.start()

    def on_backtest_complete(self, result):
        self.run_btn.setEnabled(True)
        if result is None:
            self.status_label.setText("No data found")
            return
        self.status_label.setText("Backtest complete!")
        color = "#00D4AA" if result.total_return > 0 else "#FF6B6B"
        self.metric_labels["Return"].setText(f"{result.total_return:.1%}")
        self.metric_labels["Return"].setStyleSheet(f"color: {color}; font-size: 14px; font-weight: bold;")
        self.metric_labels["Sharpe"].setText(f"{result.sharpe_ratio:.2f}")
        self.metric_labels["Drawdown"].setText(f"{result.max_drawdown:.1%}")
        self.metric_labels["Drawdown"].setStyleSheet("color: #FF6B6B; font-size: 14px; font-weight: bold;")
        self.metric_labels["Win Rate"].setText(f"{result.win_rate:.1%}")
        self.metric_labels["Trades"].setText(str(result.total_trades))
        self.equity_chart.axes.clear()

        # Plot equity curve with proper datetime handling
        equity = result.equity_curve
        dates = equity.index
        values = equity.values

        self.equity_chart.axes.plot(dates, values, color='#00D4AA', linewidth=1.5)
        self.equity_chart.axes.fill_between(dates, values, alpha=0.2, color='#00D4AA')

        # Add trade markers
        trades_df = result.trades
        if len(trades_df) > 0:
            for _, trade in trades_df.iterrows():
                entry_date = trade['entry_date']
                exit_date = trade['exit_date']
                is_long = trade['position'] == 'long'

                # Get equity values at entry/exit (approximate by finding nearest date)
                try:
                    entry_equity = equity.loc[entry_date] if entry_date in equity.index else equity.iloc[equity.index.get_indexer([entry_date], method='nearest')[0]]
                    exit_equity = equity.loc[exit_date] if exit_date in equity.index else equity.iloc[equity.index.get_indexer([exit_date], method='nearest')[0]]
                except:
                    continue

                # Entry marker: triangle up for long, down for short
                entry_marker = '^' if is_long else 'v'
                entry_color = '#3FB950' if is_long else '#F85149'
                self.equity_chart.axes.scatter([entry_date], [entry_equity],
                    marker=entry_marker, color=entry_color, s=80, zorder=5, edgecolors='white', linewidths=0.5)

                # Exit marker: square
                exit_color = '#3FB950' if trade['pnl'] > 0 else '#F85149'
                self.equity_chart.axes.scatter([exit_date], [exit_equity],
                    marker='s', color=exit_color, s=50, zorder=5, edgecolors='white', linewidths=0.5)

        # Style the chart
        self.equity_chart.axes.set_facecolor('#0D1117')
        self.equity_chart.axes.tick_params(colors='#8B949E', labelsize=8)

        # Format y-axis as currency
        self.equity_chart.axes.yaxis.set_major_formatter(
            lambda x, p: f'${x:,.0f}'
        )

        # Rotate x-axis labels for readability
        for label in self.equity_chart.axes.get_xticklabels():
            label.set_rotation(30)
            label.set_ha('right')

        self.equity_chart.fig.tight_layout()
        self.equity_chart.draw()

        # Populate trades table
        trades_df = result.trades
        self.trades_table.setRowCount(len(trades_df))

        for i, (_, trade) in enumerate(trades_df.iterrows()):
            # Type (Long/Short)
            position = trade['position'].upper()
            type_item = QTableWidgetItem(position)
            type_color = "#3FB950" if position == "LONG" else "#F85149"
            type_item.setForeground(QColor(type_color))
            self.trades_table.setItem(i, 0, type_item)

            # Entry date
            entry_date = trade['entry_date']
            if hasattr(entry_date, 'strftime'):
                entry_str = entry_date.strftime("%m/%d")
            else:
                entry_str = str(entry_date)[:5]
            self.trades_table.setItem(i, 1, QTableWidgetItem(entry_str))

            # Exit date
            exit_date = trade['exit_date']
            if hasattr(exit_date, 'strftime'):
                exit_str = exit_date.strftime("%m/%d")
            else:
                exit_str = str(exit_date)[:5]
            self.trades_table.setItem(i, 2, QTableWidgetItem(exit_str))

            # Entry price
            entry_price = QTableWidgetItem(f"${trade['entry_price']:.2f}")
            self.trades_table.setItem(i, 3, entry_price)

            # Exit price
            exit_price = QTableWidgetItem(f"${trade['exit_price']:.2f}")
            self.trades_table.setItem(i, 4, exit_price)

            # P&L with color
            pnl = trade['pnl']
            pnl_item = QTableWidgetItem(f"${pnl:+,.0f}")
            pnl_color = "#3FB950" if pnl > 0 else "#F85149" if pnl < 0 else "#8B949E"
            pnl_item.setForeground(QColor(pnl_color))
            self.trades_table.setItem(i, 5, pnl_item)

    def on_error(self, error):
        self.run_btn.setEnabled(True)
        self.status_label.setText(f"Error: {error[:40]}")


class ModelsTab(QWidget):
    model_selected = pyqtSignal(str)  # Emitted when user wants to activate a model

    def __init__(self):
        super().__init__()
        self.registry = None
        self.setup_ui()
        self.refresh_models()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Header
        header_layout = QHBoxLayout()
        header = QLabel("MODEL LIBRARY")
        header.setStyleSheet("color: #FFD93D; font-size: 22px; font-weight: bold;")
        header_layout.addWidget(header)

        header_layout.addStretch()

        # Hint about global selector
        hint = QLabel("Select a model from the header dropdown to use it across all tabs")
        hint.setStyleSheet("color: #6E7681; font-size: 12px;")
        header_layout.addWidget(hint)

        layout.addLayout(header_layout)

        # Main content area
        content_layout = QHBoxLayout()

        # Left: Models table
        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)

        self.models_table = QTableWidget()
        self.models_table.setColumnCount(5)
        self.models_table.setHorizontalHeaderLabels(["Type", "Acc", "F1", "Features", "Created"])
        self.models_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.models_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.models_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.models_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.models_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self.models_table.setColumnWidth(1, 55)
        self.models_table.setColumnWidth(2, 55)
        self.models_table.setColumnWidth(3, 60)
        self.models_table.setColumnWidth(4, 110)
        self.models_table.verticalHeader().setVisible(False)  # Hide row numbers
        self.models_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.models_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.models_table.setShowGrid(False)
        self.models_table.setAlternatingRowColors(False)  # Disable for consistent selection
        self.models_table.setStyleSheet("""
            QTableWidget {
                background-color: #161B22;
                border: 1px solid #30363D;
                border-radius: 8px;
            }
            QTableWidget::item {
                padding: 12px 8px;
                border-bottom: 1px solid #21262D;
                background-color: #161B22;
            }
            QTableWidget::item:selected {
                background-color: #1F6FEB;
                color: #FFFFFF;
            }
            QTableWidget::item:hover {
                background-color: #21262D;
            }
        """)
        self.models_table.selectionModel().selectionChanged.connect(self.on_selection_changed)
        table_layout.addWidget(self.models_table)

        # Buttons below table
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #21262D;
                border: 1px solid #30363D;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #30363D;
            }
        """)
        self.refresh_btn.clicked.connect(self.refresh_models)
        buttons_layout.addWidget(self.refresh_btn)

        self.keep_best_btn = QPushButton("Keep Best Only")
        self.keep_best_btn.setStyleSheet("""
            QPushButton {
                background-color: #1F6FEB;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #388BFD;
            }
        """)
        self.keep_best_btn.setToolTip("Delete all but the best model of each type (by F1 score)")
        self.keep_best_btn.clicked.connect(self.on_keep_best)
        buttons_layout.addWidget(self.keep_best_btn)

        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #DA3633;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #F85149;
            }
            QPushButton:disabled {
                background-color: #21262D;
                color: #6E7681;
            }
        """)
        self.delete_btn.setEnabled(False)
        self.delete_btn.clicked.connect(self.on_delete)
        buttons_layout.addWidget(self.delete_btn)

        table_layout.addLayout(buttons_layout)
        content_layout.addWidget(table_container, 2)

        # Right: Details panel
        details_panel = QGroupBox("MODEL DETAILS")
        details_panel.setStyleSheet("""
            QGroupBox {
                background-color: #161B22;
                border: 1px solid #30363D;
                border-radius: 8px;
                padding: 20px;
                margin-top: 10px;
            }
            QGroupBox::title {
                color: #8B949E;
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
            }
        """)
        details_layout = QVBoxLayout(details_panel)
        details_layout.setSpacing(15)

        self.detail_labels = {}
        for field, icon in [("Type", "◆"), ("Features", "⊞"), ("Created", "◷"), ("File", "◇")]:
            row = QHBoxLayout()

            icon_label = QLabel(icon)
            icon_label.setStyleSheet("color: #58A6FF; font-size: 14px;")
            icon_label.setFixedWidth(25)
            row.addWidget(icon_label)

            field_label = QLabel(field)
            field_label.setStyleSheet("color: #8B949E; font-size: 12px;")
            field_label.setFixedWidth(70)
            row.addWidget(field_label)

            value_label = QLabel("—")
            value_label.setStyleSheet("color: #E6EDF3; font-size: 13px;")
            value_label.setWordWrap(True)
            row.addWidget(value_label, 1)

            self.detail_labels[field] = value_label
            details_layout.addLayout(row)

        details_layout.addStretch()

        # Activate button
        self.use_btn = QPushButton("Set as Active Model")
        self.use_btn.setStyleSheet("""
            QPushButton {
                background-color: #238636;
                font-size: 13px;
                padding: 12px 24px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #2EA043;
            }
            QPushButton:disabled {
                background-color: #21262D;
                color: #6E7681;
            }
        """)
        self.use_btn.setEnabled(False)
        self.use_btn.clicked.connect(self.on_use_model)
        details_layout.addWidget(self.use_btn)

        content_layout.addWidget(details_panel, 1)

        layout.addLayout(content_layout)

    def refresh_models(self):
        """Load models from registry and populate table."""
        from ..services import ModelRegistry

        self.registry = ModelRegistry()
        models = self.registry.list_models()

        # Sort by created date (newest first)
        models.sort(key=lambda m: m.created, reverse=True)

        self.models_table.setRowCount(len(models))
        for i, model in enumerate(models):
            # Type column with colored indicator
            type_item = QTableWidgetItem(f"  {model.model_type}")
            type_item.setData(Qt.ItemDataRole.UserRole, model.name)

            # Color by type - muted colors that work with selection
            type_colors = {
                "Random Forest": "#3FB950",
                "XGBoost": "#A371F7",
                "LSTM": "#58A6FF"
            }
            color = type_colors.get(model.model_type, "#8B949E")
            type_item.setForeground(QColor(color))
            self.models_table.setItem(i, 0, type_item)

            # Accuracy
            acc = model.metrics.get("accuracy") if model.metrics else None
            acc_str = f"{acc:.0%}" if acc is not None else "—"
            acc_item = QTableWidgetItem(acc_str)
            acc_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if acc is not None and acc >= 0.6:
                acc_item.setForeground(QColor("#3FB950"))
            self.models_table.setItem(i, 1, acc_item)

            # F1 Score
            f1 = model.metrics.get("f1") if model.metrics else None
            f1_str = f"{f1:.0%}" if f1 is not None else "—"
            f1_item = QTableWidgetItem(f1_str)
            f1_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if f1 is not None and f1 >= 0.5:
                f1_item.setForeground(QColor("#3FB950"))
            self.models_table.setItem(i, 2, f1_item)

            # Features count
            features_item = QTableWidgetItem(str(model.feature_count))
            features_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.models_table.setItem(i, 3, features_item)

            # Created date
            date_str = model.created.strftime("%b %d, %H:%M")
            date_item = QTableWidgetItem(date_str)
            date_item.setForeground(QColor("#8B949E"))
            self.models_table.setItem(i, 4, date_item)

            self.models_table.setRowHeight(i, 40)

        self.clear_details()

    def clear_details(self):
        """Reset all detail labels to default."""
        for label in self.detail_labels.values():
            label.setText("—")
        self.delete_btn.setEnabled(False)
        self.delete_btn.setText("Delete Selected")
        self.use_btn.setEnabled(False)

    def on_selection_changed(self):
        """Update details panel when selection changes."""
        selected_rows = set(item.row() for item in self.models_table.selectedItems())
        num_selected = len(selected_rows)

        if num_selected == 0:
            self.clear_details()
            self.delete_btn.setText("Delete Selected")
            return

        # Update delete button text
        if num_selected == 1:
            self.delete_btn.setText("Delete Selected")
        else:
            self.delete_btn.setText(f"Delete Selected ({num_selected})")

        self.delete_btn.setEnabled(True)

        # Show details for single selection, or first selected for multi
        row = min(selected_rows)
        type_item = self.models_table.item(row, 0)
        if not type_item:
            self.clear_details()
            return

        model_name = type_item.data(Qt.ItemDataRole.UserRole)
        if not model_name or not self.registry:
            self.clear_details()
            return

        # Disable "Set as Active" button for multi-select
        self.use_btn.setEnabled(num_selected == 1)

        try:
            info = self.registry.get_model_info(model_name)
            self.detail_labels["Type"].setText(info.model_type)
            self.detail_labels["Features"].setText(f"{info.feature_count} features")
            self.detail_labels["Created"].setText(info.created.strftime("%B %d, %Y at %H:%M"))
            self.detail_labels["File"].setText(info.path.name)
            self.delete_btn.setEnabled(True)
            self.use_btn.setEnabled(True)
        except KeyError:
            self.clear_details()

    def on_delete(self):
        """Delete the selected model(s)."""
        selected_rows = sorted(set(item.row() for item in self.models_table.selectedItems()), reverse=True)
        if not selected_rows or not self.registry:
            return

        # Collect model names to delete
        models_to_delete = []
        for row in selected_rows:
            type_item = self.models_table.item(row, 0)
            if type_item:
                model_name = type_item.data(Qt.ItemDataRole.UserRole)
                if model_name:
                    try:
                        info = self.registry.get_model_info(model_name)
                        display_name = f"{info.model_type} ({info.created.strftime('%b %d, %H:%M')})"
                    except KeyError:
                        display_name = model_name
                    models_to_delete.append((model_name, display_name))

        if not models_to_delete:
            return

        # Build confirmation message
        if len(models_to_delete) == 1:
            msg = f"Delete this model?\n\n{models_to_delete[0][1]}\n\nThis cannot be undone."
        else:
            model_list = "\n".join(f"  • {name}" for _, name in models_to_delete[:5])
            if len(models_to_delete) > 5:
                model_list += f"\n  ... and {len(models_to_delete) - 5} more"
            msg = f"Delete {len(models_to_delete)} models?\n\n{model_list}\n\nThis cannot be undone."

        # Confirm deletion
        reply = QMessageBox.question(
            self, "Delete Model" if len(models_to_delete) == 1 else "Delete Models",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Delete all selected models
        deleted = 0
        for model_name, _ in models_to_delete:
            try:
                self.registry.delete_model(model_name)
                deleted += 1
            except KeyError:
                pass

        self.refresh_models()

        if deleted < len(models_to_delete):
            QMessageBox.warning(self, "Warning", f"Deleted {deleted} of {len(models_to_delete)} models")

    def on_keep_best(self):
        """Keep only the best model of each type (by F1 score), delete the rest."""
        if not self.registry:
            return

        models = self.registry.list_models()
        if not models:
            QMessageBox.information(self, "No Models", "No models to clean up.")
            return

        # Group models by type
        by_type = {}
        for model in models:
            model_type = model.model_type
            if model_type not in by_type:
                by_type[model_type] = []
            by_type[model_type].append(model)

        # Find best of each type and models to delete
        models_to_delete = []
        kept_models = []

        for model_type, type_models in by_type.items():
            if len(type_models) <= 1:
                # Only one model of this type, keep it
                if type_models:
                    kept_models.append(type_models[0])
                continue

            # Sort by F1 score (fallback to accuracy if F1 not available)
            def get_score(m):
                if m.metrics:
                    return m.metrics.get("f1", m.metrics.get("accuracy", 0)) or 0
                return 0

            sorted_models = sorted(type_models, key=get_score, reverse=True)
            best = sorted_models[0]
            kept_models.append(best)
            models_to_delete.extend(sorted_models[1:])

        if not models_to_delete:
            QMessageBox.information(self, "Already Clean", "Each model type already has only one model.")
            return

        # Build confirmation message
        keep_list = "\n".join(f"  ✓ {m.model_type}: {m.metrics.get('f1', 0):.1%} F1" for m in kept_models)
        delete_list = "\n".join(f"  • {m.model_type} ({m.created.strftime('%b %d, %H:%M')})"
                                for m in models_to_delete[:5])
        if len(models_to_delete) > 5:
            delete_list += f"\n  ... and {len(models_to_delete) - 5} more"

        msg = f"Keep best model of each type?\n\nKEEPING:\n{keep_list}\n\nDELETING ({len(models_to_delete)}):\n{delete_list}\n\nThis cannot be undone."

        reply = QMessageBox.question(
            self, "Keep Best Only",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Delete non-best models
        deleted = 0
        for model in models_to_delete:
            try:
                self.registry.delete_model(model.name)
                deleted += 1
            except KeyError:
                pass

        self.refresh_models()
        QMessageBox.information(self, "Cleanup Complete", f"Deleted {deleted} models, kept {len(kept_models)} best.")

    def on_use_model(self):
        """Emit signal when user wants to activate the selected model."""
        row = self.models_table.currentRow()
        if row < 0:
            return

        type_item = self.models_table.item(row, 0)
        if not type_item:
            return

        model_name = type_item.data(Qt.ItemDataRole.UserRole)
        if model_name:
            self.model_selected.emit(model_name)


class SwingTraderApp(QMainWindow):
    """Main application window with global model selector."""

    model_changed = pyqtSignal(str)  # Emitted when global model selection changes

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Swing Trader")
        self.setMinimumSize(1400, 900)
        self._current_model = None
        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header with global model selector
        header = QWidget()
        header.setStyleSheet("background-color: #0D1117;")
        header.setFixedHeight(60)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 10, 20, 10)

        # Left side: title
        title = QLabel("SWING TRADER")
        title.setStyleSheet("color: #FFD93D; font-size: 20px; font-weight: bold;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Right side: Global model selector
        model_label = QLabel("ACTIVE MODEL")
        model_label.setStyleSheet("color: #6E7681; font-size: 11px; margin-right: 8px;")
        header_layout.addWidget(model_label)

        self.global_model_combo = QComboBox()
        self.global_model_combo.setMinimumWidth(280)
        self.global_model_combo.setStyleSheet("""
            QComboBox {
                background-color: #21262D;
                border: 2px solid #30363D;
                border-radius: 6px;
                padding: 8px 12px;
                color: #E6EDF3;
                font-size: 13px;
                font-weight: bold;
            }
            QComboBox:hover {
                border-color: #58A6FF;
            }
            QComboBox:focus {
                border-color: #58A6FF;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 10px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #8B949E;
                margin-right: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #21262D;
                border: 1px solid #30363D;
                selection-background-color: #388BFD;
                color: #E6EDF3;
            }
        """)
        self.global_model_combo.currentIndexChanged.connect(self._on_global_model_changed)
        header_layout.addWidget(self.global_model_combo)

        layout.addWidget(header)

        self.tabs = QTabWidget()

        # Create tabs and store references
        self.signals_tab = SignalsTab()
        self.training_tab = TrainingTab()
        self.backtest_tab = BacktestTab()
        self.models_tab = ModelsTab()

        self.tabs.addTab(self.signals_tab, "  Signals  ")
        self.tabs.addTab(self.training_tab, "  Training  ")
        self.tabs.addTab(self.backtest_tab, "  Backtest  ")
        self.tabs.addTab(self.models_tab, "  Models  ")

        # Connect global model changes to tabs
        self.model_changed.connect(self.signals_tab.set_model)
        self.model_changed.connect(self.backtest_tab.set_model)

        # Refresh global model list when switching tabs (catches new models from training)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Connect training completion to refresh models
        self.training_tab.training_complete.connect(self.refresh_global_models)

        # Connect Models tab selection to global selector
        self.models_tab.model_selected.connect(self.set_active_model)

        layout.addWidget(self.tabs)

        # Initial load of models
        self.refresh_global_models()

    def refresh_global_models(self):
        """Refresh the global model dropdown."""
        from ..services import ModelRegistry
        registry = ModelRegistry()

        current_data = self.global_model_combo.currentData()
        self.global_model_combo.blockSignals(True)
        self.global_model_combo.clear()

        self.global_model_combo.addItem("No model selected", None)

        models = registry.list_models()
        # Sort by created date (newest first)
        models.sort(key=lambda m: m.created, reverse=True)

        for info in models:
            # Format: "XGBoost · Jan 31, 14:30 · 17 features"
            date_str = info.created.strftime("%b %d, %H:%M")
            display = f"{info.model_type}  ·  {date_str}  ·  {info.feature_count} features"
            self.global_model_combo.addItem(display, info.name)

        # Restore previous selection if possible
        if current_data:
            for i in range(self.global_model_combo.count()):
                if self.global_model_combo.itemData(i) == current_data:
                    self.global_model_combo.setCurrentIndex(i)
                    break

        self.global_model_combo.blockSignals(False)

    def set_active_model(self, model_name: str):
        """Set the active model by name (called from Models tab)."""
        for i in range(self.global_model_combo.count()):
            if self.global_model_combo.itemData(i) == model_name:
                self.global_model_combo.setCurrentIndex(i)
                break

    def _on_global_model_changed(self, index: int):
        """Handle global model selection change."""
        model_name = self.global_model_combo.currentData()
        self._current_model = model_name
        self.model_changed.emit(model_name if model_name else "")

    def _on_tab_changed(self, index: int):
        """Handle tab changes - refresh models and model lists."""
        self.refresh_global_models()

        widget = self.tabs.widget(index)
        if hasattr(widget, 'refresh_models'):
            widget.refresh_models()

    def get_active_model(self) -> str | None:
        """Return the currently selected model name."""
        return self._current_model


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = SwingTraderApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
