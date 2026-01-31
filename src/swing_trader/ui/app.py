import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QLineEdit, QPushButton, QComboBox, QSlider,
    QCheckBox, QProgressBar, QTableWidget, QTableWidgetItem, QGroupBox,
    QSpinBox, QDoubleSpinBox, QHeaderView, QSplitter, QFrame, QScrollArea
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

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class SignalsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
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

        scan_layout.addWidget(QLabel("Filter"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All Signals", "BUY Only", "SELL Only"])
        scan_layout.addWidget(self.filter_combo)

        self.scan_btn = QPushButton("Scan Top Stocks")
        self.scan_btn.clicked.connect(self.on_scan)
        scan_layout.addWidget(self.scan_btn)

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
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(["Ticker", "Signal", "Confidence", "Price", "RSI"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
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

        def analyze():
            from ..data.fetcher import StockDataFetcher
            from ..features.indicators import TechnicalIndicators
            fetcher = StockDataFetcher()
            indicators = TechnicalIndicators()
            data = fetcher.fetch(ticker, period=self.period_combo.currentText())
            if data is not None and not data.empty:
                data = indicators.add_all(data)
            return data, ticker

        self.worker = WorkerThread(analyze)
        self.worker.finished.connect(self.on_analyze_complete)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_analyze_complete(self, result):
        data, ticker = result
        self.analyze_btn.setEnabled(True)
        if data is None or data.empty:
            self.status_label.setText(f"No data for {ticker}")
            return
        self.status_label.setText(f"Analysis complete for {ticker}")
        self.chart.axes.clear()
        self.chart.axes.plot(data.index, data['close'], color='#58A6FF', label='Close')
        if 'sma_20' in data.columns:
            self.chart.axes.plot(data.index, data['sma_20'], color='#FFD93D', label='SMA 20', alpha=0.7)
        self.chart.axes.legend(facecolor='#0D1117', edgecolor='#30363D')
        self.chart.axes.set_facecolor('#0D1117')
        self.chart.draw()

    def on_scan(self):
        self.status_label.setText("Scanning...")
        self.scan_btn.setEnabled(False)

        def scan():
            from ..data.fetcher import StockDataFetcher
            from ..features.indicators import TechnicalIndicators
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "WMT"]
            fetcher = StockDataFetcher()
            indicators = TechnicalIndicators()
            results = []
            for ticker in tickers:
                try:
                    data = fetcher.fetch(ticker, period="6mo")
                    if data is None or data.empty:
                        continue
                    data = indicators.add_all(data)
                    latest = data.iloc[-1]
                    rsi = latest.get('rsi_14', 50)
                    macd = latest.get('macd', 0)
                    macd_signal = latest.get('macd_signal', 0)
                    if rsi < 30 and macd > macd_signal:
                        signal, conf = "BUY", min(1.0, (30 - rsi) / 30 + 0.5)
                    elif rsi > 70 and macd < macd_signal:
                        signal, conf = "SELL", min(1.0, (rsi - 70) / 30 + 0.5)
                    else:
                        signal, conf = "HOLD", 0.5
                    results.append({"ticker": ticker, "signal": signal, "confidence": conf,
                                   "price": latest['close'], "rsi": rsi})
                except:
                    continue
            return results

        self.worker = WorkerThread(scan)
        self.worker.finished.connect(self.on_scan_complete)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_scan_complete(self, results):
        self.scan_btn.setEnabled(True)
        self.status_label.setText(f"Found {len(results)} results")
        self.results_table.setRowCount(len(results))
        for i, r in enumerate(sorted(results, key=lambda x: x['confidence'], reverse=True)):
            self.results_table.setItem(i, 0, QTableWidgetItem(r['ticker']))
            signal_item = QTableWidgetItem(r['signal'])
            color = "#00D4AA" if r['signal'] == "BUY" else "#FF6B6B" if r['signal'] == "SELL" else "#6E7681"
            signal_item.setForeground(QColor(color))
            self.results_table.setItem(i, 1, signal_item)
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{r['confidence']:.0%}"))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"${r['price']:.2f}"))
            self.results_table.setItem(i, 4, QTableWidgetItem(f"{r['rsi']:.1f}"))

    def on_error(self, error):
        self.analyze_btn.setEnabled(True)
        self.scan_btn.setEnabled(True)
        self.status_label.setText(f"Error: {error[:40]}")


class TrainingTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)

        left_panel = QWidget()
        left_panel.setFixedWidth(340)
        left_layout = QVBoxLayout(left_panel)

        train_group = QGroupBox("MODEL TRAINING")
        train_layout = QVBoxLayout(train_group)

        train_layout.addWidget(QLabel("Model Type"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Random Forest", "XGBoost (GPU)", "LSTM (CUDA)", "All Models"])
        train_layout.addWidget(self.model_combo)

        train_layout.addWidget(QLabel("Training Symbols"))
        self.tickers_input = QLineEdit("AAPL, MSFT, GOOGL")
        train_layout.addWidget(self.tickers_input)

        train_layout.addWidget(QLabel("Data Period"))
        self.period_combo = QComboBox()
        self.period_combo.addItems(["1y", "2y", "5y"])
        self.period_combo.setCurrentText("2y")
        train_layout.addWidget(self.period_combo)

        self.train_btn = QPushButton("Start Training")
        self.train_btn.clicked.connect(self.on_train)
        train_layout.addWidget(self.train_btn)

        left_layout.addWidget(train_group)

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

        chart_group = QGroupBox("TRAINING METRICS")
        chart_layout = QVBoxLayout(chart_group)
        self.chart = create_chart_canvas(self, width=8, height=5)
        chart_layout.addWidget(self.chart)
        right_layout.addWidget(chart_group)

        layout.addWidget(left_panel)
        layout.addWidget(right_panel, 1)

    def on_train(self):
        self.status_label.setText("Starting training...")
        self.train_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        def train():
            from ..data.fetcher import StockDataFetcher
            from ..features.indicators import TechnicalIndicators
            from ..features.labeler import SignalLabeler
            from ..models.random_forest import RandomForestModel
            import pandas as pd
            from pathlib import Path

            tickers = [t.strip().upper() for t in self.tickers_input.text().split(",")]
            fetcher = StockDataFetcher()
            indicators = TechnicalIndicators()
            labeler = SignalLabeler()

            all_data = []
            for ticker in tickers:
                data = fetcher.fetch(ticker, period=self.period_combo.currentText())
                if data is not None and not data.empty:
                    data = indicators.add_all(data)
                    data = labeler.create_labels(data)
                    all_data.append(data)

            if not all_data:
                return None, 0

            combined = pd.concat(all_data, ignore_index=True).dropna()
            feature_cols = [c for c in combined.columns if c not in
                          ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'label', 'forward_return']]
            X = combined[feature_cols].values
            y = combined['label'].values

            model = RandomForestModel(n_estimators=100)
            model.fit(X, y)

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            accuracy = (model.predict(X_test) == y_test).mean()

            Path("models").mkdir(exist_ok=True)
            model.save(Path("models/random_forest.joblib"))

            return len(combined), accuracy

        self.worker = WorkerThread(train)
        self.worker.finished.connect(self.on_train_complete)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_train_complete(self, result):
        self.train_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        if result[0] is None:
            self.status_label.setText("No data collected")
            return
        samples, accuracy = result
        self.status_label.setText(f"Training complete! Samples: {samples}, Accuracy: {accuracy:.1%}")

    def on_error(self, error):
        self.train_btn.setEnabled(True)
        self.status_label.setText(f"Error: {error[:50]}")


class BacktestTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
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

        equity_group = QGroupBox("EQUITY CURVE")
        equity_layout = QVBoxLayout(equity_group)
        self.equity_chart = create_chart_canvas(self, width=8, height=4)
        equity_layout.addWidget(self.equity_chart)
        right_layout.addWidget(equity_group)

        metrics_group = QGroupBox("PERFORMANCE METRICS")
        metrics_layout = QHBoxLayout(metrics_group)
        self.metric_labels = {}
        for name in ["Return", "Sharpe", "Drawdown", "Win Rate", "Trades"]:
            frame = QGroupBox(name.upper())
            frame.setFixedHeight(80)
            frame_layout = QVBoxLayout(frame)
            label = QLabel("--")
            label.setStyleSheet("color: #6E7681; font-size: 14px; font-weight: bold;")
            frame_layout.addWidget(label)
            self.metric_labels[name] = label
            metrics_layout.addWidget(frame)
        right_layout.addWidget(metrics_group)

        layout.addWidget(left_panel)
        layout.addWidget(right_panel, 1)

    def on_run(self):
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            self.status_label.setText("Enter a ticker")
            return
        self.status_label.setText(f"Running backtest...")
        self.run_btn.setEnabled(False)

        def backtest():
            from ..data.fetcher import StockDataFetcher
            from ..features.indicators import TechnicalIndicators
            from ..backtest.engine import BacktestEngine
            import pandas as pd

            fetcher = StockDataFetcher()
            indicators = TechnicalIndicators()
            data = fetcher.fetch(ticker, period=self.period_combo.currentText())
            if data is None or data.empty:
                return None
            data = indicators.add_all(data)
            signals = pd.Series(index=data.index, data=0)
            for i in range(len(data)):
                rsi = data['rsi_14'].iloc[i] if 'rsi_14' in data.columns else 50
                if pd.notna(rsi):
                    if rsi < 30:
                        signals.iloc[i] = 1
                    elif rsi > 70:
                        signals.iloc[i] = -1
            engine = BacktestEngine()
            result = engine.run(data, signals, initial_capital=self.capital_spin.value())
            return result

        self.worker = WorkerThread(backtest)
        self.worker.finished.connect(self.on_backtest_complete)
        self.worker.error.connect(self.on_error)
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
        self.equity_chart.axes.plot(result.equity_curve, color='#00D4AA')
        self.equity_chart.axes.fill_between(range(len(result.equity_curve)), result.equity_curve, alpha=0.3, color='#00D4AA')
        self.equity_chart.axes.set_facecolor('#0D1117')
        self.equity_chart.draw()

    def on_error(self, error):
        self.run_btn.setEnabled(True)
        self.status_label.setText(f"Error: {error[:40]}")


class ModelsTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        header = QLabel("MODEL REGISTRY")
        header.setStyleSheet("color: #FFD93D; font-size: 18px; font-weight: bold;")
        layout.addWidget(header)
        subtitle = QLabel("Manage trained models with MLflow")
        subtitle.setStyleSheet("color: #6E7681;")
        layout.addWidget(subtitle)
        layout.addStretch()


class SwingTraderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Swing Trader")
        self.setMinimumSize(1400, 900)
        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QWidget()
        header.setStyleSheet("background-color: #0D1117; padding: 10px;")
        header_layout = QHBoxLayout(header)
        title = QLabel("SWING TRADER")
        title.setStyleSheet("color: #FFD93D; font-size: 20px; font-weight: bold;")
        header_layout.addWidget(title)
        sep = QLabel("|")
        sep.setStyleSheet("color: #30363D; margin: 0 15px;")
        header_layout.addWidget(sep)
        subtitle = QLabel("ML Trading Signals")
        subtitle.setStyleSheet("color: #8B949E;")
        header_layout.addWidget(subtitle)
        header_layout.addStretch()
        layout.addWidget(header)

        tabs = QTabWidget()
        tabs.addTab(SignalsTab(), "  Signals  ")
        tabs.addTab(TrainingTab(), "  Training  ")
        tabs.addTab(BacktestTab(), "  Backtest  ")
        tabs.addTab(ModelsTab(), "  Models  ")
        layout.addWidget(tabs)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = SwingTraderApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
