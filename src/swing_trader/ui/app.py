import sys

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
        left_panel.setFixedWidth(380)
        left_layout = QVBoxLayout(left_panel)

        train_group = QGroupBox("MODEL TRAINING")
        train_layout = QVBoxLayout(train_group)

        train_layout.addWidget(QLabel("Model Type"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Random Forest", "XGBoost", "LSTM", "All Models"])
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        train_layout.addWidget(self.model_combo)

        train_layout.addWidget(QLabel("Training Symbols"))
        self.tickers_input = QLineEdit("AAPL, MSFT, GOOGL")
        train_layout.addWidget(self.tickers_input)

        train_layout.addWidget(QLabel("Data Period"))
        self.period_combo = QComboBox()
        self.period_combo.addItems(["1y", "2y", "5y"])
        self.period_combo.setCurrentText("2y")
        train_layout.addWidget(self.period_combo)

        left_layout.addWidget(train_group)

        # Hyperparameters group
        self.hyperparam_group = QGroupBox("HYPERPARAMETERS")
        self.hyperparam_layout = QVBoxLayout(self.hyperparam_group)
        self.setup_rf_hyperparams()
        left_layout.addWidget(self.hyperparam_group)

        self.train_btn = QPushButton("Start Training")
        self.train_btn.clicked.connect(self.on_train)
        left_layout.addWidget(self.train_btn)

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

        layout.addWidget(left_panel)
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
        else:  # All Models
            self.setup_rf_hyperparams()

    def on_train(self):
        self.status_label.setText("Starting training...")
        self.train_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        # Capture ALL UI values before starting thread (Qt widgets can't be accessed from threads)
        model_type = self.model_combo.currentText()
        tickers_text = self.tickers_input.text()
        period = self.period_combo.currentText()

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
                    data = indicators.add_all(data)
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

            update_progress(f"Training {model_type}...", 40)

            # Create model based on selection
            is_lstm = False
            if model_type == "Random Forest":
                from ..models.random_forest import RandomForestModel
                md = max_depth if max_depth > 0 else None
                model = RandomForestModel(n_estimators=n_estimators, max_depth=md)
            elif model_type == "XGBoost":
                from ..models.xgboost_model import XGBoostModel
                # Auto-detect GPU availability
                use_gpu = torch.cuda.is_available()
                model = XGBoostModel(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    use_gpu=use_gpu
                )
            elif model_type == "LSTM":
                from ..models.lstm import LSTMModel
                # Auto-detect CUDA - use GPU if available, fallback to CPU
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = LSTMModel(
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    device=device
                )
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

            update_progress("Saving model...", 95)

            # Save model
            Path("models").mkdir(exist_ok=True)
            model_filename = model_type.lower().replace(" ", "_")
            model.save(Path(f"models/{model_filename}_model.joblib"))

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
    model_selected = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.registry = None
        self.setup_ui()
        self.refresh_models()

    def setup_ui(self):
        layout = QHBoxLayout(self)

        # Left panel - model list
        left_panel = QWidget()
        left_panel.setFixedWidth(350)
        left_layout = QVBoxLayout(left_panel)

        header = QLabel("TRAINED MODELS")
        header.setStyleSheet("color: #FFD93D; font-size: 18px; font-weight: bold;")
        left_layout.addWidget(header)

        # Models table
        self.models_table = QTableWidget()
        self.models_table.setColumnCount(3)
        self.models_table.setHorizontalHeaderLabels(["Name", "Type", "Created"])
        self.models_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.models_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.models_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.models_table.selectionModel().selectionChanged.connect(self.on_selection_changed)
        left_layout.addWidget(self.models_table)

        # Buttons
        buttons_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_models)
        buttons_layout.addWidget(self.refresh_btn)

        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #DA3633;
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
        left_layout.addLayout(buttons_layout)

        layout.addWidget(left_panel)

        # Right panel - model details
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        details_group = QGroupBox("MODEL DETAILS")
        details_layout = QVBoxLayout(details_group)

        # Detail labels
        self.detail_labels = {}
        for field in ["Name", "Type", "Features", "Created", "Path"]:
            row_layout = QHBoxLayout()
            label = QLabel(f"{field}:")
            label.setStyleSheet("color: #8B949E; font-weight: bold;")
            label.setFixedWidth(80)
            row_layout.addWidget(label)

            value_label = QLabel("--")
            value_label.setStyleSheet("color: #E6EDF3;")
            value_label.setWordWrap(True)
            row_layout.addWidget(value_label, 1)
            self.detail_labels[field] = value_label
            details_layout.addLayout(row_layout)

        details_layout.addStretch()

        # Use model button
        self.use_btn = QPushButton("Use This Model")
        self.use_btn.setStyleSheet("""
            QPushButton {
                background-color: #238636;
                font-size: 14px;
                padding: 15px 30px;
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

        right_layout.addWidget(details_group)
        layout.addWidget(right_panel, 1)

    def refresh_models(self):
        """Load models from registry and populate table."""
        from ..services import ModelRegistry

        self.registry = ModelRegistry()
        models = self.registry.list_models()

        # Sort by created date (newest first)
        models.sort(key=lambda m: m.created, reverse=True)

        self.models_table.setRowCount(len(models))
        for i, model in enumerate(models):
            name_item = QTableWidgetItem(model.name)
            name_item.setData(Qt.ItemDataRole.UserRole, model.name)
            self.models_table.setItem(i, 0, name_item)
            self.models_table.setItem(i, 1, QTableWidgetItem(model.model_type))
            self.models_table.setItem(i, 2, QTableWidgetItem(model.created.strftime("%Y-%m-%d %H:%M")))

        self.clear_details()

    def clear_details(self):
        """Reset all detail labels to default."""
        for label in self.detail_labels.values():
            label.setText("--")
        self.delete_btn.setEnabled(False)
        self.use_btn.setEnabled(False)

    def on_selection_changed(self):
        """Update details panel when selection changes."""
        selected = self.models_table.selectedItems()
        if not selected:
            self.clear_details()
            return

        row = self.models_table.currentRow()
        name_item = self.models_table.item(row, 0)
        if not name_item:
            self.clear_details()
            return

        model_name = name_item.data(Qt.ItemDataRole.UserRole)
        if not model_name or not self.registry:
            self.clear_details()
            return

        try:
            info = self.registry.get_model_info(model_name)
            self.detail_labels["Name"].setText(info.name)
            self.detail_labels["Type"].setText(info.model_type)
            self.detail_labels["Features"].setText(str(info.feature_count))
            self.detail_labels["Created"].setText(info.created.strftime("%Y-%m-%d %H:%M:%S"))
            self.detail_labels["Path"].setText(str(info.path))
            self.delete_btn.setEnabled(True)
            self.use_btn.setEnabled(True)
        except KeyError:
            self.clear_details()

    def on_delete(self):
        """Delete the selected model."""
        row = self.models_table.currentRow()
        if row < 0:
            return

        name_item = self.models_table.item(row, 0)
        if not name_item or not self.registry:
            return

        model_name = name_item.data(Qt.ItemDataRole.UserRole)
        if not model_name:
            return

        # Confirm deletion
        reply = QMessageBox.question(
            self, "Delete Model",
            f"Are you sure you want to delete '{model_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            self.registry.delete_model(model_name)
            self.refresh_models()
        except KeyError:
            QMessageBox.warning(self, "Error", f"Failed to delete model '{model_name}'")

    def on_use_model(self):
        """Emit signal when user wants to use the selected model."""
        row = self.models_table.currentRow()
        if row < 0:
            return

        name_item = self.models_table.item(row, 0)
        if not name_item:
            return

        model_name = name_item.data(Qt.ItemDataRole.UserRole)
        if model_name:
            self.model_selected.emit(model_name)


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
