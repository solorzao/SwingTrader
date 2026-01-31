# Model Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Connect trained models to Signals, Backtest, and Models tabs so users can select and use their trained models throughout the application.

**Architecture:** Create a shared model registry that loads models from `models/` directory. Add model selector dropdowns to Signals and Backtest tabs. Implement ModelsTab to display saved models with metadata. Use existing `SignalGenerator` class for predictions.

**Tech Stack:** PyQt6, joblib, pandas, existing swing_trader modules

---

## Task 1: Create Model Registry Service

**Files:**
- Create: `src/swing_trader/services/__init__.py`
- Create: `src/swing_trader/services/model_registry.py`

**Step 1: Create services package**

Create empty `__init__.py`:

```python
# src/swing_trader/services/__init__.py
from .model_registry import ModelRegistry

__all__ = ["ModelRegistry"]
```

**Step 2: Create ModelRegistry class**

```python
# src/swing_trader/services/model_registry.py
"""Central registry for managing trained models."""
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import joblib


@dataclass
class ModelInfo:
    """Metadata about a saved model."""
    name: str
    model_type: str  # rf, xgboost, lstm
    path: Path
    created: datetime
    feature_count: int


class ModelRegistry:
    """Singleton registry for loading and managing trained models."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_dir: Path | str = "models"):
        if self._initialized:
            return
        self.model_dir = Path(model_dir)
        self._models: dict[str, object] = {}  # name -> loaded model
        self._model_info: dict[str, ModelInfo] = {}  # name -> metadata
        self._initialized = True
        self.refresh()

    def refresh(self) -> None:
        """Scan model directory and update registry."""
        self._models.clear()
        self._model_info.clear()

        if not self.model_dir.exists():
            return

        for model_file in self.model_dir.glob("*.joblib"):
            try:
                self._register_model(model_file)
            except Exception as e:
                print(f"Warning: Could not load {model_file}: {e}")

    def _register_model(self, path: Path) -> None:
        """Load model metadata without fully loading the model."""
        data = joblib.load(path)
        name = path.stem

        # Determine model type from name or data
        model_type = "unknown"
        if "random_forest" in name or "rf" in name:
            model_type = "Random Forest"
        elif "xgboost" in name or "xgb" in name:
            model_type = "XGBoost"
        elif "lstm" in name:
            model_type = "LSTM"

        feature_count = len(data.get("feature_columns", []))
        created = datetime.fromtimestamp(path.stat().st_mtime)

        self._model_info[name] = ModelInfo(
            name=name,
            model_type=model_type,
            path=path,
            created=created,
            feature_count=feature_count
        )

    def list_models(self) -> list[ModelInfo]:
        """Return list of available models."""
        return list(self._model_info.values())

    def get_model(self, name: str):
        """Load and return a model by name."""
        if name not in self._model_info:
            raise KeyError(f"Model '{name}' not found")

        if name not in self._models:
            info = self._model_info[name]
            self._models[name] = self._load_model(info)

        return self._models[name]

    def _load_model(self, info: ModelInfo):
        """Fully load a model from disk."""
        from ..models.random_forest import RandomForestModel
        from ..models.xgboost_model import XGBoostModel
        from ..models.lstm import LSTMModel

        model_classes = {
            "Random Forest": RandomForestModel,
            "XGBoost": XGBoostModel,
            "LSTM": LSTMModel
        }

        model_class = model_classes.get(info.model_type)
        if model_class is None:
            raise ValueError(f"Unknown model type: {info.model_type}")

        model = model_class()
        model.load(info.path)
        return model

    def get_model_info(self, name: str) -> ModelInfo | None:
        """Get metadata for a model."""
        return self._model_info.get(name)

    def delete_model(self, name: str) -> bool:
        """Delete a model from disk and registry."""
        if name not in self._model_info:
            return False

        info = self._model_info[name]
        try:
            info.path.unlink()
            del self._model_info[name]
            if name in self._models:
                del self._models[name]
            return True
        except Exception:
            return False
```

**Step 3: Verify module imports**

Run: `.venv/Scripts/python.exe -c "from swing_trader.services import ModelRegistry; print('OK')"`

**Step 4: Commit**

```bash
git add src/swing_trader/services/
git commit -m "feat: add ModelRegistry service for managing trained models"
```

---

## Task 2: Implement ModelsTab with Model Listing

**Files:**
- Modify: `src/swing_trader/ui/app.py` (ModelsTab class, ~lines 943-955)

**Step 1: Replace ModelsTab placeholder**

Replace the existing `ModelsTab` class with:

```python
class ModelsTab(QWidget):
    """Tab for viewing and managing trained models."""

    model_selected = pyqtSignal(str)  # Emits model name when selected

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
        header.setStyleSheet("color: #FFD93D; font-size: 16px; font-weight: bold;")
        left_layout.addWidget(header)

        self.model_table = QTableWidget()
        self.model_table.setColumnCount(3)
        self.model_table.setHorizontalHeaderLabels(["Name", "Type", "Created"])
        self.model_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.model_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.model_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.model_table.itemSelectionChanged.connect(self.on_selection_changed)
        left_layout.addWidget(self.model_table)

        btn_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_models)
        btn_layout.addWidget(self.refresh_btn)

        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setEnabled(False)
        self.delete_btn.clicked.connect(self.on_delete)
        self.delete_btn.setStyleSheet("background-color: #DA3633;")
        btn_layout.addWidget(self.delete_btn)
        left_layout.addLayout(btn_layout)

        # Right panel - model details
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        details_group = QGroupBox("MODEL DETAILS")
        details_layout = QVBoxLayout(details_group)

        self.details_labels = {}
        for field in ["Name", "Type", "Features", "Created", "Path"]:
            row = QHBoxLayout()
            label = QLabel(f"{field}:")
            label.setStyleSheet("color: #8B949E; font-weight: bold;")
            label.setFixedWidth(80)
            value = QLabel("--")
            value.setStyleSheet("color: #E6EDF3;")
            value.setWordWrap(True)
            row.addWidget(label)
            row.addWidget(value, 1)
            details_layout.addLayout(row)
            self.details_labels[field] = value

        right_layout.addWidget(details_group)

        # Use model button
        self.use_btn = QPushButton("Use This Model")
        self.use_btn.setEnabled(False)
        self.use_btn.clicked.connect(self.on_use_model)
        self.use_btn.setStyleSheet("background-color: #238636; font-size: 14px; padding: 12px;")
        right_layout.addWidget(self.use_btn)

        right_layout.addStretch()

        layout.addWidget(left_panel)
        layout.addWidget(right_panel, 1)

    def refresh_models(self):
        """Reload models from registry."""
        from ..services import ModelRegistry
        self.registry = ModelRegistry()
        self.registry.refresh()

        models = self.registry.list_models()
        self.model_table.setRowCount(len(models))

        for i, info in enumerate(sorted(models, key=lambda x: x.created, reverse=True)):
            self.model_table.setItem(i, 0, QTableWidgetItem(info.name))
            self.model_table.setItem(i, 1, QTableWidgetItem(info.model_type))
            self.model_table.setItem(i, 2, QTableWidgetItem(info.created.strftime("%Y-%m-%d %H:%M")))

        self.delete_btn.setEnabled(False)
        self.use_btn.setEnabled(False)
        self.clear_details()

    def clear_details(self):
        """Clear the details panel."""
        for label in self.details_labels.values():
            label.setText("--")

    def on_selection_changed(self):
        """Handle model selection."""
        selected = self.model_table.selectedItems()
        if not selected:
            self.delete_btn.setEnabled(False)
            self.use_btn.setEnabled(False)
            self.clear_details()
            return

        row = selected[0].row()
        name = self.model_table.item(row, 0).text()
        info = self.registry.get_model_info(name)

        if info:
            self.details_labels["Name"].setText(info.name)
            self.details_labels["Type"].setText(info.model_type)
            self.details_labels["Features"].setText(str(info.feature_count))
            self.details_labels["Created"].setText(info.created.strftime("%Y-%m-%d %H:%M:%S"))
            self.details_labels["Path"].setText(str(info.path))
            self.delete_btn.setEnabled(True)
            self.use_btn.setEnabled(True)

    def on_delete(self):
        """Delete selected model."""
        selected = self.model_table.selectedItems()
        if not selected:
            return

        name = self.model_table.item(selected[0].row(), 0).text()
        if self.registry.delete_model(name):
            self.refresh_models()

    def on_use_model(self):
        """Emit signal to use selected model."""
        selected = self.model_table.selectedItems()
        if selected:
            name = self.model_table.item(selected[0].row(), 0).text()
            self.model_selected.emit(name)
```

**Step 2: Add pyqtSignal import if not present**

Ensure the imports at the top of app.py include:
```python
from PyQt6.QtCore import Qt, QThread, pyqtSignal
```

**Step 3: Verify imports and run**

Run: `.venv/Scripts/python.exe -c "from swing_trader.ui.app import ModelsTab; print('OK')"`

**Step 4: Commit**

```bash
git add src/swing_trader/ui/app.py
git commit -m "feat: implement ModelsTab with model listing and details"
```

---

## Task 3: Add Model Selector to SignalsTab

**Files:**
- Modify: `src/swing_trader/ui/app.py` (SignalsTab class)

**Step 1: Add model selector to SignalsTab.setup_ui()**

Add after the period_combo in the ticker_group section:

```python
        ticker_layout.addWidget(QLabel("Model"))
        self.model_combo = QComboBox()
        self.model_combo.addItem("-- Select Model --")
        ticker_layout.addWidget(self.model_combo)
        self.refresh_model_list()
```

**Step 2: Add refresh_model_list method to SignalsTab**

```python
    def refresh_model_list(self):
        """Populate model dropdown from registry."""
        from ..services import ModelRegistry
        registry = ModelRegistry()

        current = self.model_combo.currentText()
        self.model_combo.clear()
        self.model_combo.addItem("-- Select Model --")

        for info in registry.list_models():
            self.model_combo.addItem(f"{info.name} ({info.model_type})", info.name)

        # Restore selection if possible
        idx = self.model_combo.findText(current)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)

    def set_model(self, model_name: str):
        """Set the selected model by name."""
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == model_name:
                self.model_combo.setCurrentIndex(i)
                break
```

**Step 3: Update on_analyze to use selected model**

Replace the `analyze()` function inside `on_analyze()`:

```python
        model_name = self.model_combo.currentData()
        period = self.period_combo.currentText()

        def analyze(progress_callback=None):
            from ..data.fetcher import StockDataFetcher
            from ..features.indicators import TechnicalIndicators
            from ..services import ModelRegistry

            if progress_callback:
                progress_callback("Fetching data...", 20)

            fetcher = StockDataFetcher()
            indicators = TechnicalIndicators()
            data = fetcher.fetch(ticker, period=period)

            if data is None or data.empty:
                return None, ticker, None

            data = indicators.add_all(data)

            signal_info = None
            if model_name:
                if progress_callback:
                    progress_callback("Generating signal...", 60)
                try:
                    registry = ModelRegistry()
                    model = registry.get_model(model_name)
                    predictions = model.predict(data)
                    probas = model.predict_proba(data)

                    # Handle LSTM shorter output
                    if len(predictions) < len(data):
                        pad = len(data) - len(predictions)
                        predictions = np.concatenate([[0] * pad, predictions])
                        probas = np.vstack([np.full((pad, 3), 1/3), probas])

                    data['signal'] = predictions
                    latest_signal = predictions[-1]
                    latest_proba = probas[-1]
                    confidence = max(latest_proba)

                    signal_info = {
                        'signal': 'BUY' if latest_signal == 1 else 'SELL' if latest_signal == -1 else 'HOLD',
                        'confidence': confidence,
                        'probabilities': {'SELL': latest_proba[0], 'HOLD': latest_proba[1], 'BUY': latest_proba[2]}
                    }
                except Exception as e:
                    print(f"Model prediction error: {e}")

            return data, ticker, signal_info
```

**Step 4: Update on_analyze_complete to show signal**

Replace `on_analyze_complete`:

```python
    def on_analyze_complete(self, result):
        data, ticker, signal_info = result
        self.analyze_btn.setEnabled(True)

        if data is None or data.empty:
            self.status_label.setText(f"No data for {ticker}")
            return

        # Update status with signal info
        if signal_info:
            signal = signal_info['signal']
            conf = signal_info['confidence']
            color = "#00D4AA" if signal == "BUY" else "#FF6B6B" if signal == "SELL" else "#8B949E"
            self.status_label.setText(f"{ticker}: {signal} ({conf:.0%} confidence)")
            self.status_label.setStyleSheet(f"color: {color};")
        else:
            self.status_label.setText(f"Analysis complete for {ticker} (no model selected)")
            self.status_label.setStyleSheet("color: #6E7681;")

        # Update chart
        self.chart.axes.clear()
        self.chart.axes.plot(data.index, data['close'], color='#58A6FF', label='Close')
        if 'sma_20' in data.columns:
            self.chart.axes.plot(data.index, data['sma_20'], color='#FFD93D', label='SMA 20', alpha=0.7)

        # Mark signals on chart if available
        if 'signal' in data.columns and signal_info:
            buy_mask = data['signal'] == 1
            sell_mask = data['signal'] == -1
            if buy_mask.any():
                self.chart.axes.scatter(data.index[buy_mask], data['close'][buy_mask],
                                       color='#00D4AA', marker='^', s=50, label='BUY', zorder=5)
            if sell_mask.any():
                self.chart.axes.scatter(data.index[sell_mask], data['close'][sell_mask],
                                       color='#FF6B6B', marker='v', s=50, label='SELL', zorder=5)

        self.chart.axes.legend(facecolor='#0D1117', edgecolor='#30363D', labelcolor='#8B949E')
        self.chart.axes.set_facecolor('#0D1117')
        self.chart.axes.tick_params(colors='#8B949E')
        self.chart.fig.tight_layout()
        self.chart.draw()
```

**Step 5: Add numpy import at top of app.py if not present**

```python
import numpy as np
```

**Step 6: Verify and commit**

Run: `.venv/Scripts/python.exe -c "from swing_trader.ui.app import SignalsTab; print('OK')"`

```bash
git add src/swing_trader/ui/app.py
git commit -m "feat: add model selector to SignalsTab for ML-based analysis"
```

---

## Task 4: Update Scan to Use Selected Model

**Files:**
- Modify: `src/swing_trader/ui/app.py` (SignalsTab.on_scan method)

**Step 1: Replace on_scan method**

```python
    def on_scan(self):
        self.status_label.setText("Scanning...")
        self.scan_btn.setEnabled(False)

        model_name = self.model_combo.currentData()
        filter_type = self.filter_combo.currentText()

        def scan(progress_callback=None):
            from ..data.fetcher import StockDataFetcher
            from ..features.indicators import TechnicalIndicators
            from ..services import ModelRegistry

            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "WMT"]
            fetcher = StockDataFetcher()
            indicators = TechnicalIndicators()

            registry = None
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

                    if model:
                        # Use ML model
                        predictions = model.predict(data)
                        probas = model.predict_proba(data)

                        # Get latest prediction (handle LSTM shorter output)
                        latest_pred = predictions[-1]
                        latest_proba = probas[-1]

                        signal = "BUY" if latest_pred == 1 else "SELL" if latest_pred == -1 else "HOLD"
                        confidence = max(latest_proba)
                    else:
                        # Fallback to RSI/MACD
                        rsi = latest.get('rsi_14', 50)
                        macd = latest.get('macd', 0)
                        macd_signal = latest.get('macd_signal', 0)

                        if rsi < 30 and macd > macd_signal:
                            signal, confidence = "BUY", min(1.0, (30 - rsi) / 30 + 0.5)
                        elif rsi > 70 and macd < macd_signal:
                            signal, confidence = "SELL", min(1.0, (rsi - 70) / 30 + 0.5)
                        else:
                            signal, confidence = "HOLD", 0.5

                    # Apply filter
                    if filter_type == "BUY Only" and signal != "BUY":
                        continue
                    if filter_type == "SELL Only" and signal != "SELL":
                        continue

                    results.append({
                        "ticker": ticker,
                        "signal": signal,
                        "confidence": confidence,
                        "price": latest['close'],
                        "rsi": latest.get('rsi_14', 0)
                    })
                except Exception as e:
                    print(f"Error scanning {ticker}: {e}")
                    continue

            return results

        self.worker = WorkerThread(scan)
        self.worker.finished.connect(self.on_scan_complete)
        self.worker.error.connect(self.on_error)
        self.worker.progress.connect(lambda msg, pct: self.status_label.setText(msg))
        self.worker.start()
```

**Step 2: Commit**

```bash
git add src/swing_trader/ui/app.py
git commit -m "feat: update scan to use selected ML model"
```

---

## Task 5: Add Model Selector to BacktestTab

**Files:**
- Modify: `src/swing_trader/ui/app.py` (BacktestTab class)

**Step 1: Add model selector to BacktestTab.setup_ui()**

Add after capital_spin in config_group:

```python
        config_layout.addWidget(QLabel("Model"))
        self.model_combo = QComboBox()
        self.model_combo.addItem("-- Select Model --")
        config_layout.addWidget(self.model_combo)
        self.refresh_model_list()
```

**Step 2: Add refresh_model_list and set_model methods to BacktestTab**

```python
    def refresh_model_list(self):
        """Populate model dropdown from registry."""
        from ..services import ModelRegistry
        registry = ModelRegistry()

        current = self.model_combo.currentText()
        self.model_combo.clear()
        self.model_combo.addItem("-- Select Model --")

        for info in registry.list_models():
            self.model_combo.addItem(f"{info.name} ({info.model_type})", info.name)

        idx = self.model_combo.findText(current)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)

    def set_model(self, model_name: str):
        """Set the selected model by name."""
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == model_name:
                self.model_combo.setCurrentIndex(i)
                break
```

**Step 3: Update on_run to use selected model**

Replace the `backtest()` function inside `on_run()`:

```python
        model_name = self.model_combo.currentData()
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

                # Handle LSTM shorter output - pad with zeros
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
```

**Step 4: Commit**

```bash
git add src/swing_trader/ui/app.py
git commit -m "feat: add model selector to BacktestTab"
```

---

## Task 6: Connect Tabs for Model Selection Flow

**Files:**
- Modify: `src/swing_trader/ui/app.py` (SwingTraderApp class)

**Step 1: Store tab references and connect signals**

Update `SwingTraderApp.setup_ui()`:

```python
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

        # Connect model selection from Models tab to other tabs
        self.models_tab.model_selected.connect(self.on_model_selected)

        # Refresh model lists when switching to relevant tabs
        self.tabs.currentChanged.connect(self.on_tab_changed)

        layout.addWidget(self.tabs)

    def on_model_selected(self, model_name: str):
        """Handle model selection from Models tab."""
        self.signals_tab.set_model(model_name)
        self.backtest_tab.set_model(model_name)
        self.tabs.setCurrentWidget(self.signals_tab)

    def on_tab_changed(self, index: int):
        """Refresh model lists when switching tabs."""
        widget = self.tabs.widget(index)
        if hasattr(widget, 'refresh_model_list'):
            widget.refresh_model_list()
        if hasattr(widget, 'refresh_models'):
            widget.refresh_models()
```

**Step 2: Commit**

```bash
git add src/swing_trader/ui/app.py
git commit -m "feat: connect tabs for model selection flow"
```

---

## Task 7: Final Testing and Cleanup

**Step 1: Run full application test**

```bash
.venv/Scripts/python.exe -m swing_trader.ui.app
```

**Manual Test Checklist:**
- [ ] Models tab shows trained models from `models/` directory
- [ ] Selecting a model and clicking "Use This Model" switches to Signals tab
- [ ] Signals tab model dropdown shows available models
- [ ] Analyze with model selected shows ML signal and confidence
- [ ] Chart shows buy/sell markers
- [ ] Scan with model uses ML predictions
- [ ] Backtest tab model dropdown shows available models
- [ ] Backtest with model uses ML predictions

**Step 2: Final commit**

```bash
git add -A
git commit -m "feat: complete model integration across all tabs"
```

---

## Summary

This plan implements:

1. **ModelRegistry** - Central service for loading/managing trained models
2. **ModelsTab** - Full UI for viewing, selecting, and deleting models
3. **SignalsTab Integration** - Model dropdown, ML-based analyze and scan
4. **BacktestTab Integration** - Model dropdown, ML-based backtesting
5. **Cross-Tab Flow** - Model selection in Models tab propagates to other tabs

The architecture leverages the existing `SignalGenerator` pattern while keeping the UI responsive with background workers.
