# Swing Trader

ML-powered swing trading signal generator for US equities. Uses ensemble machine learning models (RandomForest, XGBoost, LSTM) to generate BUY/SELL/HOLD signals.

## Features

- **Multiple ML Models**: RandomForest, XGBoost (GPU), LSTM (CUDA) with versioned model storage
- **Configurable Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, OBV, SMA/EMA, Stochastic with adjustable parameters
- **Hyperparameter Tuning**: Optuna-based Bayesian optimization with TimeSeriesSplit CV
- **Experiment Tracking**: MLflow integration with local UI for comparing runs
- **Backtesting Engine**: Walk-forward testing with mark-to-market equity, trade markers, and performance metrics
- **Quick Scan**: Scan S&P 500 stocks with full probability breakdown
- **Desktop UI**: PyQt6 interface with dark trading terminal theme and global model selector

## Installation

### Prerequisites

- Python 3.11+ (tested on Python 3.12)
- NVIDIA GPU with CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/solorzao/SwingTrader.git
cd SwingTrader

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate

# Install the package
pip install -e .
```

## Usage

### Quick Start (Windows)

Double-click `SwingTrader.bat` to launch the desktop UI, or use the CLI:

```powershell
# From project directory
.\swing-trader.bat signal --ticker AAPL
```

### CLI Commands

```bash
# Train models on specific tickers
swing-trader train --tickers AAPL MSFT GOOGL --model all

# Train a specific model
swing-trader train --tickers AAPL --model rf        # RandomForest
swing-trader train --tickers AAPL --model xgboost   # XGBoost (GPU)
swing-trader train --tickers AAPL --model lstm      # LSTM (CUDA)

# Tune hyperparameters with Optuna
swing-trader tune --tickers AAPL --model xgboost --n-trials 50

# Generate signals for a single ticker
swing-trader signal --ticker AAPL

# Scan multiple tickers for opportunities
swing-trader scan

# Run backtests
swing-trader backtest --ticker AAPL --model xgboost

# Launch the desktop UI
swing-trader ui
```

## Desktop UI

The UI provides four main tabs with a global model selector in the header:

### Global Model Selector
Select your active model from the header dropdown. The selected model is used across all tabs (Signals, Backtest, Quick Scan). Models are named with timestamps (e.g., `xgboost_20260131_143025`) so you can train multiple versions.

### Signals Tab
- Enter a ticker symbol to analyze
- View current BUY/SELL/HOLD signal with confidence scores
- See price chart with technical indicators
- Full probability breakdown (P(Buy), P(Hold), P(Sell))

### Training Tab
- **Training Symbols**: Comma-separated tickers for training data
- **Data Period**: 3mo, 6mo, 1y, or 2y of historical data
- **Feature Selection**: Toggle individual indicators (SMA, EMA, RSI, MACD, Bollinger, ATR, OBV, Stochastic)
- **Feature Parameters**: Adjust RSI period, SMA periods, EMA periods
- **Hyperparameters**: Number of trees, max depth, min samples split (manual mode)
- **Auto-Tune (Optuna)**: Enable automatic hyperparameter optimization
  - Set number of trials (10-200)
  - Uses Bayesian optimization with cross-validation
  - Disables manual hyperparameters when enabled
- **MLflow UI**: Launch button to open experiment tracking dashboard
- **Metrics Display**: Accuracy, Precision, Recall, F1 Score, AUC
- **Visualizations**: Feature importance chart and confusion matrix

### Backtest Tab
- Configure symbol, period, and initial capital
- **Equity Curve**: Mark-to-market chart showing unrealized P&L during open positions
- **Trade Markers**: Green triangles (long entry), red triangles (short entry), squares (exits)
- **Performance Metrics**: Total return, Sharpe ratio, max drawdown, win rate, trade count
- **Trade History**: Detailed table of all trades with entry/exit prices and P&L

### Models Tab
- View all trained models with timestamps
- See performance metrics (Accuracy, F1) for each model
- Model details: type, feature count, creation date
- Multi-select deletion (Ctrl+Click or Shift+Click to select multiple)
- Set any model as the active model for use across tabs

### Quick Scan
- **Stock Lists**: FAANG, Tech Leaders, S&P 500 Top 20, or custom tickers
- **Output Columns**: Ticker, Signal, Confidence, Price, RSI, P(Buy), P(Hold), P(Sell), Trend
- Scan 20+ stocks for trading opportunities

## Architecture

```
src/swing_trader/
├── data/
│   └── fetcher.py              # Yahoo Finance data fetching
├── features/
│   ├── indicators.py           # Configurable technical indicators
│   └── labeler.py              # Signal label generation
├── models/
│   ├── base.py                 # Abstract model interface with metrics
│   ├── random_forest.py        # RandomForest classifier
│   ├── xgboost_model.py        # XGBoost with GPU support
│   ├── lstm.py                 # PyTorch LSTM with CUDA
│   └── ensemble.py             # Ensemble voting/averaging
├── backtest/
│   └── engine.py               # Mark-to-market backtesting
├── services/
│   ├── model_registry.py       # Model loading and management
│   └── mlflow_tracking.py      # MLflow experiment tracking
├── signals/
│   └── generator.py            # Signal generation pipeline
├── ui/
│   └── app.py                  # Main PyQt6 application
└── cli.py                      # Command-line interface
```

## Models

### Signal Classification

Models classify each day as:
- **BUY (1)**: Expected return > threshold (default 2%)
- **HOLD (0)**: Expected return between -threshold and +threshold
- **SELL (-1)**: Expected return < -threshold

### Model Versioning

Models are saved with timestamps: `{model_type}_{YYYYMMDD}_{HHMMSS}.joblib`

This allows you to:
- Train multiple versions of the same model type
- Compare performance across training runs
- Keep the best performing models

### Configurable Feature Set

All indicators can be toggled on/off during training:

| Indicator | Parameters | Default |
|-----------|------------|---------|
| SMA | Periods | 10, 20, 50 |
| EMA | Periods | 12, 26 |
| RSI | Period | 14 |
| MACD | Fast, Slow, Signal | 12, 26, 9 |
| Bollinger Bands | Period, Std Dev | 20, 2 |
| ATR | Period | 14 |
| OBV | - | - |
| Stochastic | K, D periods | 14, 3 |

### GPU Acceleration

- **XGBoost**: Uses `tree_method="gpu_hist"` for GPU-accelerated training
- **LSTM**: PyTorch with CUDA support for GPU tensor operations

## MLflow Tracking

Experiments are tracked in the local `mlruns/` directory.

**From the UI**: Click "Launch MLflow UI" button in the Training tab.

**From command line**:
```bash
mlflow ui --backend-store-uri mlruns
```

Then open http://localhost:5000 in your browser to:
- Compare training runs side-by-side
- View metrics history (accuracy, F1, loss curves)
- See hyperparameters used for each run
- Download model artifacts

## File Storage

The following directories contain user-specific data and are git-ignored:

| Directory | Contents |
|-----------|----------|
| `models/` | Trained model files (.joblib) |
| `mlruns/` | MLflow experiment tracking data |
| `mlartifacts/` | MLflow model artifacts |
| `data/` | Cached price data |

## License

MIT
