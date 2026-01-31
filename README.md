# Swing Trader

ML-powered swing trading signal generator for US equities. Uses ensemble machine learning models (RandomForest, XGBoost, LSTM) to generate BUY/SELL/HOLD signals.

## Features

- **Multiple ML Models**: RandomForest, XGBoost (GPU), LSTM (CUDA), and Ensemble voting
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, OBV, SMA/EMA, ROC
- **Hyperparameter Tuning**: Optuna-based Bayesian optimization with TimeSeriesSplit CV
- **Experiment Tracking**: MLflow integration for model versioning and metrics
- **Backtesting Engine**: Walk-forward testing with Sharpe ratio, max drawdown, win rate
- **Desktop UI**: PyQt6 interface with dark trading terminal theme
- **CLI Interface**: Full command-line access to all features

## Installation

### Prerequisites

- Python 3.11+ (tested on Python 3.14)
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

### Desktop UI

The UI provides four main views:

1. **Signals**: View current signals with confidence scores and price charts
2. **Training**: Train models with progress tracking and parameter configuration
3. **Backtest**: Run backtests and view performance metrics
4. **Models**: Manage trained models via MLflow registry

## Architecture

```
src/swing_trader/
├── data/
│   └── fetcher.py          # Yahoo Finance data fetching
├── features/
│   ├── indicators.py       # Technical indicator calculations
│   └── labeler.py          # Signal label generation
├── models/
│   ├── base.py             # Abstract model interface
│   ├── random_forest.py    # RandomForest classifier
│   ├── xgboost_model.py    # XGBoost with GPU support
│   ├── lstm.py             # PyTorch LSTM with CUDA
│   └── ensemble.py         # Ensemble voting/averaging
├── backtest/
│   └── engine.py           # Backtesting with metrics
├── training/
│   ├── tracker.py          # MLflow experiment tracking
│   └── tuner.py            # Optuna hyperparameter tuning
├── signals/
│   └── generator.py        # Signal generation pipeline
├── ui/
│   └── app.py              # Main PyQt6 application
└── cli.py                  # Command-line interface
```

## Models

### Signal Classification

Models classify each day as:
- **BUY (1)**: Expected return > threshold (default 2%)
- **HOLD (0)**: Expected return between -threshold and +threshold
- **SELL (-1)**: Expected return < -threshold

### Feature Set

- Price-based: SMA (20, 50), EMA (12, 26), ROC
- Momentum: RSI (14), MACD (12, 26, 9)
- Volatility: Bollinger Bands (20, 2), ATR (14)
- Volume: OBV, Volume SMA

### GPU Acceleration

- **XGBoost**: Uses `tree_method="gpu_hist"` for GPU-accelerated training
- **LSTM**: PyTorch with CUDA support for GPU tensor operations

## MLflow Tracking

Experiments are tracked in the local `mlruns/` directory. To view:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

## License

MIT
