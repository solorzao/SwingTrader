# Training & Backtesting Professional Overhaul

**Date:** 2026-02-25
**Status:** Approved

## Problem Statement

The swing-trader codebase has critical data leakage issues that produce unreliable backtest results:

1. **UI training path** uses `train_test_split()` which randomly shuffles time-series data
2. **XGBoost HP tuning** uses random split instead of temporal
3. **LSTM scaler** is fitted on full dataset (train + test)
4. **DearPyGUI and CLI backtests** don't enforce train/backtest date separation
5. **No walk-forward validation** — single static model applied to entire period
6. **No statistical rigor** — point estimates with no confidence intervals
7. **Data pipeline duplicated** 3 times with different bugs in each
8. **No data caching** — every run fetches fresh from Yahoo Finance
9. **Configuration scattered** across hardcoded class constants

## Design

### 1. Centralized Configuration (`config.py`)

Dataclasses with sensible defaults. No YAML files — defaults ARE the config, overrides via UI/CLI.

```python
@dataclass FeatureConfig     # indicator params (periods, toggles)
@dataclass LabelConfig       # forward_days, threshold
@dataclass BacktestConfig    # capital, commission, slippage, position_size
@dataclass WalkForwardConfig # mode, train_window, step_size, retrain_every
@dataclass AppConfig         # aggregates all above + paths
```

Config snapshot stored with every saved model for reproducibility.

### 2. Data Pipeline (`data/pipeline.py`, `data/cache.py`)

Single `DataPipeline` class replaces 3 duplicated fetch-indicator-label pipelines:

- `fetch(ticker, period)` — cached OHLCV via parquet files (24h TTL)
- `prepare(ticker, period)` — adds indicators + labels
- `split(df, ratios)` — delegates to TemporalSplitManager

`DataCache` stores parquet files in `data/cache/{ticker}_{period}_{date}.parquet`.

### 3. Temporal Split Manager (`data/splits.py`)

Enforces chronological integrity everywhere:

```python
@dataclass TemporalSplit:
    train, val, test, holdout  # (X, y) tuples
    train_end_date, test_end_date, holdout_start_date  # timestamps
```

- 60/10/10/20 split ratios (configurable)
- ALL splits chronological — no random shuffling
- Scaler/normalizer fit on train only
- Holdout never touched until final evaluation
- Dates stored with model, enforced at backtest time

### 4. Walk-Forward Backtesting (`backtest/walk_forward.py`)

Orchestrates retrain-predict loop:

- For each window: train on [0..T], predict on [T..T+step], advance
- Two modes: expanding window (growing) or rolling window (fixed size)
- model_factory pattern: `(X_train, y_train) -> BaseModel`
- All predictions are truly out-of-sample
- Per-window metrics detect performance degradation

Existing `BacktestEngine` unchanged — handles trade simulation.

### 5. Evaluation Framework (`evaluation/`)

- **ModelEvaluator**: train vs test comparison, overfitting score (flag if >10%)
- **Bootstrap**: 1000-sample bootstrap for Sharpe, return, win rate CIs
- **Regime analysis**: rolling 60-day volatility, high/low vol split, per-regime metrics
- **F1 (macro) replaces accuracy** as primary metric

### New File Structure

```
src/swing_trader/
  config.py                    # NEW
  data/
    pipeline.py                # NEW
    cache.py                   # NEW
    splits.py                  # NEW
  backtest/
    walk_forward.py            # NEW
  evaluation/
    __init__.py                # NEW
    evaluator.py               # NEW
    bootstrap.py               # NEW
    regime.py                  # NEW
```

### Modified Files

- `cli.py` — use DataPipeline, TemporalSplitManager
- `ui/app.py` — use DataPipeline, show new metrics, walk-forward option
- `ui/views/training.py` — remove train_test_split, use TemporalSplitManager
- `training/tuner.py` — use TemporalSplit for all HP tuning
- `models/lstm.py` — scaler fit on train only
- `models/base.py` — store config snapshot on save
- `backtest/engine.py` — accept BacktestConfig, enforce date boundaries
- `features/indicators.py` — accept FeatureConfig
- `features/labeler.py` — accept LabelConfig

## Implementation Phases

1. **Foundations**: config.py, DataPipeline, DataCache, TemporalSplitManager
2. **Leakage fixes**: training.py shuffle, tuner.py XGBoost, lstm.py scaler
3. **Walk-forward engine**: WalkForwardEngine + WalkForwardResult
4. **Evaluation**: ModelEvaluator, bootstrap, regime analysis
5. **Integration**: wire into CLI + UI, update tests
