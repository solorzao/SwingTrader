# Training & Backtesting Overhaul — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate data leakage, add walk-forward backtesting, statistical evaluation, and professional data pipeline architecture.

**Architecture:** Centralized config dataclasses → shared DataPipeline with caching → TemporalSplitManager enforcing chronological splits → WalkForwardEngine for OOS backtesting → ModelEvaluator with bootstrap CIs and regime analysis.

**Tech Stack:** Python 3.11+, pandas, numpy, scikit-learn, PyTorch, XGBoost, pytest

---

## Task 1: Centralized Configuration (`config.py`)

**Files:**
- Create: `src/swing_trader/config.py`
- Test: `tests/test_config.py`

All scattered magic numbers consolidated into typed dataclasses.

---

## Task 2: Data Cache (`data/cache.py`)

**Files:**
- Create: `src/swing_trader/data/cache.py`
- Test: `tests/test_cache.py`

Parquet-based disk cache for OHLCV data. 24h TTL. Keyed by ticker+period+date.

---

## Task 3: Temporal Split Manager (`data/splits.py`)

**Files:**
- Create: `src/swing_trader/data/splits.py`
- Test: `tests/test_splits.py`

Chronological train/val/test/holdout splits. Stores boundary dates. Enforces no shuffling.

---

## Task 4: Data Pipeline (`data/pipeline.py`)

**Files:**
- Create: `src/swing_trader/data/pipeline.py`
- Modify: `src/swing_trader/data/__init__.py`
- Test: `tests/test_pipeline.py`

Single entry point: fetch (cached) → indicators → labels → split.

---

## Task 5: Fix LSTM Scaler Leakage (`models/lstm.py`)

**Files:**
- Modify: `src/swing_trader/models/lstm.py:88-93`

Scaler must be fit on training data only.

---

## Task 6: Fix XGBoost Tuner Leakage (`training/tuner.py`)

**Files:**
- Modify: `src/swing_trader/training/tuner.py:76-83`

Replace `train_test_split` with chronological split.

---

## Task 7: Walk-Forward Engine (`backtest/walk_forward.py`)

**Files:**
- Create: `src/swing_trader/backtest/walk_forward.py`
- Modify: `src/swing_trader/backtest/__init__.py`
- Test: `tests/test_walk_forward.py`

Expanding/rolling window with retrain-predict loop.

---

## Task 8: Bootstrap Confidence Intervals (`evaluation/bootstrap.py`)

**Files:**
- Create: `src/swing_trader/evaluation/__init__.py`
- Create: `src/swing_trader/evaluation/bootstrap.py`
- Test: `tests/test_bootstrap.py`

Bootstrap CIs for Sharpe, returns, win rate.

---

## Task 9: Regime Analysis (`evaluation/regime.py`)

**Files:**
- Create: `src/swing_trader/evaluation/regime.py`
- Test: `tests/test_regime.py`

Rolling volatility regime detection, per-regime metrics.

---

## Task 10: Model Evaluator (`evaluation/evaluator.py`)

**Files:**
- Create: `src/swing_trader/evaluation/evaluator.py`
- Test: `tests/test_evaluator.py`

Train vs test comparison, overfitting detection, F1 as primary metric.

---

## Task 11: Wire into CLI (`cli.py`)

**Files:**
- Modify: `src/swing_trader/cli.py`

Replace ad-hoc pipeline with DataPipeline. Add walk-forward backtest command.

---

## Task 12: Wire into Tuner (`training/tuner.py`)

**Files:**
- Modify: `src/swing_trader/training/tuner.py`

All tuning methods use TemporalSplitManager.

---

## Task 13: Update existing tests

**Files:**
- Modify: `tests/test_models.py`
- Modify: `tests/test_integration.py`

Ensure existing tests pass with new architecture.

---
