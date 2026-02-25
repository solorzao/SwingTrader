import argparse
import sys
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="ML-powered swing trading signals",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument("--tickers", nargs="+", required=True, help="Tickers to train on")
    train_parser.add_argument("--model", choices=["rf", "xgb", "lstm", "all"], default="all")
    train_parser.add_argument("--output", type=Path, default=Path("data/models"))
    train_parser.add_argument("--period", default="2y", help="Training data period")

    # Tune command
    tune_parser = subparsers.add_parser("tune", help="Tune hyperparameters")
    tune_parser.add_argument("--tickers", nargs="+", required=True)
    tune_parser.add_argument("--model", choices=["rf", "xgb", "lstm"], required=True)
    tune_parser.add_argument("--trials", type=int, default=50)
    tune_parser.add_argument("--output", type=Path, default=Path("data/models"))

    # Signal command
    signal_parser = subparsers.add_parser("signal", help="Generate signals")
    signal_parser.add_argument("ticker", help="Stock ticker")
    signal_parser.add_argument("--models", type=Path, default=Path("data/models"))
    signal_parser.add_argument("--period", default="6mo")

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan multiple tickers")
    scan_parser.add_argument("--tickers", nargs="+", help="Tickers to scan")
    scan_parser.add_argument("--sp500", action="store_true", help="Scan S&P 500")
    scan_parser.add_argument("--models", type=Path, default=Path("data/models"))
    scan_parser.add_argument("--min-confidence", type=float, default=0.6)
    scan_parser.add_argument("--filter", choices=["buy", "sell", "all"], default="all")

    # Backtest command
    bt_parser = subparsers.add_parser("backtest", help="Backtest strategy")
    bt_parser.add_argument("ticker", help="Stock ticker")
    bt_parser.add_argument("--models", type=Path, default=Path("data/models"))
    bt_parser.add_argument("--period", default="1y")
    bt_parser.add_argument("--capital", type=float, default=10000)
    bt_parser.add_argument("--walk-forward", action="store_true", help="Use walk-forward backtesting")
    bt_parser.add_argument("--wf-mode", choices=["expanding", "rolling"], default="expanding")
    bt_parser.add_argument("--wf-window", type=int, default=252, help="Training window days")
    bt_parser.add_argument("--wf-step", type=int, default=21, help="Step size in days")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model with train/test comparison")
    eval_parser.add_argument("--tickers", nargs="+", required=True)
    eval_parser.add_argument("--model", choices=["rf", "xgb", "lstm"], default="rf")
    eval_parser.add_argument("--period", default="2y")

    # UI command
    ui_parser = subparsers.add_parser("ui", help="Launch desktop application")

    args = parser.parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "tune":
        run_tune(args)
    elif args.command == "signal":
        run_signal(args)
    elif args.command == "scan":
        run_scan(args)
    elif args.command == "backtest":
        run_backtest(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "ui":
        run_ui(args)
    else:
        parser.print_help()


def run_train(args):
    """Train models on specified tickers using DataPipeline."""
    from swing_trader.data.pipeline import DataPipeline
    from swing_trader.models import RandomForestModel, XGBoostModel, LSTMModel
    from swing_trader.evaluation.evaluator import ModelEvaluator

    pipeline = DataPipeline()
    evaluator = ModelEvaluator()

    print(f"Fetching and preparing data for {len(args.tickers)} tickers...")
    split = pipeline.prepare_and_split(args.tickers, period=args.period)
    print(f"  Train: {split.train_size}  Val: {split.val_size}  "
          f"Test: {split.test_size}  Holdout: {split.holdout_size}")
    print(f"  Train ends: {split.train_end_date.date()}  "
          f"Test ends: {split.test_end_date.date()}")

    args.output.mkdir(parents=True, exist_ok=True)

    models_to_train = []
    if args.model in ["rf", "all"]:
        models_to_train.append(("rf", RandomForestModel()))
    if args.model in ["xgb", "all"]:
        models_to_train.append(("xgb", XGBoostModel()))
    if args.model in ["lstm", "all"]:
        models_to_train.append(("lstm", LSTMModel(epochs=30)))

    feature_config = pipeline.feature_config.to_indicator_config()

    for name, model in models_to_train:
        print(f"\nTraining {name}...")
        if name == "lstm":
            model.fit(split.X_train, split.y_train, scaler_X=split.X_train)
        else:
            model.fit(split.X_train, split.y_train)

        # Evaluate with train/test comparison
        report = evaluator.evaluate(model, split)
        print(report.summary())

        # Save with metadata
        metrics = {
            "accuracy": report.test_metrics.accuracy,
            "f1_macro": report.test_metrics.f1_macro,
            "training_end_date": split.train_end_date.isoformat(),
            "test_end_date": split.test_end_date.isoformat(),
        }
        model_path = args.output / f"{name}_model.joblib"
        model.save(model_path, metrics=metrics, feature_config=feature_config)
        print(f"  Saved to {model_path}")

    print("\nTraining complete!")


def run_tune(args):
    """Tune model hyperparameters using DataPipeline."""
    from swing_trader.data.pipeline import DataPipeline
    from swing_trader.training import HyperparameterTuner, ExperimentTracker

    pipeline = DataPipeline()

    print(f"Fetching and preparing data for {len(args.tickers)} tickers...")
    combined = pipeline.prepare_multiple(args.tickers)
    X, y = pipeline.get_features_and_target(combined)
    print(f"Total samples: {len(X)}")

    tracker = ExperimentTracker()
    tuner = HyperparameterTuner(tracker=tracker)

    print(f"\nTuning {args.model} with {args.trials} trials...")

    if args.model == "rf":
        result = tuner.tune_random_forest(X, y, n_trials=args.trials)
    elif args.model == "xgb":
        result = tuner.tune_xgboost(X, y, n_trials=args.trials)
    elif args.model == "lstm":
        result = tuner.tune_lstm(X, y, n_trials=args.trials)

    print(f"\nBest parameters: {result['best_params']}")
    print(f"Best score: {result['best_value']:.4f}")


def run_signal(args):
    """Generate signals for a ticker."""
    from swing_trader.signals import SignalGenerator

    generator = SignalGenerator(model_dir=args.models)

    if not generator.models:
        print("No models found. Run 'swing-trader train' first.")
        return

    print(f"Generating signals for {args.ticker}...")
    df = generator.generate(args.ticker, period=args.period)

    latest = df.tail(10)
    signal_map = {-1: "SELL", 0: "HOLD", 1: "BUY"}

    print(f"\nLatest signals for {args.ticker}:")
    print("-" * 60)

    for idx, row in latest.iterrows():
        signal = signal_map[int(row["signal"])]
        conf = row["confidence"]
        price = row["close"]
        print(f"{idx.strftime('%Y-%m-%d')}  ${price:>8.2f}  {signal:>4}  (conf: {conf:.0%})")

    print("-" * 60)
    print(f"Current: {signal_map[int(df.iloc[-1]['signal'])]} (confidence: {df.iloc[-1]['confidence']:.0%})")


def run_scan(args):
    """Scan multiple tickers for signals."""
    from swing_trader.signals import SignalGenerator

    if args.sp500:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
                   "JPM", "V", "JNJ", "WMT", "PG", "MA", "HD", "CVX"]
    else:
        tickers = args.tickers or []

    if not tickers:
        print("No tickers specified. Use --tickers or --sp500")
        return

    generator = SignalGenerator(model_dir=args.models)

    if not generator.models:
        print("No models found. Run 'swing-trader train' first.")
        return

    print(f"Scanning {len(tickers)} tickers...")
    results = generator.scan_universe(
        tickers,
        min_confidence=args.min_confidence,
        signal_filter=args.filter
    )

    if results.empty:
        print("No signals found matching criteria.")
        return

    signal_map = {-1: "SELL", 0: "HOLD", 1: "BUY"}

    print(f"\nSignals (confidence >= {args.min_confidence:.0%}):")
    print("-" * 50)

    for _, row in results.iterrows():
        signal = signal_map[int(row["signal"])]
        print(f"{row['ticker']:>6}  ${row['close']:>8.2f}  {signal:>4}  (conf: {row['confidence']:.0%})")


def run_backtest(args):
    """Backtest strategy on a ticker."""
    from swing_trader.signals import SignalGenerator
    from swing_trader.backtest import BacktestEngine

    generator = SignalGenerator(model_dir=args.models)

    if not generator.models:
        print("No models found. Run 'swing-trader train' first.")
        return

    if args.walk_forward:
        _run_walk_forward_backtest(args, generator)
    else:
        _run_static_backtest(args, generator)


def _run_static_backtest(args, generator):
    """Standard backtest with date enforcement."""
    from swing_trader.backtest import BacktestEngine

    print(f"Backtesting {args.ticker}...")
    df = generator.generate(args.ticker, period=args.period)

    # Enforce post-training date filtering
    for model in generator.models.values():
        if hasattr(model, 'metrics') and model.metrics:
            test_end_str = model.metrics.get('test_end_date')
            if test_end_str:
                try:
                    test_end = pd.Timestamp(test_end_str)
                    pre_len = len(df)
                    df = df[df.index > test_end]
                    print(f"  Filtered to post-training data: {pre_len} -> {len(df)} rows")
                except Exception:
                    pass
            break

    if len(df) < 5:
        print("Not enough post-training data for backtesting.")
        return

    engine = BacktestEngine(initial_capital=args.capital)
    result = engine.run(df)

    print(result.summary())

    if len(result.trades) > 0:
        print("\nRecent trades:")
        print(result.trades.tail(10).to_string())


def _run_walk_forward_backtest(args, generator):
    """Walk-forward backtest with periodic retraining."""
    from swing_trader.data.pipeline import DataPipeline
    from swing_trader.backtest.walk_forward import WalkForwardEngine
    from swing_trader.config import WalkForwardConfig, BacktestConfig
    from swing_trader.models import RandomForestModel

    print(f"Walk-forward backtesting {args.ticker}...")
    pipeline = DataPipeline()
    df = pipeline.prepare(args.ticker, period=args.period)
    X, y = pipeline.get_features_and_target(df)

    wf_config = WalkForwardConfig(
        mode=args.wf_mode,
        train_window_days=args.wf_window,
        step_days=args.wf_step,
    )
    bt_config = BacktestConfig(initial_capital=args.capital)

    def model_factory(X_train, y_train):
        model = RandomForestModel(n_estimators=100)
        model.fit(X_train, y_train)
        return model

    engine = WalkForwardEngine(wf_config=wf_config, bt_config=bt_config)
    result = engine.run(X, y, df["close"], model_factory)

    print(f"\nWalk-Forward Results ({result.n_windows} windows, {wf_config.mode} mode)")
    print(f"Mean OOS accuracy: {result.mean_oos_accuracy:.2%}")
    print(result.backtest_result.summary())

    # Per-window accuracy breakdown
    print("\nPer-window accuracy:")
    for w in result.window_results:
        print(f"  Window {w.window_idx}: {w.accuracy:.2%} "
              f"({w.test_start.date()} to {w.test_end.date()}, "
              f"train={w.train_size}, test={w.test_size})")


def run_evaluate(args):
    """Evaluate model with train/test comparison and bootstrap CIs."""
    from swing_trader.data.pipeline import DataPipeline
    from swing_trader.models import RandomForestModel, XGBoostModel, LSTMModel
    from swing_trader.evaluation.evaluator import ModelEvaluator
    from swing_trader.backtest import BacktestEngine

    pipeline = DataPipeline()
    evaluator = ModelEvaluator()

    print(f"Preparing data for {len(args.tickers)} tickers...")
    split = pipeline.prepare_and_split(args.tickers, period=args.period)

    model_map = {"rf": RandomForestModel, "xgb": XGBoostModel, "lstm": LSTMModel}
    model = model_map[args.model]()

    print(f"Training {args.model}...")
    if args.model == "lstm":
        model.fit(split.X_train, split.y_train, scaler_X=split.X_train)
    else:
        model.fit(split.X_train, split.y_train)

    report = evaluator.evaluate(model, split)
    print(report.summary())

    print(f"\nClass distribution (train): {report.class_distribution_train}")
    print(f"Class distribution (test):  {report.class_distribution_test}")


def run_ui(args):
    """Launch the desktop application."""
    try:
        from swing_trader.ui.app import main as ui_main
        ui_main()
    except ImportError as e:
        print(f"Failed to load UI: {e}")
        print("Make sure dearpygui is installed: pip install dearpygui")
    except Exception as e:
        print(f"Error launching UI: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
