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
    train_parser.add_argument("--period", default="2y", help="Training data period (ignored if --start/--end given)")
    train_parser.add_argument("--start", help="Training start date (YYYY-MM-DD)")
    train_parser.add_argument("--end", help="Training end date (YYYY-MM-DD)")

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
    bt_parser.add_argument("--period", default="1y", help="Backtest period (ignored if --start/--end given)")
    bt_parser.add_argument("--start", help="Backtest start date (YYYY-MM-DD)")
    bt_parser.add_argument("--end", help="Backtest end date (YYYY-MM-DD)")
    bt_parser.add_argument("--capital", type=float, default=10000)
    bt_parser.add_argument("--check-overlap", action="store_true", help="Check for training data overlap")

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
    elif args.command == "ui":
        run_ui(args)
    else:
        parser.print_help()


def run_train(args):
    """Train models on specified tickers."""
    from swing_trader.data import StockDataFetcher
    from swing_trader.features import TechnicalIndicators, SignalLabeler
    from swing_trader.models import RandomForestModel, XGBoostModel, LSTMModel
    from swing_trader.training.metadata import TrainingMetadataStore, TrainingInfo

    fetcher = StockDataFetcher()
    indicators = TechnicalIndicators()
    labeler = SignalLabeler()

    # Determine date range
    if args.start and args.end:
        print(f"Fetching data for {len(args.tickers)} tickers ({args.start} to {args.end})...")
        fetch_method = lambda ticker: fetcher.fetch_range(ticker, start=args.start, end=args.end)
        train_start = args.start
        train_end = args.end
    else:
        print(f"Fetching data for {len(args.tickers)} tickers (period: {args.period})...")
        fetch_method = lambda ticker: fetcher.fetch(ticker, period=args.period)
        train_start = None
        train_end = None

    all_data = []
    for ticker in args.tickers:
        try:
            df = fetch_method(ticker)
            df = indicators.add_all(df)
            df["target"] = labeler.create_labels(df)
            df = df.dropna()
            all_data.append(df)
            print(f"  {ticker}: {len(df)} samples")
        except Exception as e:
            print(f"  {ticker}: FAILED - {e}")

    if not all_data:
        print("No data collected. Exiting.")
        return

    combined = pd.concat(all_data)
    combined = combined.sort_index()  # Ensure temporal order
    print(f"\nTotal training samples: {len(combined)}")

    # Derive actual date range from data
    if train_start is None:
        train_start = str(combined.index.min().date()) if hasattr(combined.index.min(), 'date') else str(combined.index.min())
    if train_end is None:
        train_end = str(combined.index.max().date()) if hasattr(combined.index.max(), 'date') else str(combined.index.max())

    print(f"Date range: {train_start} to {train_end}")

    exclude_cols = ["target", "open", "high", "low", "close", "volume"]
    feature_cols = [c for c in combined.columns if c not in exclude_cols]
    X = combined[feature_cols]
    y = combined["target"]

    # Temporal split with gap for label leakage prevention
    split_idx = int(len(X) * 0.8)
    gap = labeler.forward_days
    X_train, X_val = X[:split_idx], X[split_idx + gap:]
    y_train, y_val = y[:split_idx], y[split_idx + gap:]

    print(f"Train: {len(X_train)} samples | Gap: {gap} days | Validation: {len(X_val)} samples")

    args.output.mkdir(parents=True, exist_ok=True)

    models_to_train = []
    if args.model in ["rf", "all"]:
        models_to_train.append(("rf", RandomForestModel()))
    if args.model in ["xgb", "all"]:
        models_to_train.append(("xgb", XGBoostModel()))
    if args.model in ["lstm", "all"]:
        models_to_train.append(("lstm", LSTMModel(epochs=30)))

    for name, model in models_to_train:
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        accuracy = (preds == y_val.values[-len(preds):]).mean()
        print(f"  Validation accuracy: {accuracy:.2%}")

        model_path = args.output / f"{name}_model.joblib"
        model.save(model_path)
        print(f"  Saved to {model_path}")

        # Save training metadata
        meta_store = TrainingMetadataStore(models_dir=str(args.output))
        meta_info = TrainingInfo(
            train_start=train_start,
            train_end=train_end,
            tickers=args.tickers,
            model_type=name,
            model_filename=f"{name}_model.joblib",
            forward_days=labeler.forward_days,
        )
        meta_store.save(meta_info)

    print("\nTraining complete!")


def run_tune(args):
    """Tune model hyperparameters."""
    from swing_trader.data import StockDataFetcher
    from swing_trader.features import TechnicalIndicators, SignalLabeler
    from swing_trader.training import HyperparameterTuner, ExperimentTracker

    print(f"Fetching data for {len(args.tickers)} tickers...")
    fetcher = StockDataFetcher()
    indicators = TechnicalIndicators()
    labeler = SignalLabeler()

    all_data = []
    for ticker in args.tickers:
        try:
            df = fetcher.fetch(ticker, period="2y")
            df = indicators.add_all(df)
            df["target"] = labeler.create_labels(df)
            df = df.dropna()
            all_data.append(df)
        except Exception as e:
            print(f"  {ticker}: FAILED - {e}")

    combined = pd.concat(all_data, ignore_index=True)
    exclude_cols = ["target", "open", "high", "low", "close", "volume"]
    feature_cols = [c for c in combined.columns if c not in exclude_cols]
    X = combined[feature_cols]
    y = combined["target"]

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

    # Check for data overlap if requested
    if args.check_overlap:
        from swing_trader.training.metadata import TrainingMetadataStore
        meta_store = TrainingMetadataStore(models_dir=str(args.models))
        info = meta_store.get_latest()
        if info:
            print(f"Training data: {info.train_start} to {info.train_end}")
            print(f"Safe backtest start: {info.safe_backtest_start.isoformat()}")

            if args.start:
                result = meta_store.check_overlap(args.start, args.end or "2099-12-31")
                if result["overlaps"]:
                    print(f"\nWARNING: {result['message']}")
                else:
                    print(f"OK: {result['message']}")
            print()

    # Use date range if provided, otherwise fall back to period
    if args.start and args.end:
        print(f"Backtesting {args.ticker} ({args.start} to {args.end})...")
        df = generator.generate(args.ticker, start=args.start, end=args.end)
    else:
        print(f"Backtesting {args.ticker} (period: {args.period})...")
        df = generator.generate(args.ticker, period=args.period)

    engine = BacktestEngine(initial_capital=args.capital)
    result = engine.run(df)

    print(result.summary())

    if len(result.trades) > 0:
        print("\nRecent trades:")
        print(result.trades.tail(10).to_string())


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
