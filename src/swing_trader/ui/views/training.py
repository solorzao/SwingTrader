import dearpygui.dearpygui as dpg
import threading
import time
from ..theme import COLORS
from ...data.fetcher import StockDataFetcher
from ...features.indicators import TechnicalIndicators
from ...features.labeler import SignalLabeler
from ...training.tracker import ExperimentTracker
from ...training.tuner import HyperparameterTuner


class TrainingView:
    """Training and hyperparameter tuning view."""

    def __init__(self, parent: str | int):
        self.parent = parent
        self.fetcher = StockDataFetcher()
        self.indicators = TechnicalIndicators()
        self.labeler = SignalLabeler()
        self.tracker = ExperimentTracker()
        self._is_training = False
        self._setup()

    def _setup(self):
        """Build the training view UI."""
        with dpg.group(horizontal=True, parent=self.parent):
            self._create_controls()
            dpg.add_spacer(width=10)
            self._create_metrics()

    def _create_controls(self):
        """Create training controls panel."""
        with dpg.child_window(width=320, height=-1, border=True):
            dpg.add_text("MODEL TRAINING", color=COLORS["accent"])
            dpg.add_spacer(height=15)

            dpg.add_text("Model Type", color=COLORS["text_muted"])
            dpg.add_combo(
                tag="train_model_type",
                items=["Random Forest", "XGBoost (GPU)", "LSTM (CUDA)", "All Models"],
                default_value="Random Forest",
                width=-1
            )

            dpg.add_spacer(height=10)
            dpg.add_text("Training Symbols", color=COLORS["text_muted"])
            dpg.add_input_text(
                tag="train_tickers",
                default_value="AAPL, MSFT, GOOGL",
                width=-1,
                multiline=False
            )

            dpg.add_spacer(height=10)
            dpg.add_text("Data Period", color=COLORS["text_muted"])
            dpg.add_combo(
                tag="train_period",
                items=["1y", "2y", "5y"],
                default_value="2y",
                width=-1
            )

            dpg.add_spacer(height=15)
            dpg.add_button(
                label="Start Training",
                tag="train_start_btn",
                width=-1,
                callback=self._on_train
            )

            dpg.add_spacer(height=25)
            dpg.add_separator()
            dpg.add_spacer(height=15)

            dpg.add_text("HYPERPARAMETER TUNING", color=COLORS["accent"])
            dpg.add_spacer(height=15)

            dpg.add_text("Optimization Trials", color=COLORS["text_muted"])
            dpg.add_slider_int(
                tag="tune_trials",
                default_value=50,
                min_value=10,
                max_value=200,
                width=-1
            )

            dpg.add_spacer(height=10)
            dpg.add_checkbox(tag="tune_gpu", label="Use GPU Acceleration", default_value=True)

            dpg.add_spacer(height=15)
            dpg.add_button(
                label="Auto-Tune with Optuna",
                tag="tune_btn",
                width=-1,
                callback=self._on_tune
            )

            dpg.add_spacer(height=25)
            dpg.add_separator()
            dpg.add_spacer(height=15)

            # Status panel
            dpg.add_text("STATUS", color=COLORS["accent"])
            dpg.add_spacer(height=10)
            dpg.add_text("Ready", tag="train_status", color=COLORS["text_muted"], wrap=300)
            dpg.add_spacer(height=10)
            dpg.add_progress_bar(tag="train_progress", default_value=0, width=-1)

    def _create_metrics(self):
        """Create metrics visualization panel."""
        with dpg.child_window(width=-1, height=-1, border=True):
            dpg.add_text("TRAINING METRICS", color=COLORS["text_secondary"])
            dpg.add_spacer(height=10)

            with dpg.plot(tag="loss_chart", label="Training Progress", height=220, width=-1):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="loss_x")
                dpg.add_plot_axis(dpg.mvYAxis, label="Loss / Score", tag="loss_y")

            dpg.add_spacer(height=15)

            with dpg.plot(tag="acc_chart", label="Accuracy Over Time", height=220, width=-1):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="Iteration", tag="acc_x")
                dpg.add_plot_axis(dpg.mvYAxis, label="Accuracy", tag="acc_y")

            dpg.add_spacer(height=20)

            # Stats row
            with dpg.group(horizontal=True):
                with dpg.child_window(width=180, height=80, border=True):
                    dpg.add_text("SAMPLES", color=COLORS["text_secondary"])
                    dpg.add_text("--", tag="train_samples", color=COLORS["accent"])
                dpg.add_spacer(width=10)
                with dpg.child_window(width=180, height=80, border=True):
                    dpg.add_text("BEST ACCURACY", color=COLORS["text_secondary"])
                    dpg.add_text("--", tag="best_acc", color=COLORS["buy"])
                dpg.add_spacer(width=10)
                with dpg.child_window(width=180, height=80, border=True):
                    dpg.add_text("MODEL", color=COLORS["text_secondary"])
                    dpg.add_text("--", tag="curr_model", color=COLORS["interactive"])

    def _on_train(self, sender=None, data=None):
        """Handle training start."""
        if self._is_training:
            return

        self._is_training = True
        dpg.configure_item("train_start_btn", enabled=False)
        dpg.configure_item("tune_btn", enabled=False)
        dpg.set_value("train_status", "Preparing training data...")
        dpg.set_value("train_progress", 0)

        def run_training():
            try:
                model_type = dpg.get_value("train_model_type")
                tickers_str = dpg.get_value("train_tickers")
                period = dpg.get_value("train_period")

                tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
                if not tickers:
                    dpg.set_value("train_status", "No tickers specified")
                    return

                # Fetch and prepare data
                dpg.set_value("train_status", f"Fetching data for {len(tickers)} tickers...")
                all_data = []
                for i, ticker in enumerate(tickers):
                    dpg.set_value("train_progress", (i / len(tickers)) * 0.3)
                    data = self.fetcher.fetch(ticker, period=period)
                    if data is not None and not data.empty:
                        data = self.indicators.add_all(data)
                        data = self.labeler.create_labels(data)
                        data['ticker'] = ticker
                        all_data.append(data)

                if not all_data:
                    dpg.set_value("train_status", "No data fetched")
                    return

                import pandas as pd
                combined = pd.concat(all_data, ignore_index=True)
                combined = combined.dropna()

                dpg.set_value("train_samples", str(len(combined)))

                # Prepare features and labels
                feature_cols = [c for c in combined.columns if c not in
                               ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'ticker', 'label', 'forward_return']]
                X = combined[feature_cols].values
                y = combined['label'].values

                dpg.set_value("train_status", f"Training with {len(X)} samples...")
                dpg.set_value("train_progress", 0.4)

                # Train based on model type
                models_to_train = []
                if model_type == "All Models":
                    models_to_train = ["Random Forest", "XGBoost (GPU)", "LSTM (CUDA)"]
                else:
                    models_to_train = [model_type]

                accuracies = []
                loss_data = []
                acc_data = []

                for idx, model_name in enumerate(models_to_train):
                    dpg.set_value("curr_model", model_name)
                    dpg.set_value("train_status", f"Training {model_name}...")

                    progress_base = 0.4 + (idx / len(models_to_train)) * 0.5

                    # Start MLflow run
                    run_id = self.tracker.start_run(
                        run_name=f"{model_name}_{time.strftime('%Y%m%d_%H%M%S')}",
                        tags={"model_type": model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}
                    )

                    try:
                        if model_name == "Random Forest":
                            from ...models.random_forest import RandomForestModel
                            model = RandomForestModel(n_estimators=100)
                            model.fit(X, y)
                            accuracy = self._evaluate_model(model, X, y)

                        elif model_name == "XGBoost (GPU)":
                            from ...models.xgboost_model import XGBoostModel
                            model = XGBoostModel(n_estimators=100, use_gpu=True)
                            model.fit(X, y)
                            accuracy = self._evaluate_model(model, X, y)

                        elif model_name == "LSTM (CUDA)":
                            from ...models.lstm import LSTMModel
                            model = LSTMModel(hidden_size=64, num_layers=2)
                            # LSTM needs sequence data
                            model.fit(X, y, epochs=50)
                            accuracy = self._evaluate_model(model, X, y)

                        accuracies.append(accuracy)
                        loss_data.append(1 - accuracy)
                        acc_data.append(accuracy)

                        # Log to MLflow
                        self.tracker.log_metrics({"accuracy": accuracy})
                        self.tracker.log_params({
                            "tickers": ",".join(tickers),
                            "period": period,
                            "samples": len(X)
                        })

                        # Save model
                        from pathlib import Path
                        import joblib
                        models_dir = Path("models")
                        models_dir.mkdir(exist_ok=True)

                        model_filename = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("_gpu", "").replace("_cuda", "")
                        model.save(models_dir / f"{model_filename}.joblib")

                        dpg.set_value("train_progress", progress_base + 0.15)

                    finally:
                        self.tracker.end_run()

                # Update charts
                self._update_training_charts(loss_data, acc_data, models_to_train)

                best_acc = max(accuracies) if accuracies else 0
                dpg.set_value("best_acc", f"{best_acc:.1%}")
                dpg.set_value("train_status", f"Training complete! Best accuracy: {best_acc:.1%}")
                dpg.set_value("train_progress", 1.0)

            except Exception as e:
                dpg.set_value("train_status", f"Error: {str(e)[:60]}")
                import traceback
                traceback.print_exc()
            finally:
                self._is_training = False
                dpg.configure_item("train_start_btn", enabled=True)
                dpg.configure_item("tune_btn", enabled=True)

        thread = threading.Thread(target=run_training, daemon=True)
        thread.start()

    def _evaluate_model(self, model, X, y):
        """Evaluate model accuracy."""
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        predictions = model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        return accuracy

    def _update_training_charts(self, loss_data, acc_data, model_names):
        """Update the training charts."""
        try:
            # Clear existing series
            for tag in ["loss_series", "acc_series"]:
                if dpg.does_item_exist(tag):
                    dpg.delete_item(tag)

            x_data = list(range(len(loss_data)))

            # Add loss series
            dpg.add_bar_series(
                x_data, loss_data,
                label="Loss (1-Accuracy)",
                parent="loss_y",
                tag="loss_series"
            )

            # Add accuracy series
            dpg.add_bar_series(
                x_data, acc_data,
                label="Accuracy",
                parent="acc_y",
                tag="acc_series"
            )

            dpg.fit_axis_data("loss_x")
            dpg.fit_axis_data("loss_y")
            dpg.fit_axis_data("acc_x")
            dpg.fit_axis_data("acc_y")

        except Exception as e:
            print(f"Chart update error: {e}")

    def _on_tune(self, sender=None, data=None):
        """Handle tuning start."""
        if self._is_training:
            return

        self._is_training = True
        dpg.configure_item("train_start_btn", enabled=False)
        dpg.configure_item("tune_btn", enabled=False)
        dpg.set_value("train_status", "Starting hyperparameter tuning...")
        dpg.set_value("train_progress", 0)

        def run_tuning():
            try:
                model_type = dpg.get_value("train_model_type")
                tickers_str = dpg.get_value("train_tickers")
                period = dpg.get_value("train_period")
                n_trials = dpg.get_value("tune_trials")
                use_gpu = dpg.get_value("tune_gpu")

                tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
                if not tickers:
                    dpg.set_value("train_status", "No tickers specified")
                    return

                # Fetch data
                dpg.set_value("train_status", "Fetching data for tuning...")
                all_data = []
                for ticker in tickers:
                    data = self.fetcher.fetch(ticker, period=period)
                    if data is not None and not data.empty:
                        data = self.indicators.add_all(data)
                        data = self.labeler.create_labels(data)
                        all_data.append(data)

                if not all_data:
                    dpg.set_value("train_status", "No data fetched")
                    return

                import pandas as pd
                combined = pd.concat(all_data, ignore_index=True).dropna()

                feature_cols = [c for c in combined.columns if c not in
                               ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'label', 'forward_return']]
                X = combined[feature_cols].values
                y = combined['label'].values

                dpg.set_value("train_samples", str(len(combined)))

                # Run tuning
                tuner = HyperparameterTuner()
                best_params = None
                best_score = 0

                if model_type in ["Random Forest", "All Models"]:
                    dpg.set_value("train_status", f"Tuning Random Forest ({n_trials} trials)...")
                    dpg.set_value("curr_model", "Random Forest")
                    best_params, best_score = tuner.tune_random_forest(X, y, n_trials=n_trials)
                    dpg.set_value("train_progress", 0.33 if model_type == "All Models" else 1.0)

                if model_type in ["XGBoost (GPU)", "All Models"]:
                    dpg.set_value("train_status", f"Tuning XGBoost ({n_trials} trials)...")
                    dpg.set_value("curr_model", "XGBoost")
                    params, score = tuner.tune_xgboost(X, y, n_trials=n_trials, use_gpu=use_gpu)
                    if score > best_score:
                        best_params, best_score = params, score
                    dpg.set_value("train_progress", 0.66 if model_type == "All Models" else 1.0)

                if model_type in ["LSTM (CUDA)", "All Models"]:
                    dpg.set_value("train_status", f"Tuning LSTM ({n_trials} trials)...")
                    dpg.set_value("curr_model", "LSTM")
                    params, score = tuner.tune_lstm(X, y, n_trials=min(n_trials, 20))  # LSTM tuning is slow
                    if score > best_score:
                        best_params, best_score = params, score
                    dpg.set_value("train_progress", 1.0)

                dpg.set_value("best_acc", f"{best_score:.1%}")
                dpg.set_value("train_status", f"Tuning complete! Best score: {best_score:.1%}")

            except Exception as e:
                dpg.set_value("train_status", f"Tuning error: {str(e)[:50]}")
                import traceback
                traceback.print_exc()
            finally:
                self._is_training = False
                dpg.configure_item("train_start_btn", enabled=True)
                dpg.configure_item("tune_btn", enabled=True)

        thread = threading.Thread(target=run_tuning, daemon=True)
        thread.start()
