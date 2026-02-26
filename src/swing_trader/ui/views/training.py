import dearpygui.dearpygui as dpg
import threading
import time
from ..theme import COLORS
from ...data.pipeline import DataPipeline
from ...evaluation.evaluator import ModelEvaluator
from ...training.tracker import ExperimentTracker
from ...training.tuner import HyperparameterTuner


def detect_gpu() -> dict:
    """Detect available GPU for training."""
    result = {"cuda": False, "cuda_name": None, "xgboost_gpu": False}

    # Check PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            result["cuda"] = True
            result["cuda_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    # Check XGBoost GPU
    try:
        import xgboost as xgb
        # XGBoost GPU is available if CUDA is available
        result["xgboost_gpu"] = result["cuda"]
    except ImportError:
        pass

    return result


class TrainingView:
    """Training and hyperparameter tuning view."""

    def __init__(self, parent: str | int):
        self.parent = parent
        self.tracker = ExperimentTracker()
        self._is_training = False
        self._gpu_info = detect_gpu()
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
                width=-1,
                callback=self._on_model_type_change
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

            # GPU checkbox - only enabled if GPU is available
            gpu_available = self._gpu_info["cuda"]
            gpu_label = "Use GPU Acceleration"
            if gpu_available:
                gpu_label += f" ({self._gpu_info['cuda_name']})"
            else:
                gpu_label += " (No GPU detected)"

            dpg.add_checkbox(
                tag="tune_gpu",
                label=gpu_label,
                default_value=gpu_available,
                enabled=gpu_available
            )

            dpg.add_spacer(height=15)

            # Model-specific hyperparameter section
            dpg.add_text("Model Parameters", color=COLORS["text_muted"])
            dpg.add_spacer(height=5)

            # Container for dynamic hyperparameter controls
            self._hyperparam_container = dpg.add_child_window(height=120, border=False, tag="hyperparam_container")
            self._update_hyperparameter_ui()

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

                # Fetch and prepare data using DataPipeline (temporal splits)
                dpg.set_value("train_status", f"Fetching data for {len(tickers)} tickers...")
                pipeline = DataPipeline()

                try:
                    split = pipeline.prepare_and_split(tickers, period=period)
                except Exception as e:
                    dpg.set_value("train_status", f"Data error: {str(e)[:60]}")
                    return

                dpg.set_value("train_progress", 0.3)
                feature_config = pipeline.feature_config.to_indicator_config()
                dpg.set_value("train_samples", f"{split.train_size} train / {split.test_size} test")

                dpg.set_value("train_status", f"Training on {split.train_size} samples...")
                dpg.set_value("train_progress", 0.4)

                # Train based on model type
                models_to_train = []
                if model_type == "All Models":
                    models_to_train = ["Random Forest", "XGBoost (GPU)", "LSTM (CUDA)"]
                else:
                    models_to_train = [model_type]

                evaluator = ModelEvaluator()
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
                            model.fit(split.X_train, split.y_train)

                        elif model_name == "XGBoost (GPU)":
                            from ...models.xgboost_model import XGBoostModel
                            model = XGBoostModel(n_estimators=100, use_gpu=True)
                            model.fit(split.X_train, split.y_train)

                        elif model_name == "LSTM (CUDA)":
                            from ...models.lstm import LSTMModel
                            model = LSTMModel(hidden_size=64, num_layers=2, epochs=50)
                            model.fit(split.X_train, split.y_train, scaler_X=split.X_train)

                        # Evaluate with proper train/test comparison
                        report = evaluator.evaluate(model, split)
                        accuracy = report.test_metrics.accuracy

                        accuracies.append(accuracy)
                        loss_data.append(1 - accuracy)
                        acc_data.append(accuracy)

                        # Log to MLflow
                        self.tracker.log_metrics({
                            "accuracy": accuracy,
                            "f1_macro": report.test_metrics.f1_macro,
                            "train_accuracy": report.train_metrics.accuracy,
                            "overfit_score": report.overfit_score,
                        })
                        self.tracker.log_params({
                            "tickers": ",".join(tickers),
                            "period": period,
                            "train_samples": split.train_size,
                            "test_samples": split.test_size,
                        })

                        # Save model with metrics and feature config
                        from pathlib import Path
                        models_dir = Path("models")
                        models_dir.mkdir(exist_ok=True)

                        model_filename = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("_gpu", "").replace("_cuda", "")
                        model_metrics = {
                            "accuracy": report.test_metrics.accuracy,
                            "f1_macro": report.test_metrics.f1_macro,
                            "training_end_date": split.train_end_date.isoformat(),
                            "test_end_date": split.test_end_date.isoformat(),
                        }
                        model.save(models_dir / f"{model_filename}.joblib",
                                   metrics=model_metrics, feature_config=feature_config)

                        dpg.set_value("train_progress", progress_base + 0.15)

                        # Show overfit warning if detected
                        if report.is_overfit:
                            dpg.set_value("train_status",
                                f"{model_name}: OVERFIT (train F1: {report.train_metrics.f1_macro:.0%} vs test: {report.test_metrics.f1_macro:.0%})")

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

                # Fetch data using DataPipeline
                dpg.set_value("train_status", "Fetching data for tuning...")
                pipeline = DataPipeline()

                try:
                    combined = pipeline.prepare_multiple(tickers, period=period)
                    X, y = pipeline.get_features_and_target(combined)
                except Exception as e:
                    dpg.set_value("train_status", f"Data error: {str(e)[:60]}")
                    return

                dpg.set_value("train_samples", str(len(X)))

                # Run tuning
                tuner = HyperparameterTuner()
                best_params = None
                best_score = 0

                if model_type in ["Random Forest", "All Models"]:
                    dpg.set_value("train_status", f"Tuning Random Forest ({n_trials} trials)...")
                    dpg.set_value("curr_model", "Random Forest")
                    result = tuner.tune_random_forest(X, y, n_trials=n_trials)
                    best_params = result.get("best_params", {})
                    best_score = result.get("best_value", 0)
                    dpg.set_value("train_progress", 0.33 if model_type == "All Models" else 1.0)

                if model_type in ["XGBoost (GPU)", "All Models"]:
                    dpg.set_value("train_status", f"Tuning XGBoost ({n_trials} trials)...")
                    dpg.set_value("curr_model", "XGBoost")
                    result = tuner.tune_xgboost(X, y, n_trials=n_trials, use_gpu=use_gpu)
                    score = result.get("best_value", 0)
                    if score > best_score:
                        best_params = result.get("best_params", {})
                        best_score = score
                    dpg.set_value("train_progress", 0.66 if model_type == "All Models" else 1.0)

                if model_type in ["LSTM (CUDA)", "All Models"]:
                    dpg.set_value("train_status", f"Tuning LSTM ({n_trials} trials)...")
                    dpg.set_value("curr_model", "LSTM")
                    result = tuner.tune_lstm(X, y, n_trials=min(n_trials, 20))
                    score = result.get("best_value", 0)
                    if score > best_score:
                        best_params = result.get("best_params", {})
                        best_score = score
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

    def _on_model_type_change(self, sender=None, data=None):
        """Handle model type selection change."""
        self._update_hyperparameter_ui()

    def _update_hyperparameter_ui(self):
        """Update hyperparameter controls based on selected model."""
        dpg.delete_item("hyperparam_container", children_only=True)

        model_type = dpg.get_value("train_model_type") if dpg.does_item_exist("train_model_type") else "Random Forest"

        with dpg.group(parent="hyperparam_container"):
            if model_type == "Random Forest":
                dpg.add_text("n_estimators", color=COLORS["text_muted"])
                dpg.add_slider_int(tag="hp_n_estimators", default_value=100, min_value=10, max_value=500, width=-1)
                dpg.add_spacer(height=5)
                dpg.add_text("max_depth", color=COLORS["text_muted"])
                dpg.add_slider_int(tag="hp_max_depth", default_value=10, min_value=3, max_value=30, width=-1)

            elif model_type == "XGBoost (GPU)":
                dpg.add_text("n_estimators", color=COLORS["text_muted"])
                dpg.add_slider_int(tag="hp_n_estimators", default_value=100, min_value=10, max_value=500, width=-1)
                dpg.add_spacer(height=5)
                dpg.add_text("learning_rate", color=COLORS["text_muted"])
                dpg.add_slider_float(tag="hp_learning_rate", default_value=0.1, min_value=0.01, max_value=0.5, format="%.3f", width=-1)

            elif model_type == "LSTM (CUDA)":
                dpg.add_text("hidden_size", color=COLORS["text_muted"])
                dpg.add_slider_int(tag="hp_hidden_size", default_value=64, min_value=16, max_value=256, width=-1)
                dpg.add_spacer(height=5)
                dpg.add_text("num_layers", color=COLORS["text_muted"])
                dpg.add_slider_int(tag="hp_num_layers", default_value=2, min_value=1, max_value=4, width=-1)

            else:  # All Models
                dpg.add_text("Tuning will optimize each model", color=COLORS["text_muted"])
                dpg.add_text("with its own parameter ranges.", color=COLORS["text_muted"])
