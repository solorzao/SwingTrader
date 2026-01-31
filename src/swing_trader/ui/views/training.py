import dearpygui.dearpygui as dpg
from ..theme import COLORS

class TrainingView:
    """Training and hyperparameter tuning view."""

    def __init__(self, parent: str | int):
        self.parent = parent
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
                default_value="All Models",
                width=-1
            )

            dpg.add_spacer(height=10)
            dpg.add_text("Training Symbols", color=COLORS["text_muted"])
            dpg.add_input_text(
                tag="train_tickers",
                hint="AAPL, MSFT, GOOGL",
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
                width=-1,
                callback=self._on_tune
            )

            dpg.add_spacer(height=25)
            dpg.add_separator()
            dpg.add_spacer(height=15)

            # Status panel
            dpg.add_text("STATUS", color=COLORS["accent"])
            dpg.add_spacer(height=10)
            dpg.add_text("Ready", tag="train_status", color=COLORS["text_muted"])
            dpg.add_progress_bar(tag="train_progress", default_value=0, width=-1)

    def _create_metrics(self):
        """Create metrics visualization panel."""
        with dpg.child_window(width=-1, height=-1, border=True):
            dpg.add_text("TRAINING METRICS", color=COLORS["text_secondary"])
            dpg.add_spacer(height=10)

            with dpg.plot(tag="loss_chart", label="Loss Curve", height=220, width=-1):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="Epoch", tag="loss_x")
                dpg.add_plot_axis(dpg.mvYAxis, label="Loss", tag="loss_y")

            dpg.add_spacer(height=15)

            with dpg.plot(tag="acc_chart", label="Accuracy", height=220, width=-1):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="Epoch", tag="acc_x")
                dpg.add_plot_axis(dpg.mvYAxis, label="Accuracy", tag="acc_y")

            dpg.add_spacer(height=20)

            # Stats row
            with dpg.group(horizontal=True):
                with dpg.child_window(width=180, height=80, border=True):
                    dpg.add_text("GPU UTILIZATION", color=COLORS["text_secondary"])
                    dpg.add_text("--", tag="gpu_util", color=COLORS["accent"])
                dpg.add_spacer(width=10)
                with dpg.child_window(width=180, height=80, border=True):
                    dpg.add_text("BEST ACCURACY", color=COLORS["text_secondary"])
                    dpg.add_text("--", tag="best_acc", color=COLORS["buy"])
                dpg.add_spacer(width=10)
                with dpg.child_window(width=180, height=80, border=True):
                    dpg.add_text("CURRENT EPOCH", color=COLORS["text_secondary"])
                    dpg.add_text("--", tag="curr_epoch", color=COLORS["interactive"])

    def _on_train(self, sender=None, data=None):
        """Handle training start."""
        dpg.set_value("train_status", "Training...")
        print("Starting training...")

    def _on_tune(self, sender=None, data=None):
        """Handle tuning start."""
        dpg.set_value("train_status", "Tuning hyperparameters...")
        print("Starting Optuna tuning...")
