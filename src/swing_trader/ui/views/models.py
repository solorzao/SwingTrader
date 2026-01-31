import dearpygui.dearpygui as dpg
from ..theme import COLORS
from ...training.tracker import ExperimentTracker
from pathlib import Path


class ModelsView:
    """Model registry and management view."""

    def __init__(self, parent: str | int):
        self.parent = parent
        self.tracker = ExperimentTracker()
        self._runs_container = None
        self._setup()
        self._refresh_runs()

    def _setup(self):
        """Build the models view UI."""
        with dpg.group(parent=self.parent):
            dpg.add_text("MODEL REGISTRY", color=COLORS["accent"])
            dpg.add_text("Manage trained models with MLflow", color=COLORS["text_muted"])
            dpg.add_spacer(height=20)

            with dpg.group(horizontal=True):
                self._create_model_card("Random Forest", "rf")
                dpg.add_spacer(width=15)
                self._create_model_card("XGBoost (GPU)", "xgb")
                dpg.add_spacer(width=15)
                self._create_model_card("LSTM (CUDA)", "lstm")

            dpg.add_spacer(height=30)
            dpg.add_separator()
            dpg.add_spacer(height=20)

            dpg.add_text("MLFLOW RUNS", color=COLORS["accent"])
            dpg.add_spacer(height=10)

            with dpg.group(horizontal=True):
                dpg.add_button(label="Refresh Runs", callback=self._refresh_runs)
                dpg.add_spacer(width=10)
                dpg.add_button(label="Open MLflow UI", callback=self._open_mlflow_ui)

            dpg.add_spacer(height=10)

            # Runs table container
            self._runs_container = dpg.add_child_window(height=300, border=True)
            self._show_no_runs_message()

    def _create_model_card(self, name: str, model_id: str):
        """Create a model info card."""
        with dpg.child_window(width=280, height=220, border=True):
            dpg.add_text(name, color=COLORS["accent"])
            dpg.add_spacer(height=15)

            dpg.add_text("Status", color=COLORS["text_muted"])
            dpg.add_text("Not trained", tag=f"{model_id}_status", color=COLORS["sell"])

            dpg.add_spacer(height=10)
            dpg.add_text("Version", color=COLORS["text_muted"])
            dpg.add_text("--", tag=f"{model_id}_version")

            dpg.add_spacer(height=10)
            dpg.add_text("Accuracy", color=COLORS["text_muted"])
            dpg.add_text("--", tag=f"{model_id}_accuracy")

            dpg.add_spacer(height=10)
            dpg.add_text("Last Trained", color=COLORS["text_muted"])
            dpg.add_text("--", tag=f"{model_id}_trained")

            dpg.add_spacer(height=15)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Load", width=80, callback=lambda s, a, u: self._load_model(u), user_data=model_id)
                dpg.add_spacer(width=10)
                dpg.add_button(label="Details", width=80, callback=lambda s, a, u: self._show_details(u), user_data=model_id)

    def _show_no_runs_message(self):
        """Show message when no runs exist."""
        dpg.delete_item(self._runs_container, children_only=True)
        with dpg.group(parent=self._runs_container):
            dpg.add_text("Run History", color=COLORS["text_secondary"])
            dpg.add_spacer(height=10)
            dpg.add_text("No runs recorded yet. Train a model to see runs here.", color=COLORS["text_muted"])

    def _refresh_runs(self):
        """Refresh MLflow runs list."""
        try:
            runs = self.tracker.get_run_history(max_results=20)

            dpg.delete_item(self._runs_container, children_only=True)

            if not runs:
                self._show_no_runs_message()
                return

            with dpg.group(parent=self._runs_container):
                dpg.add_text(f"Recent Runs ({len(runs)})", color=COLORS["text_secondary"])
                dpg.add_spacer(height=10)

                # Create table header
                with dpg.table(header_row=True, borders_innerH=True, borders_outerH=True,
                              borders_innerV=True, borders_outerV=True, row_background=True):
                    dpg.add_table_column(label="Run Name", width_fixed=True, init_width_or_weight=150)
                    dpg.add_table_column(label="Model", width_fixed=True, init_width_or_weight=100)
                    dpg.add_table_column(label="Accuracy", width_fixed=True, init_width_or_weight=80)
                    dpg.add_table_column(label="Status", width_fixed=True, init_width_or_weight=80)
                    dpg.add_table_column(label="Start Time", width_fixed=True, init_width_or_weight=150)

                    for run in runs:
                        with dpg.table_row():
                            run_name = run.get("tags.mlflow.runName", "Unnamed")
                            dpg.add_text(run_name[:20] if run_name else "--")

                            model_type = run.get("tags.model_type", "--")
                            dpg.add_text(model_type)

                            accuracy = run.get("metrics.accuracy")
                            if accuracy is not None:
                                acc_color = COLORS["buy"] if accuracy > 0.6 else COLORS["text_primary"]
                                dpg.add_text(f"{accuracy:.2%}", color=acc_color)
                            else:
                                dpg.add_text("--")

                            status = run.get("status", "UNKNOWN")
                            status_color = COLORS["buy"] if status == "FINISHED" else COLORS["warning"]
                            dpg.add_text(status, color=status_color)

                            start_time = run.get("start_time")
                            if start_time:
                                from datetime import datetime
                                dt = datetime.fromtimestamp(start_time / 1000)
                                dpg.add_text(dt.strftime("%Y-%m-%d %H:%M"))
                            else:
                                dpg.add_text("--")

            # Update model cards with best runs
            self._update_model_cards(runs)

        except Exception as e:
            dpg.delete_item(self._runs_container, children_only=True)
            with dpg.group(parent=self._runs_container):
                dpg.add_text("Error loading runs", color=COLORS["sell"])
                dpg.add_text(str(e), color=COLORS["text_muted"])

    def _update_model_cards(self, runs: list):
        """Update model cards with latest run info."""
        model_mapping = {
            "random_forest": "rf",
            "rf": "rf",
            "xgboost": "xgb",
            "xgb": "xgb",
            "lstm": "lstm"
        }

        # Group runs by model type and find best for each
        model_runs = {}
        for run in runs:
            model_type = run.get("tags.model_type", "").lower()
            card_id = model_mapping.get(model_type)
            if card_id and card_id not in model_runs:
                model_runs[card_id] = run

        for card_id, run in model_runs.items():
            try:
                # Update status
                if dpg.does_item_exist(f"{card_id}_status"):
                    status = run.get("status", "UNKNOWN")
                    if status == "FINISHED":
                        dpg.set_value(f"{card_id}_status", "Trained")
                        dpg.configure_item(f"{card_id}_status", color=COLORS["buy"])
                    else:
                        dpg.set_value(f"{card_id}_status", status)

                # Update accuracy
                if dpg.does_item_exist(f"{card_id}_accuracy"):
                    accuracy = run.get("metrics.accuracy")
                    if accuracy is not None:
                        dpg.set_value(f"{card_id}_accuracy", f"{accuracy:.2%}")

                # Update last trained
                if dpg.does_item_exist(f"{card_id}_trained"):
                    start_time = run.get("start_time")
                    if start_time:
                        from datetime import datetime
                        dt = datetime.fromtimestamp(start_time / 1000)
                        dpg.set_value(f"{card_id}_trained", dt.strftime("%Y-%m-%d"))

                # Update version (run count as proxy)
                if dpg.does_item_exist(f"{card_id}_version"):
                    run_id = run.get("run_id", "")[:8]
                    dpg.set_value(f"{card_id}_version", run_id if run_id else "--")

            except Exception:
                pass  # Skip if card elements don't exist

    def _load_model(self, model_id: str):
        """Load a model from registry."""
        model_names = {"rf": "random_forest", "xgb": "xgboost", "lstm": "lstm"}
        model_name = model_names.get(model_id, model_id)

        try:
            model = self.tracker.load_model(model_name)
            self._show_popup("Success", f"Model '{model_name}' loaded successfully!")
        except Exception as e:
            self._show_popup("Error", f"Failed to load model: {e}")

    def _show_details(self, model_id: str):
        """Show model details in a popup."""
        model_names = {"rf": "Random Forest", "xgb": "XGBoost (GPU)", "lstm": "LSTM (CUDA)"}
        model_name = model_names.get(model_id, model_id)

        try:
            best_run = self.tracker.get_best_run(metric="accuracy", maximize=True)
            if best_run:
                details = f"Best Run ID: {best_run.get('run_id', 'N/A')[:8]}\n"
                details += f"Accuracy: {best_run.get('metrics.accuracy', 'N/A')}\n"
                details += f"Status: {best_run.get('status', 'N/A')}"
                self._show_popup(f"{model_name} Details", details)
            else:
                self._show_popup(f"{model_name} Details", "No training runs found for this model.")
        except Exception as e:
            self._show_popup("Error", f"Failed to get details: {e}")

    def _show_popup(self, title: str, message: str):
        """Show a popup dialog."""
        popup_id = dpg.generate_uuid()

        with dpg.window(label=title, modal=True, tag=popup_id, width=400, height=150,
                       pos=[dpg.get_viewport_width() // 2 - 200, dpg.get_viewport_height() // 2 - 75]):
            dpg.add_text(message, wrap=380)
            dpg.add_spacer(height=20)
            dpg.add_button(label="OK", width=100, callback=lambda: dpg.delete_item(popup_id))

    def _open_mlflow_ui(self):
        """Open MLflow UI in browser."""
        import subprocess
        import webbrowser
        import threading

        def start_mlflow():
            try:
                # Start MLflow UI server
                subprocess.Popen(
                    ["mlflow", "ui", "--port", "5000"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
                )
                import time
                time.sleep(2)  # Wait for server to start
                webbrowser.open("http://localhost:5000")
            except Exception as e:
                print(f"Failed to start MLflow UI: {e}")

        thread = threading.Thread(target=start_mlflow, daemon=True)
        thread.start()
        self._show_popup("MLflow UI", "Starting MLflow UI at http://localhost:5000\n\nOpening in browser...")
