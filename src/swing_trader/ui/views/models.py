import dearpygui.dearpygui as dpg
from ..theme import COLORS

class ModelsView:
    """Model registry and management view."""

    def __init__(self, parent: str | int):
        self.parent = parent
        self._setup()

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

            # Runs table placeholder
            with dpg.child_window(height=300, border=True):
                dpg.add_text("Run History", color=COLORS["text_secondary"])
                dpg.add_spacer(height=10)
                dpg.add_text("No runs recorded yet", color=COLORS["text_muted"])
                dpg.add_spacer(height=10)
                dpg.add_button(label="Refresh Runs", callback=self._refresh_runs)

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
                dpg.add_button(label="Load", width=80, callback=lambda: self._load_model(model_id))
                dpg.add_spacer(width=10)
                dpg.add_button(label="Details", width=80, callback=lambda: self._show_details(model_id))

    def _load_model(self, model_id: str):
        """Load a model from registry."""
        print(f"Loading model: {model_id}")

    def _show_details(self, model_id: str):
        """Show model details."""
        print(f"Showing details for: {model_id}")

    def _refresh_runs(self):
        """Refresh MLflow runs list."""
        print("Refreshing MLflow runs...")
