import dearpygui.dearpygui as dpg
from pathlib import Path

from .theme import setup_theme, COLORS
from .views.signals import SignalsView
from .views.training import TrainingView
from .views.backtest import BacktestView
from .views.models import ModelsView


class SwingTraderApp:
    """Main Dear PyGui application for swing trading."""

    def __init__(self):
        self.width = 1600
        self.height = 950
        self.title = "Swing Trader"

    def setup(self):
        """Initialize Dear PyGui context and viewport."""
        dpg.create_context()

        # Create viewport
        dpg.create_viewport(
            title=self.title,
            width=self.width,
            height=self.height,
            min_width=1200,
            min_height=700
        )

        # Apply custom theme
        setup_theme()

        # Setup DearPyGui
        dpg.setup_dearpygui()

        # Create main window
        self._create_main_window()

    def _create_main_window(self):
        """Create the main application window with navigation."""
        with dpg.window(tag="main_window", label="", no_title_bar=True):
            # Header bar
            with dpg.group(horizontal=True):
                dpg.add_text("SWING TRADER", color=COLORS["accent"])
                dpg.add_spacer(width=20)
                dpg.add_text("|", color=COLORS["border"])
                dpg.add_spacer(width=20)
                dpg.add_text("ML Trading Signals", color=COLORS["text_secondary"])

            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=10)

            # Main tab bar with views
            with dpg.tab_bar(tag="main_tabs"):
                # Signals Tab
                with dpg.tab(label="  Signals  ", tag="tab_signals"):
                    dpg.add_spacer(height=20)
                    SignalsView("tab_signals")

                # Training Tab
                with dpg.tab(label="  Training  ", tag="tab_training"):
                    dpg.add_spacer(height=20)
                    TrainingView("tab_training")

                # Backtest Tab
                with dpg.tab(label="  Backtest  ", tag="tab_backtest"):
                    dpg.add_spacer(height=20)
                    BacktestView("tab_backtest")

                # Models Tab
                with dpg.tab(label="  Models  ", tag="tab_models"):
                    dpg.add_spacer(height=20)
                    ModelsView("tab_models")

        dpg.set_primary_window("main_window", True)

    def run(self):
        """Start the application."""
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()


def main():
    """Entry point for the desktop application."""
    app = SwingTraderApp()
    app.setup()
    app.run()


if __name__ == "__main__":
    main()
