import dearpygui.dearpygui as dpg

# Color palette - Dark trading terminal aesthetic
COLORS = {
    # Backgrounds
    "bg_dark": (13, 17, 23),           # #0D1117
    "bg_medium": (22, 27, 34),         # #161B22
    "bg_light": (33, 38, 45),          # #21262D

    # Text
    "text_primary": (230, 237, 243),   # #E6EDF3
    "text_secondary": (139, 148, 158), # #8B949E
    "text_muted": (110, 118, 129),     # #6E7681

    # Accents
    "buy": (0, 212, 170),              # #00D4AA - cyan/green
    "sell": (255, 107, 107),           # #FF6B6B - coral red
    "hold": (139, 148, 158),           # #8B949E - gray
    "accent": (255, 217, 61),          # #FFD93D - gold
    "interactive": (88, 166, 255),     # #58A6FF - blue
    "warning": (255, 193, 7),          # #FFC107 - amber/yellow

    # UI elements
    "border": (48, 54, 61),            # #30363D
    "button": (35, 134, 54),           # #238636 - green button
    "button_hover": (46, 160, 67),     # #2EA043
}

def setup_theme():
    """Create and apply the trading terminal theme."""
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            # Window backgrounds
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, COLORS["bg_dark"])
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, COLORS["bg_medium"])
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, COLORS["bg_medium"])

            # Frame/input backgrounds
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, COLORS["bg_light"])
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, COLORS["border"])
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, COLORS["border"])

            # Text colors
            dpg.add_theme_color(dpg.mvThemeCol_Text, COLORS["text_primary"])
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, COLORS["text_muted"])

            # Buttons
            dpg.add_theme_color(dpg.mvThemeCol_Button, COLORS["button"])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, COLORS["button_hover"])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, COLORS["interactive"])

            # Headers/tabs
            dpg.add_theme_color(dpg.mvThemeCol_Header, COLORS["bg_light"])
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, COLORS["border"])
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, COLORS["interactive"])
            dpg.add_theme_color(dpg.mvThemeCol_Tab, COLORS["bg_medium"])
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, COLORS["bg_light"])
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, COLORS["bg_light"])
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocused, COLORS["bg_dark"])
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocusedActive, COLORS["bg_medium"])

            # Borders
            dpg.add_theme_color(dpg.mvThemeCol_Border, COLORS["border"])
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, (0, 0, 0, 0))

            # Title bar
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, COLORS["bg_dark"])
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, COLORS["bg_medium"])

            # Scrollbar
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, COLORS["bg_dark"])
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, COLORS["border"])
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, COLORS["text_muted"])
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, COLORS["text_secondary"])

            # Checkmarks and sliders
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, COLORS["buy"])
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, COLORS["interactive"])
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, COLORS["accent"])

            # Plot colors
            dpg.add_theme_color(dpg.mvPlotCol_PlotBg, COLORS["bg_dark"])
            dpg.add_theme_color(dpg.mvPlotCol_PlotBorder, COLORS["border"])
            dpg.add_theme_color(dpg.mvPlotCol_FrameBg, COLORS["bg_medium"])

            # Styling
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 6)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 6)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 12, 12)

    dpg.bind_theme(global_theme)
    return global_theme


def create_signal_theme(signal_type: str):
    """Create theme for signal-specific styling."""
    color = COLORS.get(signal_type, COLORS["text_primary"])

    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_Text, color)

    return theme
