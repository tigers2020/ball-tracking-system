#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
UI Theme for the Stereo Image Player application.
This file contains all the theme-related styles and colors.
"""

from PySide6.QtGui import QColor, QPalette, QFont
from PySide6.QtWidgets import QApplication


class Colors:
    """Color definitions for the application theme"""
    PRIMARY = QColor(53, 53, 53)
    SECONDARY = QColor(35, 35, 35)
    ACCENT = QColor(42, 130, 218)
    TEXT = QColor(255, 255, 255)
    TEXT_SECONDARY = QColor(200, 200, 200)
    WARNING = QColor(240, 160, 0)
    ERROR = QColor(240, 0, 0)
    SUCCESS = QColor(0, 180, 0)
    BACKGROUND = QColor(45, 45, 45)
    BORDER = QColor(80, 80, 80)
    BUTTON_BACKGROUND = QColor(70, 70, 70)
    BUTTON_HOVER = QColor(90, 90, 90)
    BUTTON_PRESSED = QColor(30, 30, 30)
    SLIDER_HANDLE = QColor(60, 130, 200)
    SLIDER_GROOVE = QColor(60, 60, 60)


class Fonts:
    """Font definitions for the application theme"""
    BASE_FONT_FAMILY = "Segoe UI"
    BASE_FONT_SIZE = 10
    TITLE_FONT_SIZE = 14
    SUBTITLE_FONT_SIZE = 12
    SMALL_FONT_SIZE = 9

    @staticmethod
    def get_base_font():
        """Returns the base font for the application"""
        font = QFont(Fonts.BASE_FONT_FAMILY, Fonts.BASE_FONT_SIZE)
        return font

    @staticmethod
    def get_title_font():
        """Returns the title font for the application"""
        font = QFont(Fonts.BASE_FONT_FAMILY, Fonts.TITLE_FONT_SIZE)
        font.setBold(True)
        return font

    @staticmethod
    def get_subtitle_font():
        """Returns the subtitle font for the application"""
        font = QFont(Fonts.BASE_FONT_FAMILY, Fonts.SUBTITLE_FONT_SIZE)
        font.setBold(True)
        return font

    @staticmethod
    def get_small_font():
        """Returns the small font for the application"""
        font = QFont(Fonts.BASE_FONT_FAMILY, Fonts.SMALL_FONT_SIZE)
        return font


class StyleManager:
    """Manages styles for different widgets"""

    @staticmethod
    def get_main_window_style():
        """Returns the style for the main window"""
        return f"""
        QMainWindow {{
            background-color: {Colors.BACKGROUND.name()};
            color: {Colors.TEXT.name()};
        }}
        """

    @staticmethod
    def get_button_style():
        """Returns the style for buttons"""
        return f"""
        QPushButton {{
            background-color: {Colors.BUTTON_BACKGROUND.name()};
            color: {Colors.TEXT.name()};
            border: 1px solid {Colors.BORDER.name()};
            border-radius: 3px;
            padding: 5px 10px;
        }}
        QPushButton:hover {{
            background-color: {Colors.BUTTON_HOVER.name()};
        }}
        QPushButton:pressed {{
            background-color: {Colors.BUTTON_PRESSED.name()};
        }}
        QPushButton:disabled {{
            background-color: {Colors.SECONDARY.name()};
            color: {Colors.TEXT_SECONDARY.name()};
        }}
        """

    @staticmethod
    def get_calibration_button_style():
        """Returns the style for calibration buttons"""
        return f"""
        QPushButton {{
            background-color: {Colors.BUTTON_BACKGROUND.name()};
            color: {Colors.TEXT.name()};
            border: 1px solid {Colors.BORDER.name()};
            border-radius: 3px;
            padding: 5px 10px;
            min-width: 80px;
        }}
        QPushButton:hover {{
            background-color: {Colors.BUTTON_HOVER.name()};
        }}
        QPushButton:pressed {{
            background-color: {Colors.BUTTON_PRESSED.name()};
        }}
        QPushButton[class="primary"] {{
            background-color: {Colors.ACCENT.name()};
        }}
        QPushButton[class="primary"]:hover {{
            background-color: QColor({Colors.ACCENT.red()}, {Colors.ACCENT.green()}, {Colors.ACCENT.blue()}, 220);
        }}
        QPushButton:disabled {{
            background-color: {Colors.SECONDARY.name()};
            color: {Colors.TEXT_SECONDARY.name()};
        }}
        """

    @staticmethod
    def get_toolbar_style():
        """Returns the style for toolbars"""
        return f"""
        QToolBar {{
            background-color: {Colors.PRIMARY.name()};
            border-bottom: 1px solid {Colors.BORDER.name()};
            spacing: 5px;
        }}
        QToolBar QToolButton {{
            background-color: transparent;
            color: {Colors.TEXT.name()};
            border: none;
            padding: 5px;
        }}
        QToolBar QToolButton:hover {{
            background-color: {Colors.BUTTON_HOVER.name()};
            border-radius: 3px;
        }}
        QToolBar QToolButton:pressed {{
            background-color: {Colors.BUTTON_PRESSED.name()};
            border-radius: 3px;
        }}
        """

    @staticmethod
    def get_menu_style():
        """Returns the style for menus"""
        return f"""
        QMenuBar {{
            background-color: {Colors.PRIMARY.name()};
            color: {Colors.TEXT.name()};
        }}
        QMenuBar::item {{
            background: transparent;
            padding: 5px 10px;
        }}
        QMenuBar::item:selected {{
            background-color: {Colors.ACCENT.name()};
        }}
        QMenu {{
            background-color: {Colors.PRIMARY.name()};
            color: {Colors.TEXT.name()};
            border: 1px solid {Colors.BORDER.name()};
        }}
        QMenu::item {{
            padding: 5px 30px 5px 20px;
        }}
        QMenu::item:selected {{
            background-color: {Colors.ACCENT.name()};
        }}
        """

    @staticmethod
    def get_slider_style():
        """Returns the style for sliders"""
        return f"""
        QSlider::groove:horizontal {{
            border: 1px solid {Colors.BORDER.name()};
            height: 6px;
            background: {Colors.SLIDER_GROOVE.name()};
            margin: 0px;
            border-radius: 3px;
        }}
        QSlider::handle:horizontal {{
            background: {Colors.SLIDER_HANDLE.name()};
            border: 1px solid {Colors.BORDER.name()};
            width: 14px;
            margin: -4px 0;
            border-radius: 7px;
        }}
        QSlider::sub-page:horizontal {{
            background: {Colors.ACCENT.name()};
            border-radius: 3px;
        }}
        """

    @staticmethod
    def get_label_style():
        """Returns the style for labels"""
        return f"""
        QLabel {{
            color: {Colors.TEXT.name()};
        }}
        """

    @staticmethod
    def get_status_bar_style():
        """Returns the style for the status bar"""
        return f"""
        QStatusBar {{
            background-color: {Colors.PRIMARY.name()};
            color: {Colors.TEXT.name()};
            border-top: 1px solid {Colors.BORDER.name()};
        }}
        """

    @staticmethod
    def get_progress_dialog_style():
        """Returns the style for progress dialogs"""
        return f"""
        QProgressDialog {{
            background-color: {Colors.BACKGROUND.name()};
            color: {Colors.TEXT.name()};
            border: 1px solid {Colors.BORDER.name()};
        }}
        QProgressBar {{
            border: 1px solid {Colors.BORDER.name()};
            border-radius: 3px;
            text-align: center;
        }}
        QProgressBar::chunk {{
            background-color: {Colors.ACCENT.name()};
        }}
        """

    @staticmethod
    def get_graphics_view_style():
        """Returns the style for QGraphicsView"""
        return f"""
        QGraphicsView {{
            background-color: {Colors.SECONDARY.name()};
            border: 1px solid {Colors.BORDER.name()};
            border-radius: 3px;
        }}
        """

    @staticmethod
    def get_tab_widget_style():
        """Returns the style for QTabWidget"""
        return f"""
        QTabWidget::pane {{
            border: 1px solid {Colors.BORDER.name()};
            background: {Colors.BACKGROUND.name()};
        }}
        QTabWidget::tab-bar {{
            left: 5px;
        }}
        QTabBar::tab {{
            background: {Colors.PRIMARY.name()};
            color: {Colors.TEXT.name()};
            border: 1px solid {Colors.BORDER.name()};
            border-bottom-color: {Colors.BORDER.name()};
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            min-width: 8ex;
            padding: 6px 10px;
        }}
        QTabBar::tab:selected, QTabBar::tab:hover {{
            background: {Colors.ACCENT.name()};
        }}
        QTabBar::tab:selected {{
            border-color: {Colors.BORDER.name()};
            border-bottom-color: {Colors.ACCENT.name()};
        }}
        QTabBar::tab:!selected {{
            margin-top: 2px;
        }}
        """

    @staticmethod
    def get_group_box_style():
        """Returns the style for QGroupBox"""
        return f"""
        QGroupBox {{
            background-color: {Colors.PRIMARY.name()};
            border: 1px solid {Colors.BORDER.name()};
            border-radius: 5px;
            margin-top: 10px;
            font-weight: bold;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
            color: {Colors.TEXT.name()};
        }}
        """


class ThemeManager:
    """Manages the application theme"""

    @staticmethod
    def apply_theme(app):
        """
        Apply the theme to the entire application
        
        Args:
            app (QApplication): The application instance
        """
        # Set application font
        app.setFont(Fonts.get_base_font())
        
        # Set application style sheet
        style_sheet = "\n".join([
            StyleManager.get_main_window_style(),
            StyleManager.get_button_style(),
            StyleManager.get_calibration_button_style(),
            StyleManager.get_toolbar_style(),
            StyleManager.get_menu_style(),
            StyleManager.get_slider_style(),
            StyleManager.get_label_style(),
            StyleManager.get_status_bar_style(),
            StyleManager.get_progress_dialog_style(),
            StyleManager.get_graphics_view_style(),
            StyleManager.get_tab_widget_style(),
            StyleManager.get_group_box_style(),
        ])
        
        app.setStyleSheet(style_sheet)
        
        # Set application palette
        palette = QPalette()
        palette.setColor(QPalette.Window, Colors.BACKGROUND)
        palette.setColor(QPalette.WindowText, Colors.TEXT)
        palette.setColor(QPalette.Base, Colors.SECONDARY)
        palette.setColor(QPalette.AlternateBase, Colors.PRIMARY)
        palette.setColor(QPalette.ToolTipBase, Colors.PRIMARY)
        palette.setColor(QPalette.ToolTipText, Colors.TEXT)
        palette.setColor(QPalette.Text, Colors.TEXT)
        palette.setColor(QPalette.Button, Colors.BUTTON_BACKGROUND)
        palette.setColor(QPalette.ButtonText, Colors.TEXT)
        palette.setColor(QPalette.BrightText, Colors.TEXT)
        palette.setColor(QPalette.Link, Colors.ACCENT)
        palette.setColor(QPalette.Highlight, Colors.ACCENT)
        palette.setColor(QPalette.HighlightedText, Colors.TEXT)
        
        app.setPalette(palette) 