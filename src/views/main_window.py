#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main Window module.
This module contains the MainWindow class which serves as the main UI for the Stereo Image Player.
"""

import os
import logging

from PySide6.QtCore import Qt, Slot, Signal
from PySide6.QtGui import QAction, QIcon, QPixmap, QBrush, QPalette
from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QFileDialog, QMenuBar, 
    QMenu, QStatusBar, QMessageBox, QProgressDialog, QTabWidget
)

from src.utils.ui_constants import WindowSize, Messages, Layout, FileDialog, Icons
from src.utils.constants import UI_COLORS
from src.views.image_view import ImageView
from src.views.setting_view import SettingView
from src.views.calibration_view import CalibrationView
from src.views.project_info_tab import ProjectInfoTab
from src.controllers.calibration_controller import CalibrationController
from src.models.calibration_model import CalibrationModel
from src.views.widgets.inout_indicator import InOutLED


class MainWindow(QMainWindow):
    """
    Main window for the Stereo Image Player application.
    """
    
    # Signal for app closing
    app_closing = Signal()
    
    def __init__(self, parent=None):
        """
        Initialize the main window.
        
        Args:
            parent (QWidget, optional): Parent widget
        """
        super(MainWindow, self).__init__(parent)
        
        # Set window properties
        self.setWindowTitle("Stereo Image Player")
        self.resize(WindowSize.DEFAULT_WIDTH, WindowSize.DEFAULT_HEIGHT)
        self.setMinimumSize(WindowSize.MIN_WIDTH, WindowSize.MIN_HEIGHT)
        
        # Create IN/OUT LED indicator
        self.inout_led = InOutLED()
        
        # Set background image
        self._set_background_image()
        
        # Set up UI
        self._setup_ui()
        
        # Show ready message
        self.status_bar.showMessage(Messages.READY)
    
    def _set_background_image(self):
        """Set the background image for the main window."""
        try:
            # Try multiple possible locations for the background image
            possible_paths = [
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                            "src", "resources", "images", "background.png"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                            "src", "resources", "background.png"),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                            "resources", "images", "background.png"),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                            "resources", "background.png")
            ]
            
            # Find the first existing path
            background_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    background_path = path
                    break
            
            if background_path:
                pixmap = QPixmap(background_path)
                if not pixmap.isNull():
                    # Create semi-transparent background effect
                    palette = self.palette()
                    brush = QBrush(pixmap)
                    palette.setBrush(QPalette.Window, brush)
                    self.setPalette(palette)
                    self.setAutoFillBackground(True)
                    logging.info(f"Background image loaded from {background_path}")
                else:
                    logging.warning("Could not load background image: pixmap is null")
            else:
                logging.warning("Background image not found in any of the expected locations")
        except Exception as e:
            logging.error(f"Error setting background image: {e}")
    
    def _setup_ui(self):
        """Set up the user interface."""
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(Layout.MARGIN, Layout.MARGIN, Layout.MARGIN, Layout.MARGIN)
        main_layout.setSpacing(Layout.SPACING)
        
        # Create tab widget with enhanced styling
        self._setup_tab_widget()
        
        # Create calibration model and controller
        self.calibration_model = CalibrationModel()
        self.calibration_controller = CalibrationController(
            self.calibration_model,
            self.calibration_view
        )
        
        # Connect signals from settings view
        self.setting_view.settings_changed.connect(self._on_settings_changed)
        
        # Set up menu bar
        self._setup_menu_bar()
        
        # Set up status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
    
    def _setup_tab_widget(self):
        """Set up the tab widget for different views."""
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{ 
                border: 1px solid #444; 
                border-radius: 8px;
                padding: 25px; 
                background-color: rgba(30, 30, 45, 210);
            }}
            QTabBar::tab {{ 
                padding: 12px 28px;
                margin: 2px 6px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                background-color: {UI_COLORS.TAB_BG};
                color: {UI_COLORS.TAB_TEXT};
                font-family: 'Segoe UI';
                font-size: 13px;
                font-weight: 500;
            }}
            QTabBar::tab:selected, QTabBar::tab:hover {{
                background-color: {UI_COLORS.TAB_BG_HOVER};
                color: {UI_COLORS.TAB_TEXT_SELECTED};
                border-bottom: 3px solid {UI_COLORS.ACCENT_PRIMARY};
            }}
        """)
        
        # Create and add tabs
        self.image_view = ImageView()
        self.setting_view = SettingView()
        self.calibration_view = CalibrationView()
        self.project_info_tab = ProjectInfoTab()
        
        # Add tabs to the tab widget
        self.tab_widget.addTab(self.image_view, "Image View")
        self.tab_widget.addTab(self.setting_view, "Settings")
        self.tab_widget.addTab(self.calibration_view, "Calibration")
        self.tab_widget.addTab(self.project_info_tab, "Project Information")
        
        # Add tab widget to main layout
        self.centralWidget().layout().addWidget(self.tab_widget)
    
    def _setup_menu_bar(self):
        """Set up the menu bar."""
        menu_bar = QMenuBar()
        self.setMenuBar(menu_bar)
        
        # File menu
        file_menu = QMenu("&File", self)
        menu_bar.addMenu(file_menu)
        
        # Open action
        open_action = QAction(QIcon(Icons.OPEN), "&Open Folder...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.setStatusTip("Open a folder containing stereo images")
        open_action.triggered.connect(self._on_open_folder)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction(QIcon(Icons.EXIT), "E&xit", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = QMenu("&View", self)
        menu_bar.addMenu(view_menu)
        
        # Help menu
        help_menu = QMenu("&Help", self)
        menu_bar.addMenu(help_menu)
        
        # About action
        about_action = QAction("&About", self)
        about_action.setStatusTip("Show information about the application")
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)
    
    def _on_open_folder(self):
        """Handle open folder action."""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            FileDialog.DIALOG_CAPTION,
            os.path.expanduser("~"),
            QFileDialog.ShowDirsOnly
        )
        
        if folder_path:
            self.open_folder(folder_path)
    
    def _on_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Stereo Image Player",
            "<h3>Stereo Image Player</h3>"
            "<p>Version 1.0</p>"
            "<p>A simple player for stereo image pairs.</p>"
            "<p>Developed with PySide6 and OpenCV.</p>"
        )
    
    def _on_settings_changed(self, settings):
        """
        Handle settings changes from the settings view.
        
        Args:
            settings (dict): New settings
        """
        # Apply settings to the image view
        # This will be implemented by the controller
        pass
    
    @Slot(str)
    def show_error_message(self, message):
        """
        Show an error message dialog.
        
        Args:
            message (str): Error message
        """
        QMessageBox.critical(self, "Error", message)
    
    @Slot(str)
    def show_warning_message(self, message):
        """
        Show a warning message dialog.
        
        Args:
            message (str): Warning message
        """
        QMessageBox.warning(self, "Warning", message)
    
    @Slot(str)
    def show_info_message(self, message):
        """
        Show an information message dialog.
        
        Args:
            message (str): Information message
        """
        QMessageBox.information(self, "Information", message)
    
    @Slot(str)
    def update_status(self, message):
        """
        Update the status bar message.
        
        Args:
            message (str): Status message
        """
        self.status_bar.showMessage(message)
    
    def create_progress_dialog(self, title, message, minimum=0, maximum=100):
        """
        Create and return a progress dialog.
        
        Args:
            title (str): Dialog title
            message (str): Dialog message
            minimum (int): Minimum progress value
            maximum (int): Maximum progress value
            
        Returns:
            QProgressDialog: Progress dialog
        """
        progress_dialog = QProgressDialog(message, "Cancel", minimum, maximum, self)
        progress_dialog.setWindowTitle(title)
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)  # Show immediately
        progress_dialog.setAutoClose(True)
        progress_dialog.setAutoReset(True)
        
        return progress_dialog
    
    def closeEvent(self, event):
        """
        Handle window close event.
        Emit signal for saving tracking data before closing.
        
        Args:
            event (QCloseEvent): Close event
        """
        # Emit app closing signal
        self.app_closing.emit()
        
        # Accept the close event
        event.accept()
    
    def open_folder(self, folder_path):
        """
        Open a folder containing stereo images.
        This method should be implemented by the controller.
        
        Args:
            folder_path (str): Path to the folder
        """
        # This will be overridden by the controller
        pass

    def update_calibration_images(self, left_image, right_image):
        """
        Update the images in the calibration view.
        
        Args:
            left_image: QPixmap or QImage for the left view
            right_image: QPixmap or QImage for the right view
        """
        if hasattr(self, 'calibration_controller'):
            self.calibration_controller.set_images(left_image, right_image)

    def connect_game_analyzer(self, analyzer):
        """
        Connect to a game analyzer to receive bounce events.
        
        Args:
            analyzer: GameAnalyzer instance
        """
        if analyzer:
            # Connect in/out LED to game analyzer
            analyzer.in_out_detected.connect(self.inout_led.on_in_out)
            logging.info("IN/OUT LED connected to game analyzer")
            
            # Add LED to status bar
            self.status_bar.addPermanentWidget(self.inout_led) 