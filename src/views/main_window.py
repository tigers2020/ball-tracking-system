#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main Window module.
This module contains the MainWindow class which serves as the main UI for the Stereo Image Player.
"""

import os
import logging

from PySide6.QtCore import Qt, Slot, Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QFileDialog, QMenuBar, 
    QMenu, QStatusBar, QMessageBox, QProgressDialog, QTabWidget
)

from src.utils.ui_constants import WindowSize, Messages, Layout, FileDialog, Icons
from src.views.image_view import ImageView
from src.views.setting_view import SettingView
from src.views.calibration_tab import CourtCalibrationView


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
        
        # Set up UI
        self._setup_ui()
        
        # Show ready message
        self.status_bar.showMessage(Messages.READY)
    
    def _setup_ui(self):
        """Set up the user interface."""
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(Layout.MARGIN, Layout.MARGIN, Layout.MARGIN, Layout.MARGIN)
        main_layout.setSpacing(Layout.SPACING)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create image view tab
        self.image_view = ImageView()
        self.tab_widget.addTab(self.image_view, "Image View")
        
        # Create settings tab
        self.setting_view = SettingView()
        self.tab_widget.addTab(self.setting_view, "Settings")
        
        # Create court calibration tab
        self.calibration_view = CourtCalibrationView()
        self.tab_widget.addTab(self.calibration_view, "Court Calibration")
        
        # Connect signals from settings view
        self.setting_view.settings_changed.connect(self._on_settings_changed)
        
        # Set up menu bar
        self._setup_menu_bar()
        
        # Set up status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
    
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