#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setting View module.
This module contains the SettingView class for the settings tab in the Stereo Image Player.
"""

import logging
import os
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QPushButton, QFileDialog, QLineEdit
)

from src.utils.ui_constants import Layout, FileDialog


class SettingView(QWidget):
    """
    Widget for the settings tab in the Stereo Image Player.
    Contains application settings.
    """
    
    # Signals
    settings_changed = Signal(dict)
    folder_selected = Signal(str)
    
    def __init__(self, parent=None):
        """
        Initialize the settings view.
        
        Args:
            parent (QWidget, optional): Parent widget
        """
        super(SettingView, self).__init__(parent)
        
        # Default settings
        self.settings = {
            "last_image_folder": ""
        }
        
        # Set up UI
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(Layout.MARGIN, Layout.MARGIN, Layout.MARGIN, Layout.MARGIN)
        main_layout.setSpacing(Layout.SPACING)
        
        # Folder settings group
        folder_group = QGroupBox("Image Folder Settings")
        folder_layout = QVBoxLayout()
        
        # Folder path layout
        folder_path_layout = QHBoxLayout()
        
        # Folder path label and line edit
        folder_path_label = QLabel("Last Image Folder:")
        self.folder_path_edit = QLineEdit()
        self.folder_path_edit.setReadOnly(True)
        self.folder_path_edit.setPlaceholderText("Select a folder...")
        
        # Browse button
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._on_browse_clicked)
        
        # Add widgets to layout
        folder_path_layout.addWidget(folder_path_label)
        folder_path_layout.addWidget(self.folder_path_edit, 1)  # Stretch factor 1
        folder_path_layout.addWidget(self.browse_button)
        
        folder_layout.addLayout(folder_path_layout)
        
        # Open button
        self.open_button = QPushButton("Open This Folder")
        self.open_button.clicked.connect(self._on_open_clicked)
        folder_layout.addWidget(self.open_button)
        
        folder_group.setLayout(folder_layout)
        main_layout.addWidget(folder_group)
        
        # Add stretch to push everything to the top
        main_layout.addStretch()
    
    def _on_browse_clicked(self):
        """Handle browse button click."""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            FileDialog.DIALOG_CAPTION,
            os.path.expanduser("~"),
            QFileDialog.ShowDirsOnly
        )
        
        if folder_path:
            self.set_folder_path(folder_path)
            self.settings["last_image_folder"] = folder_path
            self.settings_changed.emit(self.settings.copy())
    
    def _on_open_clicked(self):
        """Handle open button click."""
        folder_path = self.folder_path_edit.text()
        if folder_path and os.path.isdir(folder_path):
            self.folder_selected.emit(folder_path)
    
    def set_folder_path(self, folder_path):
        """
        Set the folder path in the UI.
        
        Args:
            folder_path (str): Folder path
        """
        self.folder_path_edit.setText(folder_path)
        self.settings["last_image_folder"] = folder_path
        
        # Enable/disable open button based on folder path
        self.open_button.setEnabled(bool(folder_path and os.path.isdir(folder_path)))
    
    def get_settings(self):
        """
        Get the current settings.
        
        Returns:
            dict: Current settings
        """
        return self.settings.copy() 