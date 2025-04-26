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
    QPushButton, QFileDialog, QLineEdit, QTabWidget, 
    QFormLayout, QDoubleSpinBox, QScrollArea
)

from src.utils.ui_constants import Layout, FileDialog
from src.utils.config_manager import ConfigManager


class SettingView(QWidget):
    """
    Widget for the settings tab in the Stereo Image Player.
    Contains application settings.
    """
    
    # Signals
    settings_changed = Signal(dict)
    folder_selected = Signal(str)
    camera_settings_changed = Signal(dict)
    
    def __init__(self, parent=None):
        """
        Initialize the settings view.
        
        Args:
            parent (QWidget, optional): Parent widget
        """
        super(SettingView, self).__init__(parent)
        
        # Create configuration manager
        self.config_manager = ConfigManager()
        
        # Default settings
        self.settings = {
            "last_image_folder": ""
        }
        
        # Load camera settings
        self.camera_settings = self.config_manager.get_camera_settings()
        
        # Set up UI
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(Layout.MARGIN, Layout.MARGIN, Layout.MARGIN, Layout.MARGIN)
        main_layout.setSpacing(Layout.SPACING)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create folder settings tab
        folder_tab = QWidget()
        self.tab_widget.addTab(folder_tab, "Folder Settings")
        
        # Folder settings layout
        folder_layout = QVBoxLayout(folder_tab)
        
        # Folder settings group
        folder_group = QGroupBox("Image Folder Settings")
        folder_group_layout = QVBoxLayout()
        
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
        
        folder_group_layout.addLayout(folder_path_layout)
        
        # Open button
        self.open_button = QPushButton("Open This Folder")
        self.open_button.clicked.connect(self._on_open_clicked)
        folder_group_layout.addWidget(self.open_button)
        
        folder_group.setLayout(folder_group_layout)
        folder_layout.addWidget(folder_group)
        folder_layout.addStretch()
        
        # Create camera settings tab
        camera_tab = QWidget()
        self.tab_widget.addTab(camera_tab, "Camera Settings")
        
        # Create a scroll area for camera settings
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        camera_container = QWidget()
        camera_layout = QVBoxLayout(camera_container)
        
        # Camera settings group
        camera_group = QGroupBox("Camera Settings")
        camera_group_layout = QFormLayout()
        
        # Camera location
        self.camera_location_x = self._create_double_spin_box(-1000, 1000, 1, self.camera_settings["camera_location_x"])
        self.camera_location_y = self._create_double_spin_box(-1000, 1000, 1, self.camera_settings["camera_location_y"])
        self.camera_location_z = self._create_double_spin_box(-1000, 1000, 1, self.camera_settings["camera_location_z"])
        camera_group_layout.addRow("Camera Location X:", self.camera_location_x)
        camera_group_layout.addRow("Camera Location Y:", self.camera_location_y)
        camera_group_layout.addRow("Camera Location Z:", self.camera_location_z)
        
        # Camera rotation
        self.camera_rotation_x = self._create_double_spin_box(-180, 180, 1, self.camera_settings["camera_rotation_x"])
        self.camera_rotation_y = self._create_double_spin_box(-180, 180, 1, self.camera_settings["camera_rotation_y"])
        self.camera_rotation_z = self._create_double_spin_box(-180, 180, 1, self.camera_settings["camera_rotation_z"])
        camera_group_layout.addRow("Camera Rotation X:", self.camera_rotation_x)
        camera_group_layout.addRow("Camera Rotation Y:", self.camera_rotation_y)
        camera_group_layout.addRow("Camera Rotation Z:", self.camera_rotation_z)
        
        # Camera parameters
        self.focal_length = self._create_double_spin_box(1, 1000, 0.1, self.camera_settings["focal_length_mm"])
        self.baseline = self._create_double_spin_box(0.01, 10, 0.01, self.camera_settings["baseline_m"])
        camera_group_layout.addRow("Focal Length (mm):", self.focal_length)
        camera_group_layout.addRow("Baseline (m):", self.baseline)
        
        # Sensor parameters
        self.sensor_width = self._create_double_spin_box(1, 100, 0.1, self.camera_settings["sensor_width_mm"])
        self.sensor_height = self._create_double_spin_box(1, 100, 0.1, self.camera_settings["sensor_height_mm"])
        camera_group_layout.addRow("Sensor Width (mm):", self.sensor_width)
        camera_group_layout.addRow("Sensor Height (mm):", self.sensor_height)
        
        # Principal point
        self.principal_point_x = self._create_double_spin_box(0, 1000, 0.1, self.camera_settings["principal_point_x"])
        self.principal_point_y = self._create_double_spin_box(0, 1000, 0.1, self.camera_settings["principal_point_y"])
        camera_group_layout.addRow("Principal Point X:", self.principal_point_x)
        camera_group_layout.addRow("Principal Point Y:", self.principal_point_y)
        
        # Connect all camera spinboxes to update function
        self.camera_location_x.valueChanged.connect(self._on_camera_settings_changed)
        self.camera_location_y.valueChanged.connect(self._on_camera_settings_changed)
        self.camera_location_z.valueChanged.connect(self._on_camera_settings_changed)
        self.camera_rotation_x.valueChanged.connect(self._on_camera_settings_changed)
        self.camera_rotation_y.valueChanged.connect(self._on_camera_settings_changed)
        self.camera_rotation_z.valueChanged.connect(self._on_camera_settings_changed)
        self.focal_length.valueChanged.connect(self._on_camera_settings_changed)
        self.baseline.valueChanged.connect(self._on_camera_settings_changed)
        self.sensor_width.valueChanged.connect(self._on_camera_settings_changed)
        self.sensor_height.valueChanged.connect(self._on_camera_settings_changed)
        self.principal_point_x.valueChanged.connect(self._on_camera_settings_changed)
        self.principal_point_y.valueChanged.connect(self._on_camera_settings_changed)
        
        camera_group.setLayout(camera_group_layout)
        camera_layout.addWidget(camera_group)
        camera_layout.addStretch()
        
        scroll_area.setWidget(camera_container)
        camera_tab_layout = QVBoxLayout(camera_tab)
        camera_tab_layout.addWidget(scroll_area)
    
    def _create_double_spin_box(self, min_val, max_val, step, value):
        """
        Create a QDoubleSpinBox with the specified properties.
        
        Args:
            min_val (float): Minimum value
            max_val (float): Maximum value
            step (float): Step value
            value (float): Initial value
            
        Returns:
            QDoubleSpinBox: Configured spin box
        """
        spin_box = QDoubleSpinBox()
        spin_box.setMinimum(min_val)
        spin_box.setMaximum(max_val)
        spin_box.setSingleStep(step)
        spin_box.setValue(value)
        spin_box.setDecimals(3)
        return spin_box
    
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
    
    def _on_camera_settings_changed(self):
        """Handle camera settings change."""
        # Update camera settings from UI values
        self.camera_settings["camera_location_x"] = self.camera_location_x.value()
        self.camera_settings["camera_location_y"] = self.camera_location_y.value()
        self.camera_settings["camera_location_z"] = self.camera_location_z.value()
        self.camera_settings["camera_rotation_x"] = self.camera_rotation_x.value()
        self.camera_settings["camera_rotation_y"] = self.camera_rotation_y.value()
        self.camera_settings["camera_rotation_z"] = self.camera_rotation_z.value()
        self.camera_settings["focal_length_mm"] = self.focal_length.value()
        self.camera_settings["baseline_m"] = self.baseline.value()
        self.camera_settings["sensor_width_mm"] = self.sensor_width.value()
        self.camera_settings["sensor_height_mm"] = self.sensor_height.value()
        self.camera_settings["principal_point_x"] = self.principal_point_x.value()
        self.camera_settings["principal_point_y"] = self.principal_point_y.value()
        
        # Save to configuration
        self.config_manager.set_camera_settings(self.camera_settings)
        
        # Emit signal
        self.camera_settings_changed.emit(self.camera_settings.copy())
        logging.debug("Camera settings updated")
    
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
        
    def get_camera_settings(self):
        """
        Get the current camera settings.
        
        Returns:
            dict: Current camera settings
        """
        return self.camera_settings.copy()
    
    def set_camera_settings(self, camera_settings):
        """
        Set camera settings and update UI.
        
        Args:
            camera_settings (dict): Camera settings
        """
        # Update settings
        for key, value in camera_settings.items():
            if key in self.camera_settings:
                self.camera_settings[key] = value
        
        # Update UI controls
        self.camera_location_x.setValue(self.camera_settings["camera_location_x"])
        self.camera_location_y.setValue(self.camera_settings["camera_location_y"])
        self.camera_location_z.setValue(self.camera_settings["camera_location_z"])
        self.camera_rotation_x.setValue(self.camera_settings["camera_rotation_x"])
        self.camera_rotation_y.setValue(self.camera_settings["camera_rotation_y"])
        self.camera_rotation_z.setValue(self.camera_settings["camera_rotation_z"])
        self.focal_length.setValue(self.camera_settings["focal_length_mm"])
        self.baseline.setValue(self.camera_settings["baseline_m"])
        self.sensor_width.setValue(self.camera_settings["sensor_width_mm"])
        self.sensor_height.setValue(self.camera_settings["sensor_height_mm"])
        self.principal_point_x.setValue(self.camera_settings["principal_point_x"])
        self.principal_point_y.setValue(self.camera_settings["principal_point_y"]) 