#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Info View module.
This module contains the InfoView class for displaying detection information.
"""

import logging
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
    QLabel, QGridLayout, QFormLayout
)

from src.utils.ui_constants import Layout


class InfoView(QWidget):
    """
    Widget for displaying detection information in the Stereo Image Player.
    Displays detection rate, 2D pixel coordinates, and 3D position coordinates.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the info view.
        
        Args:
            parent (QWidget, optional): Parent widget
        """
        super(InfoView, self).__init__(parent)
        
        # Default values
        self.detection_rate = 0.0  # Percentage
        self.pixel_coords_2d = {"x": 0, "y": 0}
        self.position_coords_3d = {"x": 0.0, "y": 0.0, "z": 0.0}
        
        # Set up UI
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(Layout.MARGIN, Layout.MARGIN, Layout.MARGIN, Layout.MARGIN)
        main_layout.setSpacing(Layout.SPACING)
        
        # Detection rate group
        detection_group = QGroupBox("Detection Rate")
        detection_layout = QVBoxLayout()
        self.detection_label = QLabel("0.00%")
        self.detection_label.setAlignment(Qt.AlignCenter)
        detection_layout.addWidget(self.detection_label)
        detection_group.setLayout(detection_layout)
        
        # 2D pixel coordinates group
        pixel_group = QGroupBox("2D Pixel Coordinates")
        pixel_layout = QFormLayout()
        self.pixel_x_label = QLabel("0")
        self.pixel_y_label = QLabel("0")
        pixel_layout.addRow("X:", self.pixel_x_label)
        pixel_layout.addRow("Y:", self.pixel_y_label)
        pixel_group.setLayout(pixel_layout)
        
        # 3D position coordinates group
        position_group = QGroupBox("3D Position Coordinates")
        position_layout = QFormLayout()
        self.position_x_label = QLabel("0.000")
        self.position_y_label = QLabel("0.000")
        self.position_z_label = QLabel("0.000")
        position_layout.addRow("X:", self.position_x_label)
        position_layout.addRow("Y:", self.position_y_label)
        position_layout.addRow("Z:", self.position_z_label)
        position_group.setLayout(position_layout)
        
        # Add all groups to main layout
        main_layout.addWidget(detection_group)
        main_layout.addWidget(pixel_group)
        main_layout.addWidget(position_group)
    
    def set_detection_rate(self, rate):
        """
        Set the detection rate value.
        
        Args:
            rate (float): Detection rate (0.0 to 1.0)
        """
        self.detection_rate = rate
        self.detection_label.setText(f"{rate:.2%}")
    
    def set_pixel_coords(self, x, y):
        """
        Set the 2D pixel coordinates.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
        """
        self.pixel_coords_2d["x"] = x
        self.pixel_coords_2d["y"] = y
        self.pixel_x_label.setText(str(x))
        self.pixel_y_label.setText(str(y))
    
    def set_position_coords(self, x, y, z):
        """
        Set the 3D position coordinates.
        
        Args:
            x (float): X coordinate
            y (float): Y coordinate
            z (float): Z coordinate
        """
        self.position_coords_3d["x"] = x
        self.position_coords_3d["y"] = y
        self.position_coords_3d["z"] = z
        self.position_x_label.setText(f"{x:.3f}")
        self.position_y_label.setText(f"{y:.3f}")
        self.position_z_label.setText(f"{z:.3f}")
    
    def clear_info(self):
        """
        Clear all information and reset to default values.
        """
        self.set_detection_rate(0.0)
        self.set_pixel_coords(0, 0)
        self.set_position_coords(0.0, 0.0, 0.0) 