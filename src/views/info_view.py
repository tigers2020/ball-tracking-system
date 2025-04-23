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
        self.left_pixel_coords = {"x": 0, "y": 0, "r": 0}   # Left camera 2D coordinates
        self.right_pixel_coords = {"x": 0, "y": 0, "r": 0}  # Right camera 2D coordinates
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
        
        # Left 2D pixel coordinates group
        left_pixel_group = QGroupBox("Left Camera (2D)")
        left_pixel_layout = QFormLayout()
        self.left_pixel_x_label = QLabel("0")
        self.left_pixel_y_label = QLabel("0")
        self.left_pixel_r_label = QLabel("0")
        left_pixel_layout.addRow("X:", self.left_pixel_x_label)
        left_pixel_layout.addRow("Y:", self.left_pixel_y_label)
        left_pixel_layout.addRow("R:", self.left_pixel_r_label)
        left_pixel_group.setLayout(left_pixel_layout)
        
        # Right 2D pixel coordinates group
        right_pixel_group = QGroupBox("Right Camera (2D)")
        right_pixel_layout = QFormLayout()
        self.right_pixel_x_label = QLabel("0")
        self.right_pixel_y_label = QLabel("0")
        self.right_pixel_r_label = QLabel("0")
        right_pixel_layout.addRow("X:", self.right_pixel_x_label)
        right_pixel_layout.addRow("Y:", self.right_pixel_y_label)
        right_pixel_layout.addRow("R:", self.right_pixel_r_label)
        right_pixel_group.setLayout(right_pixel_layout)
        
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
        main_layout.addWidget(left_pixel_group)
        main_layout.addWidget(right_pixel_group)
        main_layout.addWidget(position_group)
    
    def set_detection_rate(self, rate):
        """
        Set the detection rate value.
        
        Args:
            rate (float): Detection rate (0.0 to 1.0)
        """
        self.detection_rate = rate
        self.detection_label.setText(f"{rate:.2%}")
    
    def set_left_pixel_coords(self, x, y, r=0):
        """
        Set the left camera 2D pixel coordinates.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            r (int, optional): Radius
        """
        self.left_pixel_coords["x"] = x
        self.left_pixel_coords["y"] = y
        self.left_pixel_coords["r"] = r
        self.left_pixel_x_label.setText(str(x))
        self.left_pixel_y_label.setText(str(y))
        self.left_pixel_r_label.setText(str(r))
    
    def set_right_pixel_coords(self, x, y, r=0):
        """
        Set the right camera 2D pixel coordinates.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            r (int, optional): Radius
        """
        self.right_pixel_coords["x"] = x
        self.right_pixel_coords["y"] = y
        self.right_pixel_coords["r"] = r
        self.right_pixel_x_label.setText(str(x))
        self.right_pixel_y_label.setText(str(y))
        self.right_pixel_r_label.setText(str(r))
    
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
        self.set_left_pixel_coords(0, 0, 0)
        self.set_right_pixel_coords(0, 0, 0)
        self.set_position_coords(0.0, 0.0, 0.0)
    
    def connect_tracking_controller(self, controller):
        """
        Connect to a ball tracking controller to receive updates.
        
        Args:
            controller: BallTrackingController instance
        """
        if controller:
            # Connect detection update signal
            controller.detection_updated.connect(self._on_detection_updated)
            logging.info("Connected to ball tracking controller")
    
    def _on_detection_updated(self, detection_rate, left_coords, right_coords):
        """
        Handle detection update signal from ball tracking controller.
        
        Args:
            detection_rate (float): Current detection rate
            left_coords (tuple): Left camera coordinates (x, y, r) or None
            right_coords (tuple): Right camera coordinates (x, y, r) or None
        """
        # Update detection rate
        self.set_detection_rate(detection_rate)
        
        # Update left pixel coordinates
        if left_coords:
            self.set_left_pixel_coords(left_coords[0], left_coords[1], left_coords[2])
        else:
            self.set_left_pixel_coords(0, 0, 0)
            
        # Update right pixel coordinates
        if right_coords:
            self.set_right_pixel_coords(right_coords[0], right_coords[1], right_coords[2])
        else:
            self.set_right_pixel_coords(0, 0, 0)
            
        logging.debug(f"Info view updated: rate={detection_rate:.2f}, left={left_coords}, right={right_coords}") 