#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tracking Overlay module.
This module contains the TrackingOverlay class for displaying real-time tracking coordinates.
"""

import logging
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QFrame
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont

from src.utils.ui_constants import TRACKING_OVERLAY


class TrackingOverlay(QWidget):
    """
    Widget for displaying real-time tracking coordinates.
    Shows 2D coordinates (left and right) and 3D world coordinates.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the tracking overlay.
        
        Args:
            parent (QWidget, optional): Parent widget
        """
        super(TrackingOverlay, self).__init__(parent)
        
        # Set up UI
        self._setup_ui()
        
        # Initialize tracking data
        self.reset_data()
    
    def _setup_ui(self):
        """Set up the user interface."""
        # Set fixed height for the overlay
        self.setFixedHeight(TRACKING_OVERLAY.HEIGHT)
        
        # Create container frame
        self.frame = QFrame(self)
        self.frame.setObjectName("trackingOverlayFrame")
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setAutoFillBackground(True)
        
        # Set stylesheet for the frame
        self.frame.setStyleSheet(f"""
            #trackingOverlayFrame {{
                background-color: {TRACKING_OVERLAY.BACKGROUND_COLOR};
                border-radius: {TRACKING_OVERLAY.BORDER_RADIUS}px;
                border: 1px solid {TRACKING_OVERLAY.BORDER_COLOR};
            }}
            QLabel {{
                color: {TRACKING_OVERLAY.TEXT_COLOR};
                font-size: {TRACKING_OVERLAY.FONT_SIZE}pt;
            }}
            QLabel[labelType="title"] {{
                color: {TRACKING_OVERLAY.LABEL_COLOR};
                font-weight: bold;
            }}
        """)
        
        # Create layout for the frame
        main_layout = QHBoxLayout(self.frame)
        main_layout.setContentsMargins(
            TRACKING_OVERLAY.PADDING, 
            TRACKING_OVERLAY.PADDING, 
            TRACKING_OVERLAY.PADDING, 
            TRACKING_OVERLAY.PADDING
        )
        main_layout.setSpacing(TRACKING_OVERLAY.SPACING)
        
        # Create labels for tracking information
        # Frame information
        self.frame_title_label = QLabel("Frame:")
        self.frame_title_label.setProperty("labelType", "title")
        self.frame_value_label = QLabel("0")
        
        # 2D left coordinates
        self.left_2d_title_label = QLabel("2D (L):")
        self.left_2d_title_label.setProperty("labelType", "title")
        self.left_2d_value_label = QLabel("(--, --)")
        
        # 2D right coordinates
        self.right_2d_title_label = QLabel("2D (R):")
        self.right_2d_title_label.setProperty("labelType", "title")
        self.right_2d_value_label = QLabel("(--, --)")
        
        # 3D world coordinates
        self.world_3d_title_label = QLabel("3D (X,Y,Z):")
        self.world_3d_title_label.setProperty("labelType", "title")
        self.world_3d_value_label = QLabel("(--, --, --)")
        
        # Processing time
        self.time_title_label = QLabel("Time:")
        self.time_title_label.setProperty("labelType", "title")
        self.time_value_label = QLabel("-- ms")
        
        # Confidence/Status
        self.status_title_label = QLabel("Status:")
        self.status_title_label.setProperty("labelType", "title")
        self.status_value_label = QLabel("No tracking")
        
        # Add labels to layout
        main_layout.addWidget(self.frame_title_label)
        main_layout.addWidget(self.frame_value_label)
        main_layout.addStretch(1)
        
        main_layout.addWidget(self.left_2d_title_label)
        main_layout.addWidget(self.left_2d_value_label)
        main_layout.addStretch(1)
        
        main_layout.addWidget(self.right_2d_title_label)
        main_layout.addWidget(self.right_2d_value_label)
        main_layout.addStretch(1)
        
        main_layout.addWidget(self.world_3d_title_label)
        main_layout.addWidget(self.world_3d_value_label)
        main_layout.addStretch(1)
        
        main_layout.addWidget(self.time_title_label)
        main_layout.addWidget(self.time_value_label)
        main_layout.addStretch(1)
        
        main_layout.addWidget(self.status_title_label)
        main_layout.addWidget(self.status_value_label)
        
        # Create layout for the widget
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.frame)
    
    def reset_data(self):
        """Reset all tracking data to default values."""
        self.frame_value_label.setText("0")
        self.left_2d_value_label.setText("(--, --)")
        self.right_2d_value_label.setText("(--, --)")
        self.world_3d_value_label.setText("(--, --, --)")
        self.time_value_label.setText("-- ms")
        self.status_value_label.setText("No tracking")
    
    @Slot(dict)
    def update_tracking_info(self, tracking_data):
        """
        Update tracking information display.
        
        Args:
            tracking_data (dict): Dictionary containing tracking information
                {
                    'frame_idx': int,
                    'left_2d': (x, y) or None,
                    'right_2d': (x, y) or None,
                    'world_3d': (x, y, z) or None,
                    'processing_time': float,
                    'status': str,
                    'confidence': float
                }
        """
        # Update frame number
        if 'frame_idx' in tracking_data:
            self.frame_value_label.setText(str(tracking_data['frame_idx']))
        
        # Update 2D left coordinates
        if 'left_2d' in tracking_data and tracking_data['left_2d'] is not None:
            x, y = tracking_data['left_2d']
            self.left_2d_value_label.setText(
                f"({x:.{TRACKING_OVERLAY.DECIMAL_PLACES_2D}f}, "
                f"{y:.{TRACKING_OVERLAY.DECIMAL_PLACES_2D}f})"
            )
        else:
            self.left_2d_value_label.setText("(--, --)")
        
        # Update 2D right coordinates
        if 'right_2d' in tracking_data and tracking_data['right_2d'] is not None:
            x, y = tracking_data['right_2d']
            self.right_2d_value_label.setText(
                f"({x:.{TRACKING_OVERLAY.DECIMAL_PLACES_2D}f}, "
                f"{y:.{TRACKING_OVERLAY.DECIMAL_PLACES_2D}f})"
            )
        else:
            self.right_2d_value_label.setText("(--, --)")
        
        # Update 3D world coordinates
        if 'world_3d' in tracking_data and tracking_data['world_3d'] is not None:
            x, y, z = tracking_data['world_3d']
            self.world_3d_value_label.setText(
                f"({x:.{TRACKING_OVERLAY.DECIMAL_PLACES_3D}f} m, "
                f"{y:.{TRACKING_OVERLAY.DECIMAL_PLACES_3D}f} m, "
                f"{z:.{TRACKING_OVERLAY.DECIMAL_PLACES_3D}f} m)"
            )
        else:
            self.world_3d_value_label.setText("(--, --, --)")
        
        # Update processing time
        if 'processing_time' in tracking_data:
            self.time_value_label.setText(f"{tracking_data['processing_time']:.1f} ms")
        
        # Update status/confidence
        if 'status' in tracking_data:
            status_text = tracking_data['status']
            if 'confidence' in tracking_data and tracking_data['confidence'] is not None:
                confidence = tracking_data['confidence']
                status_text += f" ({confidence:.0%})"
            self.status_value_label.setText(status_text) 