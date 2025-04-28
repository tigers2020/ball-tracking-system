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
    
    def update_tracking_info(self, tracking_data=None):
        """Update tracking information display with current tracking data.
        
        Args:
            tracking_data (dict): Dictionary containing tracking information
        """
        if tracking_data is None:
            tracking_data = {}
            
        # Log for debugging
        logging.debug(f"Updating tracking overlay with data: {tracking_data}")
        
        # Update frame index
        frame_idx = tracking_data.get('frame_idx', 'N/A')
        self.frame_value_label.setText(f"Frame: {frame_idx}")
        
        # 키 이름 호환성 처리 추가 - left/right/world 좌표에 대해 두 가지 키 모두 지원
        left_coords = tracking_data.get('left_coords') or tracking_data.get('left_2d', (0, 0))
        right_coords = tracking_data.get('right_coords') or tracking_data.get('right_2d', (0, 0))
        world_coords = tracking_data.get('world_coords') or tracking_data.get('world_3d', (0, 0, 0))
        
        if left_coords:
            left_x, left_y = left_coords
            self.left_2d_value_label.setText(f"Left: ({left_x:.1f}, {left_y:.1f})")
        else:
            self.left_2d_value_label.setText("Left: (-, -)")
            
        if right_coords:
            right_x, right_y = right_coords
            self.right_2d_value_label.setText(f"Right: ({right_x:.1f}, {right_y:.1f})")
        else:
            self.right_2d_value_label.setText("Right: (-, -)")
        
        # Update world coordinates
        if world_coords:
            x, y, z = world_coords
            self.world_3d_value_label.setText(f"World: ({x:.2f}, {y:.2f}, {z:.2f})")
        else:
            self.world_3d_value_label.setText("World: (-, -, -)")
        
        # Update processing time - 키 이름 호환성 처리 추가
        process_time = tracking_data.get('process_time') or tracking_data.get('processing_time', 0.0)
        self.time_value_label.setText(f"Process: {process_time:.1f} ms")
        
        # Update status
        status = tracking_data.get('status', 'No data')
        self.status_value_label.setText(f"Status: {status}")
        
        # Highlight status with color based on success/failure
        if status.lower() in ['success', 'detected', 'tracking']:
            self.status_value_label.setStyleSheet("QLabel { color: green; }")
        elif status.lower() in ['lost', 'not detected']:
            self.status_value_label.setStyleSheet("QLabel { color: orange; }")
        elif status.lower() in ['error', 'failed']:
            self.status_value_label.setStyleSheet("QLabel { color: red; }")
        else:
            self.status_value_label.setStyleSheet("")  # Reset to default
            
        # Force immediate update to prevent visual lag
        self.update() 