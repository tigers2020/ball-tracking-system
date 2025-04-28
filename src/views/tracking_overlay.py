#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tracking Overlay Widget.
This module contains the TrackingOverlay class for displaying real-time tracking coordinates.
"""

import logging
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QFrame, 
    QVBoxLayout, QGroupBox, QSizePolicy, QApplication
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont, QColor

from src.utils.ui_theme import Colors, Fonts, StyleManager
from src.utils.ui_tracking_constants import TrackingColors, TrackingLayout, TrackingFormatting


class TrackingOverlay(QWidget):
    """
    Widget displaying real-time tracking information including 2D and 3D coordinates.
    Designed to be placed at the top of the ImageView.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the tracking overlay widget.
        
        Args:
            parent (QWidget, optional): Parent widget
        """
        super(TrackingOverlay, self).__init__(parent)
        
        # Initialize internal state
        self._tracking_data = {}
        
        # Set up the UI
        self._setup_ui()
        
        # Initialize with empty data
        self.reset_tracking_info()
        
        # Ensure overlay displays on top
        self.raise_()
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        
    def _setup_ui(self):
        """Set up the tracking overlay user interface."""
        # Main layout
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(
            TrackingLayout.MARGIN, 
            TrackingLayout.MARGIN, 
            TrackingLayout.MARGIN, 
            TrackingLayout.MARGIN
        )
        self.main_layout.setSpacing(TrackingLayout.GROUP_SPACING)
        
        # Create frame with border and background for better visibility
        self.setAutoFillBackground(True)
        self.setStyleSheet(f"""
            TrackingOverlay {{
                background-color: rgba(20, 20, 20, 230);
                border: 2px solid #1f6feb;
                border-radius: 4px;
            }}
            QGroupBox {{
                font-weight: bold;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 3px;
                margin-top: 0.5em;
                padding-top: 0.5em;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
                background-color: #1f6feb;
            }}
            QLabel {{
                color: {TrackingColors.TEXT_PRIMARY};
                font-family: '{Fonts.BASE_FONT_FAMILY}';
                font-size: {Fonts.BASE_FONT_SIZE}pt;
            }}
            QLabel[isTitle="true"] {{
                font-weight: bold;
                color: {Colors.ACCENT.name()};
            }}
            QLabel[isValue="true"] {{
                font-family: monospace;
                font-weight: bold;
                background-color: rgba(50, 50, 50, 180);
                border-radius: 2px;
                padding: 2px;
            }}
            QLabel[isStatus="true"] {{
                font-weight: bold;
                border-radius: {TrackingFormatting.STATUS_BORDER_RADIUS};
                padding: {TrackingFormatting.STATUS_PADDING};
            }}
        """)
        
        # Set fixed height for the overlay
        self.setFixedHeight(TrackingLayout.HEIGHT)
        self.setMinimumWidth(400)  # Ensure minimum width
        
        # Set size policy to make sure the widget takes space
        size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        size_policy.setHorizontalStretch(1)
        self.setSizePolicy(size_policy)
        
        # Frame info section
        self.frame_group = QGroupBox("Frame")
        self.frame_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        frame_layout = QVBoxLayout(self.frame_group)
        frame_layout.setContentsMargins(
            TrackingLayout.MARGIN, 
            TrackingLayout.MARGIN, 
            TrackingLayout.MARGIN, 
            TrackingLayout.MARGIN
        )
        frame_layout.setSpacing(5)
        
        self.frame_label = QLabel("0")
        self.frame_label.setProperty("isValue", True)
        self.frame_label.setAlignment(Qt.AlignCenter)
        frame_layout.addWidget(self.frame_label)
        
        # Status indicator
        self.status_label = QLabel("Waiting")
        self.status_label.setProperty("isStatus", True)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(f"background-color: {TrackingColors.WARNING};")
        frame_layout.addWidget(self.status_label)
        
        self.main_layout.addWidget(self.frame_group)
        
        # 2D Coordinates section
        self.coords_2d_group = QGroupBox("2D Coordinates")
        self.coords_2d_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        coords_2d_layout = QVBoxLayout(self.coords_2d_group)
        coords_2d_layout.setContentsMargins(
            TrackingLayout.MARGIN, 
            TrackingLayout.MARGIN, 
            TrackingLayout.MARGIN, 
            TrackingLayout.MARGIN
        )
        coords_2d_layout.setSpacing(5)
        
        self.left_coords_label = QLabel("Left: (0.0, 0.0)")
        self.left_coords_label.setProperty("isValue", True)
        self.left_coords_label.setMinimumWidth(TrackingLayout.VALUE_MIN_WIDTH)
        coords_2d_layout.addWidget(self.left_coords_label)
        
        self.right_coords_label = QLabel("Right: (0.0, 0.0)")
        self.right_coords_label.setProperty("isValue", True)
        self.right_coords_label.setMinimumWidth(TrackingLayout.VALUE_MIN_WIDTH)
        coords_2d_layout.addWidget(self.right_coords_label)
        
        self.main_layout.addWidget(self.coords_2d_group)
        
        # 3D World Coordinates section
        self.coords_3d_group = QGroupBox("3D World Coordinates")
        self.coords_3d_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        coords_3d_layout = QVBoxLayout(self.coords_3d_group)
        coords_3d_layout.setContentsMargins(
            TrackingLayout.MARGIN, 
            TrackingLayout.MARGIN, 
            TrackingLayout.MARGIN, 
            TrackingLayout.MARGIN
        )
        coords_3d_layout.setSpacing(5)
        
        self.world_coords_label = QLabel("Position: (0.00 m, 0.00 m, 0.00 m)")
        self.world_coords_label.setProperty("isValue", True)
        self.world_coords_label.setMinimumWidth(TrackingLayout.VALUE_MIN_WIDTH)
        coords_3d_layout.addWidget(self.world_coords_label)
        
        self.confidence_label = QLabel("Confidence: 0.0%")
        self.confidence_label.setProperty("isValue", True)
        self.confidence_label.setMinimumWidth(TrackingLayout.VALUE_MIN_WIDTH)
        coords_3d_layout.addWidget(self.confidence_label)
        
        self.main_layout.addWidget(self.coords_3d_group)
        
        # Processing info section
        self.processing_group = QGroupBox("Processing")
        self.processing_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        processing_layout = QVBoxLayout(self.processing_group)
        processing_layout.setContentsMargins(
            TrackingLayout.MARGIN, 
            TrackingLayout.MARGIN, 
            TrackingLayout.MARGIN, 
            TrackingLayout.MARGIN
        )
        processing_layout.setSpacing(5)
        
        self.processing_time_label = QLabel("0.0 ms")
        self.processing_time_label.setProperty("isValue", True)
        self.processing_time_label.setAlignment(Qt.AlignCenter)
        self.processing_time_label.setMinimumWidth(TrackingLayout.LABEL_MIN_WIDTH)
        processing_layout.addWidget(self.processing_time_label)
        
        self.fps_label = QLabel("0.0 FPS")
        self.fps_label.setProperty("isValue", True)
        self.fps_label.setAlignment(Qt.AlignCenter)
        self.fps_label.setMinimumWidth(TrackingLayout.LABEL_MIN_WIDTH)
        processing_layout.addWidget(self.fps_label)
        
        self.main_layout.addWidget(self.processing_group)
    
    def reset_tracking_info(self):
        """Reset all tracking information to default values."""
        self.update_tracking_info({
            'frame_idx': 0,
            'left_coords': (0, 0),
            'right_coords': (0, 0),
            'world_coords': (0, 0, 0),
            'confidence': 0,
            'status': 'Waiting',
            'process_time': 0,
            'fps': 0
        })
    
    @Slot(dict)
    def update_tracking_info(self, tracking_data=None):
        """
        Update the tracking information display.
        
        Args:
            tracking_data (dict): Dictionary containing tracking information
                Keys:
                - frame_idx (int): Current frame index
                - left_coords/left_2d (tuple): 2D coordinates in left image (x, y)
                - right_coords/right_2d (tuple): 2D coordinates in right image (x, y)
                - world_coords/world_3d (tuple): 3D world coordinates (x, y, z) in meters
                - confidence (float): Detection confidence (0-100)
                - status (str): Tracking status (e.g., 'Tracking', 'Lost', 'Waiting')
                - process_time/processing_time (float): Processing time in milliseconds
                - fps (float): Frames per second
        """
        if tracking_data is None:
            tracking_data = {}
            
        # Log received data for debugging
        logging.critical(f"★★★ OVERLAY UPDATE CALLED! Data: {tracking_data}")
        
        # Debug widget state
        logging.critical(f"OVERLAY WIDGET STATE: Visible={self.isVisible()}, Size={self.width()}x{self.height()}")
        
        try:
            # Get values with key compatibility for different naming conventions
            frame_idx = tracking_data.get('frame_idx', 0)
            
            # Handle different key naming formats for coordinates with careful extraction
            left_coords = tracking_data.get('left_coords') or tracking_data.get('left_2d', (0.0, 0.0))
            if left_coords is None or not isinstance(left_coords, tuple):
                if isinstance(left_coords, (list, np.ndarray)):
                    left_coords = tuple(left_coords)
                else:
                    left_coords = (0.0, 0.0)
                    
            right_coords = tracking_data.get('right_coords') or tracking_data.get('right_2d', (0.0, 0.0))
            if right_coords is None or not isinstance(right_coords, tuple):
                if isinstance(right_coords, (list, np.ndarray)):
                    right_coords = tuple(right_coords)
                else:
                    right_coords = (0.0, 0.0)
                
            # Fix for numpy array world_coords - don't use 'or' operator with numpy arrays
            if 'world_coords' in tracking_data and tracking_data['world_coords'] is not None:
                world_coords = tracking_data['world_coords']
            elif 'world_3d' in tracking_data and tracking_data['world_3d'] is not None:
                world_coords = tracking_data['world_3d']
            else:
                world_coords = (0.0, 0.0, 0.0)
                
            # Convert numpy array to tuple if needed
            if isinstance(world_coords, np.ndarray):
                world_coords = tuple(float(v) for v in world_coords)
            elif not isinstance(world_coords, tuple):
                if isinstance(world_coords, list):
                    world_coords = tuple(world_coords)
                else:
                    world_coords = (0.0, 0.0, 0.0)
            
            # Ensure we have exactly 2 values for 2D coordinates and 3 for 3D
            if len(left_coords) != 2:
                left_coords = left_coords[:2] if len(left_coords) > 2 else (left_coords[0], 0.0) if len(left_coords) == 1 else (0.0, 0.0)
            if len(right_coords) != 2:
                right_coords = right_coords[:2] if len(right_coords) > 2 else (right_coords[0], 0.0) if len(right_coords) == 1 else (0.0, 0.0)
            if len(world_coords) != 3:
                world_coords = world_coords[:3] if len(world_coords) > 3 else world_coords + (0.0,) * (3 - len(world_coords))
            
            # Force to float values
            left_coords = tuple(float(v) for v in left_coords)
            right_coords = tuple(float(v) for v in right_coords)
            world_coords = tuple(float(v) for v in world_coords)
            
            confidence = float(tracking_data.get('confidence', 0))
            status = tracking_data.get('status', 'Waiting')
            
            # Handle different key naming for processing time
            process_time = float(tracking_data.get('process_time') or tracking_data.get('processing_time', 0))
            fps = float(tracking_data.get('fps', 0))
            
            # Update labels with formatted values
            self.frame_label.setText(f"{frame_idx}")
            
            # Format coordinates with consistent precision
            x_left, y_left = left_coords
            x_right, y_right = right_coords
            x_world, y_world, z_world = world_coords
            
            # Log the actual values being displayed
            logging.critical(f"★★★ UPDATING DISPLAY - Left: ({x_left:.1f}, {y_left:.1f}), Right: ({x_right:.1f}, {y_right:.1f})")
            logging.critical(f"★★★ UPDATING DISPLAY - 3D: ({x_world:.2f}, {y_world:.2f}, {z_world:.2f})")
            
            # Actually update the text
            self.left_coords_label.setText(
                f"Left: ({TrackingFormatting.COORDINATE_FORMAT.format(x_left)}, "
                f"{TrackingFormatting.COORDINATE_FORMAT.format(y_left)})"
            )
            
            self.right_coords_label.setText(
                f"Right: ({TrackingFormatting.COORDINATE_FORMAT.format(x_right)}, "
                f"{TrackingFormatting.COORDINATE_FORMAT.format(y_right)})"
            )
            
            # Format 3D coordinates with units
            self.world_coords_label.setText(
                f"Position: ("
                f"{TrackingFormatting.POSITION_FORMAT.format(x_world)}, "
                f"{TrackingFormatting.POSITION_FORMAT.format(y_world)}, "
                f"{TrackingFormatting.POSITION_FORMAT.format(z_world)}"
                f")"
            )
            
            self.confidence_label.setText(
                f"Confidence: {TrackingFormatting.CONFIDENCE_FORMAT.format(confidence)}"
            )
            
            # Update processing information
            self.processing_time_label.setText(
                f"{TrackingFormatting.TIME_FORMAT.format(process_time)}"
            )
            
            self.fps_label.setText(
                f"{TrackingFormatting.FPS_FORMAT.format(fps)}"
            )
            
            # Update status with color coding
            self.status_label.setText(status)
            
            # Set status color based on status value
            status_lower = status.lower()
            if status_lower == 'tracking' or status_lower == 'detected':
                self.status_label.setStyleSheet(f"background-color: {TrackingColors.SUCCESS};")
            elif status_lower == 'lost' or status_lower == 'failed':
                self.status_label.setStyleSheet(f"background-color: {TrackingColors.ERROR};")
            elif status_lower == 'predicted':
                self.status_label.setStyleSheet(f"background-color: {TrackingColors.INFO};")
            else:
                self.status_label.setStyleSheet(f"background-color: {TrackingColors.WARNING};")
            
            # Store tracking data for reference
            self._tracking_data = tracking_data
            
            # Force widget update to ensure repaint
            QApplication.processEvents()  # Process pending events
            self.update()
            self.repaint()  # Explicitly call repaint for immediate update
            
            # Log update completion for debugging
            logging.critical(f"★★★ TRACKING OVERLAY UPDATED SUCCESSFULLY! Frame: {frame_idx}")
            
        except Exception as e:
            logging.critical(f"★★★ ERROR updating tracking overlay: {str(e)}")
            import traceback
            logging.critical(traceback.format_exc())
        
    def showEvent(self, event):
        """Handle widget show event to ensure proper stacking order."""
        super(TrackingOverlay, self).showEvent(event)
        
        # Ensure this widget is visible and on top of siblings
        self.raise_()
        
        # Log visibility for debugging
        logging.debug("Tracking overlay shown and brought to front") 