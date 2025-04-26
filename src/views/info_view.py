#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Info View module.
This module contains the InfoView class for displaying tracking information in a widget.
"""

import logging
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QGroupBox, QFormLayout
)
from PySide6.QtCore import Qt, Slot

from src.utils.constants import LAYOUT
from src.views.visualization.hsv_mask_visualizer import HSVMaskVisualizer
from src.views.visualization.roi_mask_visualizer import ROIMaskVisualizer
from src.views.visualization.hough_circle_visualizer import HoughCircleVisualizer
from src.views.visualization.kalman_path_visualizer import KalmanPathVisualizer
from src.views.widgets.inout_indicator import InOutLED


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
        
        # ROI values
        self.left_roi = {"x": 0, "y": 0, "width": 0, "height": 0}
        self.right_roi = {"x": 0, "y": 0, "width": 0, "height": 0}
        
        # Kalman state values
        self.left_state = {"pos": (0.0, 0.0), "vel": (0.0, 0.0)}
        self.right_state = {"pos": (0.0, 0.0), "vel": (0.0, 0.0)}
        
        # Controller reference (will be set later)
        self.tracking_controller = None
        self.game_analyzer = None
        
        # Visualizers list (will be initialized in _setup_visualizers)
        self._visualizers = []
        
        # Set up UI
        self._setup_ui()
        
        # Setup visualizers
        self._setup_visualizers()
    
    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(LAYOUT.MARGIN, LAYOUT.MARGIN, LAYOUT.MARGIN, LAYOUT.MARGIN)
        main_layout.setSpacing(LAYOUT.SPACING)
        
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
        position_group = QGroupBox("3D Court Position")
        position_layout = QFormLayout()
        self.position_x_label = QLabel("0.000")
        self.position_y_label = QLabel("0.000")
        self.position_z_label = QLabel("0.000")
        
        # Add IN/OUT indicator
        self.in_out_led = InOutLED()
        
        position_layout.addRow("X:", self.position_x_label)
        position_layout.addRow("Y:", self.position_y_label)
        position_layout.addRow("Z:", self.position_z_label)
        position_layout.addRow("IN/OUT:", self.in_out_led)
        position_group.setLayout(position_layout)
        
        # ROI information group
        roi_group = QGroupBox("ROI")
        roi_layout = QFormLayout()
        self.left_roi_label = QLabel("(0, 0, 0, 0)")
        self.right_roi_label = QLabel("(0, 0, 0, 0)")
        roi_layout.addRow("Left:", self.left_roi_label)
        roi_layout.addRow("Right:", self.right_roi_label)
        roi_group.setLayout(roi_layout)
        
        # Kalman state group
        kalman_group = QGroupBox("Kalman State")
        kalman_layout = QFormLayout()
        self.left_state_label = QLabel("pos=(0, 0), vel=(0, 0)")
        self.right_state_label = QLabel("pos=(0, 0), vel=(0, 0)")
        kalman_layout.addRow("Left:", self.left_state_label)
        kalman_layout.addRow("Right:", self.right_state_label)
        kalman_group.setLayout(kalman_layout)
        
        # Add all groups to main layout
        main_layout.addWidget(detection_group)
        main_layout.addWidget(left_pixel_group)
        main_layout.addWidget(right_pixel_group)
        main_layout.addWidget(position_group)
        main_layout.addWidget(roi_group)
        main_layout.addWidget(kalman_group)
    
    def _setup_visualizers(self):
        """Set up visualizers for different aspects of tracking."""
        # Initialize empty list (will be populated when controller is connected)
        self._visualizers = []
    
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
    
    def set_left_roi(self, x, y, width, height):
        """
        Set the left ROI coordinates.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            width (int): Width
            height (int): Height
        """
        self.left_roi = {"x": x, "y": y, "width": width, "height": height}
        self.left_roi_label.setText(f"({x}, {y}, {width}, {height})")
    
    def set_right_roi(self, x, y, width, height):
        """
        Set the right ROI coordinates.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            width (int): Width
            height (int): Height
        """
        self.right_roi = {"x": x, "y": y, "width": width, "height": height}
        self.right_roi_label.setText(f"({x}, {y}, {width}, {height})")
    
    def set_left_state(self, pos_x, pos_y, vel_x, vel_y):
        """
        Set the left camera Kalman state.
        
        Args:
            pos_x (float): Position X coordinate
            pos_y (float): Position Y coordinate
            vel_x (float): Velocity X component
            vel_y (float): Velocity Y component
        """
        self.left_state = {"pos": (pos_x, pos_y), "vel": (vel_x, vel_y)}
        self.left_state_label.setText(f"pos=({pos_x:.1f}, {pos_y:.1f}), vel=({vel_x:.1f}, {vel_y:.1f})")
    
    def set_right_state(self, pos_x, pos_y, vel_x, vel_y):
        """
        Set the right camera Kalman state.
        
        Args:
            pos_x (float): Position X coordinate
            pos_y (float): Position Y coordinate
            vel_x (float): Velocity X component
            vel_y (float): Velocity Y component
        """
        self.right_state = {"pos": (pos_x, pos_y), "vel": (vel_x, vel_y)}
        self.right_state_label.setText(f"pos=({pos_x:.1f}, {pos_y:.1f}), vel=({vel_x:.1f}, {vel_y:.1f})")
    
    def clear_info(self):
        """Reset all information displays."""
        self.set_detection_rate(0.0)
        self.set_left_pixel_coords(0, 0, 0)
        self.set_right_pixel_coords(0, 0, 0)
        self.set_position_coords(0.0, 0.0, 0.0)
        self.set_left_roi(0, 0, 0, 0)
        self.set_right_roi(0, 0, 0, 0)
        self.set_left_state(0.0, 0.0, 0.0, 0.0)
        self.set_right_state(0.0, 0.0, 0.0, 0.0)
    
    def connect_tracking_controller(self, controller):
        """
        Connect to a ball tracking controller to receive updates.
        
        Args:
            controller: BallTrackingController instance
        """
        if controller:
            self.tracking_controller = controller
            
            # Connect detection update signal with proper signature
            controller.detection_updated.connect(
                lambda frame_idx, detection_rate, left_coords, right_coords:
                self._on_detection_updated(frame_idx, detection_rate, left_coords, right_coords)
            )
            
            # Connect ROI update signal
            controller.roi_updated.connect(self._on_roi_updated)
            
            # We no longer connect to tracking_updated from BallTrackingController
            # This is now handled by the GameAnalyzer connection
            
            # Connect prediction update signal for Kalman state
            controller.prediction_updated.connect(self._on_kalman_predicted)
            
            # Set up visualizers with controller
            self._setup_visualizers_with_controller(controller)
            
            logging.info("InfoView connected to ball tracking controller")
    
    def connect_game_analyzer(self, analyzer):
        """
        Connect to a game analyzer to receive court position updates.
        
        Args:
            analyzer: GameAnalyzer instance
        """
        if analyzer:
            self.game_analyzer = analyzer
            
            # Connect to court position updates (x, y, z in court coordinate system)
            analyzer.court_position_updated.connect(self._on_court_position_updated)
            
            # Connect to in/out signal for the LED indicator
            analyzer.in_out_detected.connect(self.in_out_led.on_in_out)
            
            logging.info("InfoView connected to game analyzer")
    
    def _setup_visualizers_with_controller(self, controller):
        """
        Set up visualizers with the connected controller.
        
        Args:
            controller: BallTrackingController instance
        """
        # Create visualizers with the controller
        self._visualizers = [
            HSVMaskVisualizer(controller),
            ROIMaskVisualizer(controller),
            HoughCircleVisualizer(controller),
            KalmanPathVisualizer(controller)
        ]
        
        logging.info(f"Set up {len(self._visualizers)} visualizers")
    
    def _on_detection_updated(self, frame_idx, detection_rate, left_coords, right_coords, position_coords=None):
        """
        Handle detection update signal from ball tracking controller.
        
        Args:
            frame_idx (int): Frame index
            detection_rate (float): Detection rate between 0-1
            left_coords (tuple): Coordinates in left image (x, y, r) or None
            right_coords (tuple): Coordinates in right image (x, y, r) or None
            position_coords (tuple, optional): 3D position coordinates (x, y, z)
        """
        # Update detection rate
        self.set_detection_rate(detection_rate)
        
        # Safety check for left_coords - must be tuple-like (list, tuple, array)
        if left_coords and isinstance(left_coords, (tuple, list, np.ndarray)):
            try:
                self.set_left_pixel_coords(left_coords[0], left_coords[1], left_coords[2] if len(left_coords) > 2 else 0)
            except (IndexError, TypeError) as e:
                logging.error(f"Error processing left_coords {left_coords}: {e}")
                self.set_left_pixel_coords(0, 0, 0)
        else:
            self.set_left_pixel_coords(0, 0, 0)
            
        # Safety check for right_coords - must be tuple-like
        if right_coords and isinstance(right_coords, (tuple, list, np.ndarray)):
            try:
                self.set_right_pixel_coords(right_coords[0], right_coords[1], right_coords[2] if len(right_coords) > 2 else 0)
            except (IndexError, TypeError) as e:
                logging.error(f"Error processing right_coords {right_coords}: {e}")
                self.set_right_pixel_coords(0, 0, 0)
        else:
            self.set_right_pixel_coords(0, 0, 0)
        
        # We don't update 3D position from BallTrackingController anymore
        # This is now handled by the GameAnalyzer's court_position_updated signal
        
        # Log with proper type information
        left_type = type(left_coords).__name__ if left_coords is not None else "None"
        right_type = type(right_coords).__name__ if right_coords is not None else "None"
        logging.debug(f"Info view updated: frame={frame_idx}, rate={detection_rate:.2f}, left=({left_type}){left_coords}, right=({right_type}){right_coords}")
    
    def _on_roi_updated(self, left_roi, right_roi):
        """
        Handle ROI update signal from ball tracking controller.
        
        Args:
            left_roi (dict): Left camera ROI dictionary
            right_roi (dict): Right camera ROI dictionary
        """
        if left_roi:
            self.set_left_roi(left_roi.get('x', 0), left_roi.get('y', 0), 
                              left_roi.get('width', 0), left_roi.get('height', 0))
        else:
            self.set_left_roi(0, 0, 0, 0)
            
        if right_roi:
            self.set_right_roi(right_roi.get('x', 0), right_roi.get('y', 0), 
                               right_roi.get('width', 0), right_roi.get('height', 0))
        else:
            self.set_right_roi(0, 0, 0, 0)
            
        logging.debug(f"ROI info updated: left={left_roi}, right={right_roi}")
    
    def _on_court_position_updated(self, x, y, z):
        """
        Handle court position update signal from game analyzer.
        
        Args:
            x (float): X coordinate in court space
            y (float): Y coordinate in court space
            z (float): Z coordinate in court space
        """
        self.set_position_coords(x, y, z)
        logging.debug(f"Court position updated: ({x:.3f}, {y:.3f}, {z:.3f})")
    
    def _on_kalman_predicted(self, camera, x, y, vx, vy):
        """
        Handle Kalman prediction update signal from ball tracking controller.
        
        Args:
            camera (str): Camera identifier ('left' or 'right')
            x (float): Position X coordinate
            y (float): Position Y coordinate
            vx (float): Velocity X component
            vy (float): Velocity Y component
        """
        # Update the appropriate state label based on camera
        if camera.lower() == "left":
            self.set_left_state(x, y, vx, vy)
        elif camera.lower() == "right":
            self.set_right_state(x, y, vx, vy)
            
        logging.debug(f"Kalman state updated for {camera} camera: pos=({x:.1f}, {y:.1f}), vel=({vx:.1f}, {vy:.1f})")
    
    def get_visualizers(self):
        """
        Get the list of visualizers.
        
        Returns:
            list: List of visualizer objects
        """
        return self._visualizers 