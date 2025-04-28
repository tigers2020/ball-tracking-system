#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tracking Coordinates Controller module.
This module contains the TrackingCoordinatesController class that connects BallTrackingController
with CoordinateCombiner and provides tracking data for the TrackingOverlay UI.
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple

from PySide6.QtCore import QObject, Signal, Slot, QTimer

from src.controllers.ball_tracking_controller import BallTrackingController, TrackingState
from src.services.coordinate_combiner import CoordinateCombiner
from src.utils.constants import TRACKING_OVERLAY


class TrackingCoordinatesController(QObject):
    """
    Controller for tracking coordinates display.
    Connects ball tracking results with the tracking overlay UI.
    Coordinates data collection, combination, and transmission to the UI.
    """
    
    # Signal for updating tracking information (dict contains all necessary data)
    tracking_info_updated = Signal(dict)
    
    def __init__(self, ball_tracking_controller: BallTrackingController, config_manager):
        """
        Initialize the tracking coordinates controller.
        
        Args:
            ball_tracking_controller: BallTrackingController instance
            config_manager: ConfigManager instance
        """
        super(TrackingCoordinatesController, self).__init__()
        
        # Store references
        self.ball_tracker = ball_tracking_controller
        self.config_manager = config_manager
        
        # Create coordinate combiner service
        self.coordinate_combiner = CoordinateCombiner(config_manager)
        
        # Initialize tracking data
        self.left_hsv_point = None
        self.left_hough_point = None
        self.left_kalman_point = None
        self.right_hsv_point = None
        self.right_hough_point = None
        self.right_kalman_point = None
        self.last_frame_idx = 0
        self.last_update_time = time.time()
        self.processing_times = []  # Store last N processing times for average
        
        # Create update timer
        self.update_timer = QTimer()
        self.update_timer.setInterval(TRACKING_OVERLAY.UPDATE_INTERVAL_MS)
        self.update_timer.timeout.connect(self._update_tracking_info)
        
        # Connect to ball tracking controller signals
        self._connect_signals()
    
    def _connect_signals(self):
        """Connect to ball tracking controller signals."""
        # Connect to tracking state changes
        if hasattr(self.ball_tracker, 'tracking_state_changed'):
            self.ball_tracker.tracking_state_changed.connect(self._on_tracking_state_changed)
        
        # Connect to detection updates (if available)
        if hasattr(self.ball_tracker, 'detection_updated'):
            self.ball_tracker.detection_updated.connect(self._on_detection_updated)
        
        # Connect to tracking updates (original simple x,y,z signal)
        if hasattr(self.ball_tracker, 'tracking_updated'):
            self.ball_tracker.tracking_updated.connect(self._on_tracking_updated)
    
    def start_updates(self):
        """Start periodic updates of tracking information."""
        self.update_timer.start()
        logging.info("Started tracking coordinates updates")
    
    def stop_updates(self):
        """Stop periodic updates of tracking information."""
        self.update_timer.stop()
        logging.info("Stopped tracking coordinates updates")
    
    def set_update_interval(self, interval_ms):
        """
        Set the update interval for tracking info.
        
        Args:
            interval_ms (int): Update interval in milliseconds
        """
        self.update_timer.setInterval(interval_ms)
    
    @Slot(object)
    def _on_tracking_state_changed(self, state):
        """
        Handle tracking state changes.
        
        Args:
            state (TrackingState): New tracking state
        """
        # Process tracking state
        status = "Unknown"
        
        if state == TrackingState.TRACKING:
            status = "Tracking"
        elif state == TrackingState.TRACKING_LOST:
            status = "Lost"
        elif state == TrackingState.RESET:
            status = "Reset"
            # Reset coordinate data
            self._reset_coordinates()
        elif state == TrackingState.DISABLED:
            status = "Disabled"
        
        # Force an immediate update
        self._update_tracking_info(status=status)
    
    @Slot(int, float, tuple, tuple, tuple)
    def _on_detection_updated(self, frame_idx, detection_rate, left_coords, right_coords, position_coords):
        """
        Handle detection updates from ball tracking controller.
        
        Args:
            frame_idx (int): Frame index
            detection_rate (float): Detection rate
            left_coords (tuple): Left camera coordinates (x, y)
            right_coords (tuple): Right camera coordinates (x, y)
            position_coords (tuple): 3D position coordinates (x, y, z)
        """
        # Store current frame index
        self.last_frame_idx = frame_idx
        
        # Store coordinates (could be from HSV or Hough, we'll use as Hough for now)
        if left_coords and left_coords[0] is not None and left_coords[1] is not None:
            self.left_hough_point = left_coords
        
        if right_coords and right_coords[0] is not None and right_coords[1] is not None:
            self.right_hough_point = right_coords
    
    @Slot(float, float, float)
    def _on_tracking_updated(self, x, y, z):
        """
        Handle tracking updates from ball tracking controller.
        
        Args:
            x (float): X coordinate in 3D space
            y (float): Y coordinate in 3D space
            z (float): Z coordinate in 3D space
        """
        # We'll handle this in _update_tracking_info by getting the latest coordinates
        pass
    
    def _reset_coordinates(self):
        """Reset all coordinate data."""
        self.left_hsv_point = None
        self.left_hough_point = None
        self.left_kalman_point = None
        self.right_hsv_point = None
        self.right_hough_point = None
        self.right_kalman_point = None
    
    def _get_latest_coordinates(self):
        """
        Get the latest coordinates from the ball tracking controller.
        
        Returns:
            dict: Dictionary with latest coordinate data
        """
        # Try to get coordinates from ball tracking controller
        try:
            if hasattr(self.ball_tracker, 'get_latest_coordinates'):
                coords = self.ball_tracker.get_latest_coordinates()
                
                # Update stored points based on available data
                if 'left_coords' in coords and coords['left_coords'] is not None:
                    self.left_hough_point = coords['left_coords']
                
                if 'right_coords' in coords and coords['right_coords'] is not None:
                    self.right_hough_point = coords['right_coords']
                
                if 'position' in coords and coords['position'] is not None:
                    # This is already a 3D point, we'll use it directly
                    world_pos = coords['position']
                else:
                    world_pos = None
                
                # Get Kalman predictions if available
                if hasattr(self.ball_tracker, 'get_predictions'):
                    predictions = self.ball_tracker.get_predictions()
                    if predictions:
                        if 'left' in predictions and predictions['left'] is not None:
                            self.left_kalman_point = (predictions['left']['x'], predictions['left']['y'])
                        
                        if 'right' in predictions and predictions['right'] is not None:
                            self.right_kalman_point = (predictions['right']['x'], predictions['right']['y'])
                
                return {
                    'left_coords': self.left_hough_point,
                    'right_coords': self.right_hough_point,
                    'position': world_pos,
                    'left_kalman': self.left_kalman_point,
                    'right_kalman': self.right_kalman_point
                }
            
            return {}
                
        except Exception as e:
            logging.error(f"Error getting latest coordinates: {e}")
            return {}
    
    def _calculate_processing_time(self):
        """
        Calculate processing time since last update.
        
        Returns:
            float: Processing time in milliseconds
        """
        current_time = time.time()
        processing_time = (current_time - self.last_update_time) * 1000  # Convert to ms
        
        # Store time for next calculation
        self.last_update_time = current_time
        
        # Add to processing times list and keep last 10
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 10:
            self.processing_times.pop(0)
        
        # Return average processing time
        return sum(self.processing_times) / len(self.processing_times)
    
    def _get_tracking_status(self):
        """
        Get the current tracking status.
        
        Returns:
            tuple: (status, confidence)
        """
        # Default values
        status = "No tracking"
        confidence = 0.0
        
        try:
            # Get tracking state if available
            if hasattr(self.ball_tracker, 'tracking_state'):
                state = self.ball_tracker.tracking_state
                
                if state == TrackingState.TRACKING:
                    status = "Tracking"
                    confidence = 1.0
                elif state == TrackingState.TRACKING_LOST:
                    status = "Lost"
                    confidence = 0.3
                elif state == TrackingState.RESET:
                    status = "Reset"
                    confidence = 0.0
                elif state == TrackingState.DISABLED:
                    status = "Disabled"
                    confidence = 0.0
            
            # Get detection rate if available
            if hasattr(self.ball_tracker, 'get_detection_rate'):
                detection_rate = self.ball_tracker.get_detection_rate()
                if detection_rate is not None:
                    confidence = detection_rate
        
        except Exception as e:
            logging.error(f"Error getting tracking status: {e}")
            status = "Error"
            confidence = 0.0
        
        return (status, confidence)
    
    @Slot(str)
    def _update_tracking_info(self, status=None):
        """
        Update tracking information and emit signal.
        
        Args:
            status (str, optional): Override status message
        """
        # Get latest coordinates
        coords = self._get_latest_coordinates()
        
        # Calculate processing time
        processing_time = self._calculate_processing_time()
        
        # Get tracking status
        if status is None:
            status, confidence = self._get_tracking_status()
        else:
            # Use provided status with default confidence
            confidence = 0.5
        
        # Combine coordinates using CoordinateCombiner
        combined_data = self.coordinate_combiner.combine_and_triangulate(
            left_hough=self.left_hough_point,
            right_hough=self.right_hough_point,
            left_kalman=self.left_kalman_point,
            right_kalman=self.right_kalman_point
        )
        
        # If we already have a 3D position from the controller, use it
        if 'position' in coords and coords['position'] is not None:
            combined_data['world_3d'] = coords['position']
        
        # Create tracking info dictionary
        tracking_info = {
            'frame_idx': self.last_frame_idx,
            'left_2d': combined_data['left_2d'],
            'right_2d': combined_data['right_2d'],
            'world_3d': combined_data['world_3d'],
            'processing_time': processing_time,
            'status': status,
            'confidence': confidence
        }
        
        # Emit signal with tracking info
        self.tracking_info_updated.emit(tracking_info)
    
    def connect_to_view(self, tracking_overlay):
        """
        Connect to tracking overlay view.
        
        Args:
            tracking_overlay: TrackingOverlay widget instance
        """
        self.tracking_info_updated.connect(tracking_overlay.update_tracking_info)
        logging.info("Connected tracking overlay to coordinate controller") 