#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ball Tracking Controller module.
This module contains the BallTrackingController class for handling ball tracking functionality.
"""

import logging
import cv2
import numpy as np
import json
import os
import time
import xml.etree.ElementTree as ET
import atexit
from pathlib import Path
from PySide6.QtCore import QObject, Signal, Slot
from typing import Dict, List, Tuple, Optional, Any, Union

from src.models.tracking_data_model import TrackingDataModel
from src.services.hsv_mask_generator import HSVMaskGenerator
from src.services.roi_computer import ROIComputer
from src.services.circle_detector import CircleDetector
from src.services.kalman_processor import KalmanProcessor
from src.services.data_saver import DataSaver
from src.services.triangulation_service import TriangulationService
from src.utils.config_manager import ConfigManager
from src.utils.coord_utils import fuse_coordinates


class BallTrackingController(QObject):
    """
    Controller class for ball tracking functionality.
    Handles orchestration of various services for ball detection and tracking.
    """
    
    # Signals
    mask_updated = Signal(np.ndarray, np.ndarray)  # left_mask, right_mask
    roi_updated = Signal(dict, dict)  # left_roi, right_roi
    detection_updated = Signal(float, tuple, tuple)  # detection_rate, left_coords, right_coords
    circles_processed = Signal(np.ndarray, np.ndarray)  # left_circle_image, right_circle_image
    tracking_updated = Signal(float, float, float)  # x, y, z
    
    def __init__(self, model: TrackingDataModel, config_manager: ConfigManager):
        """
        Initialize BallTrackingController.
        
        Args:
            model: Data model
            config_manager: Configuration manager
        """
        super().__init__()
        
        self.model = model
        self.model.is_enabled = False
        self.config_manager = config_manager
        
        # Load settings from configuration
        self.model.hsv_values = self.config_manager.get_hsv_settings()
        self.model.roi_settings = self.config_manager.get_roi_settings()
        self.model.hough_settings = self.config_manager.get_hough_circle_settings()
        
        # Create services with proper configuration
        self.hsv_mask_generator = HSVMaskGenerator(self.model.hsv_values)
        self.roi_computer = ROIComputer(self.model.roi_settings)
        self.circle_detector = CircleDetector(self.model.hough_settings)
        
        # Get Kalman settings and initialize processor
        self.kalman_settings = self.config_manager.get_kalman_settings()
        self.kalman_processor = KalmanProcessor(self.kalman_settings)
        
        # Initialize triangulation service with camera settings
        self.camera_settings = self.config_manager.get_camera_settings()
        self.triangulator = TriangulationService(self.camera_settings)
        
        self.data_saver = DataSaver()
        
        # Register finalize_xml with atexit to ensure XML is properly closed at exit
        atexit.register(self.data_saver.finalize_xml)
        
        # Timestamp tracking for Kalman filter dt calculation
        self.last_update_time = {"left": None, "right": None}
        
        # Tracking enabled flag
        self._enabled = False
        
        # Flags and state variables
        self.left_image = None
        self.right_image = None
        self._detection_counter = 0
    
    @property
    def is_enabled(self):
        """
        Get the enabled state of ball tracking.
        
        Returns:
            bool: True if ball tracking is enabled, False otherwise
        """
        return self.model.is_enabled
    
    @property
    def detection_stats(self):
        """
        Get detection statistics.
        
        Returns:
            dict: Detection statistics including tracking status and counters
        """
        return self.model.detection_stats
    
    @property
    def xml_root(self):
        """
        Get XML root element from data saver.
        
        Returns:
            Element: XML root element or None if not initialized
        """
        return self.data_saver.xml_root
    
    @property
    def left_image(self):
        """
        Get the current left image.
        
        Returns:
            numpy.ndarray: Left image or None if not available
        """
        return self.model.left_image
    
    @property
    def right_image(self):
        """
        Get the current right image.
        
        Returns:
            numpy.ndarray: Right image or None if not available
        """
        return self.model.right_image
    
    def set_hsv_values(self, hsv_values):
        """
        Set HSV threshold values for ball detection.
        
        Args:
            hsv_values (dict): Dictionary containing HSV min/max values
        """
        # Update model and configuration
        self.model.set_hsv_values(hsv_values)
        self.config_manager.set_hsv_settings(hsv_values)
        
        # Update service
        self.hsv_mask_generator.update_hsv_values(hsv_values)
        
        logging.info(f"HSV values updated and saved: {hsv_values}")
        
        # Apply updated HSV values if enabled
        if self.model.is_enabled and (self.model.left_image is not None or self.model.right_image is not None):
            self._process_images()
    
    def set_images(self, left_image, right_image):
        """
        Set the current stereo images for processing.
        
        Args:
            left_image (numpy.ndarray): Left OpenCV image
            right_image (numpy.ndarray): Right OpenCV image
        """
        # Update model
        self.model.set_images(left_image, right_image)
        
        # Process images if enabled
        if self.model.is_enabled:
            self._process_images()
    
    def enable(self, enabled=True):
        """
        Enable or disable ball tracking.
        
        Args:
            enabled (bool): True to enable, False to disable
        """
        if self.model.is_enabled != enabled:
            self.model.is_enabled = enabled
            logging.info(f"Ball tracking {'enabled' if enabled else 'disabled'}")
            
            if enabled and (self.model.left_image is not None or self.model.right_image is not None):
                self._process_images()
            else:
                # Clear masks and emit signal
                self.model.left_mask = None
                self.model.right_mask = None
                self.mask_updated.emit(None, None)
    
    def _process_images(self):
        """Process the current images to generate HSV masks, ROIs, and detect circles."""
        # Create masks
        left_result = self.hsv_mask_generator.generate_mask(self.model.left_image) if self.model.left_image is not None else None
        right_result = self.hsv_mask_generator.generate_mask(self.model.right_image) if self.model.right_image is not None else None
        
        # Extract binary masks from results (original_img, masked_img, binary_mask, centroid, mask_too_narrow)
        left_mask = None
        right_mask = None
        left_mask_too_narrow = False
        right_mask_too_narrow = False
        
        if left_result is not None:
            if len(left_result) >= 5:
                _, _, left_mask, left_centroid, left_mask_too_narrow = left_result
        
        if right_result is not None:
            if len(right_result) >= 5:
                _, _, right_mask, right_centroid, right_mask_too_narrow = right_result
        
        # Store mask quality info in model for use in detection stats calculation
        self.model.left_mask_too_narrow = left_mask_too_narrow
        self.model.right_mask_too_narrow = right_mask_too_narrow
        
        # Update model with masks
        self.model.set_masks(left_mask, right_mask)
        
        # Calculate ROIs if enabled
        left_roi = None
        right_roi = None
        
        if self.model.roi_settings.get("enabled", False):
            left_roi = self.roi_computer.compute_roi(left_mask, self.model.left_image) if left_mask is not None else None
            right_roi = self.roi_computer.compute_roi(right_mask, self.model.right_image) if right_mask is not None else None
            
            # Update model with ROIs
            self.model.set_rois(left_roi, right_roi)
            
            # Emit ROI signal
            self.roi_updated.emit(left_roi, right_roi)
        else:
            # Clear ROIs
            self.model.set_rois(None, None)
            self.roi_updated.emit(None, None)
        
        # Emit signal with masks
        self.mask_updated.emit(left_mask, right_mask)
        
        # Crop ROI images
        left_cropped = self.roi_computer.crop_roi_image(self.model.left_image, left_roi) if left_roi is not None else None
        right_cropped = self.roi_computer.crop_roi_image(self.model.right_image, right_roi) if right_roi is not None else None
        
        # Store cropped images in model
        self.model.cropped_images["left"] = left_cropped
        self.model.cropped_images["right"] = right_cropped
        
        # Detect circles in the current ROIs
        self._detect_circles()
        
        # Update detection stats
        self.model.update_detection_stats()
        
        # Update detection rate signal
        self._update_detection_signal()
    
    def _detect_circles(self) -> bool:
        """
        Detect circles in the masked images within ROIs.
        
        Returns:
            bool: True if circles were detected, False otherwise
        """
        if self.model.left_mask is None or self.model.right_mask is None:
            return False
        
        # Get ROIs
        left_roi = self.model.left_roi
        right_roi = self.model.right_roi
        
        # Apply ROIs to mask if enabled and ROIs exist
        left_roi_enabled = self.model.roi_settings.get("enabled", False)
        
        if left_roi_enabled:
            # Apply ROIs to masks
            if left_roi:
                left_mask_roi = self.roi_computer.apply_roi_to_mask(self.model.left_mask, left_roi)
            else:
                left_mask_roi = self.model.left_mask
                
            if right_roi:
                right_mask_roi = self.roi_computer.apply_roi_to_mask(self.model.right_mask, right_roi)
            else:
                right_mask_roi = self.model.right_mask
        else:
            # Use full masks
            left_mask_roi = self.model.left_mask
            right_mask_roi = self.model.right_mask
        
        # Detect circles using Hough transform
        left_circles = self.circle_detector.detect_circles(
            self.model.left_image, 
            left_mask_roi, 
            roi=left_roi if left_roi_enabled else None
        )
        
        right_circles = self.circle_detector.detect_circles(
            self.model.right_image, 
            right_mask_roi, 
            roi=right_roi if left_roi_enabled else None
        )
        
        # Store the detected circles in the model
        self.model.set_circles(left_circles, right_circles)
        
        # Get the best circles from left and right images
        best_left_circle = self._get_best_circle(left_circles) if left_circles is not None else None
        best_right_circle = self._get_best_circle(right_circles) if right_circles is not None else None
        
        # Apply Kalman filtering if circles were detected
        left_prediction = None
        right_prediction = None
        
        # Update Kalman filter with left detection
        if best_left_circle is not None:
            x, y, r = best_left_circle
            self.model.add_coordinate("left", x, y, r)
            left_prediction = self.kalman_processor.update("left", x, y)
        else:
            # If no circle is detected, use prediction only
            left_prediction = self.kalman_processor.get_prediction("left")
        
        # Update Kalman filter with right detection
        if best_right_circle is not None:
            x, y, r = best_right_circle
            self.model.add_coordinate("right", x, y, r)
            right_prediction = self.kalman_processor.update("right", x, y)
        else:
            # If no circle is detected, use prediction only
            right_prediction = self.kalman_processor.get_prediction("right")
        
        # Store the predictions in the model
        self.model.set_predictions(left_prediction, right_prediction)
        
        # Fuse the coordinates from both cameras
        fused_coords = self._fuse_coordinates()
        
        # Use triangulation service to get 3D world point if we have fused coordinates
        if fused_coords and hasattr(self, 'triangulator') and self.triangulator is not None:
            uL, vL, uR, vR = fused_coords
            world_point = self.triangulator.triangulate(uL, vL, uR, vR)
            
            # Add 3D point to model if valid
            if world_point is not None:
                X, Y, Z = world_point
                self.model.add_3d_point(X, Y, Z)
                
                # Emit signal to update UI with 3D position
                self.tracking_updated.emit(X, Y, Z)
                
                logging.debug(f"3D world point: ({X:.3f}, {Y:.3f}, {Z:.3f}) m")
        
        # Check if there was a detection
        if best_left_circle is not None or best_right_circle is not None:
            self._detection_counter += 1
            return True
        else:
            return False
    
    def _get_best_circle(self, circles):
        """
        Get the best circle from a list of detected circles.
        
        Args:
            circles: List of detected circles
            
        Returns:
            tuple: Best circle (x, y, r) or None if no circles are detected
        """
        if circles:
            return circles[0]
        else:
            return None
    
    def _fuse_coordinates(self):
        """
        Fuse coordinates from both left and right images.
        
        Returns:
            tuple: Fused coordinates (uL, vL, uR, vR) or None if no circles are detected
        """
        if self.model.left_circles and self.model.right_circles:
            left_circle = self.model.left_circles[0]
            right_circle = self.model.right_circles[0]
            return (left_circle[0], left_circle[1], right_circle[0], right_circle[1])
        else:
            return None
    
    def _check_out_of_bounds(self):
        """Check if ball is predicted to be out of bounds and update tracking state."""
        if not self.model.detection_stats["is_tracking"]:
            return
            
        left_pred = self.kalman_processor.get_prediction("left")
        right_pred = self.kalman_processor.get_prediction("right")
        
        if left_pred is None and right_pred is None:
            return
            
        # Define image boundaries
        left_bounds = (0, 0, self.model.left_image.shape[1], self.model.left_image.shape[0]) if self.model.left_image is not None else None
        right_bounds = (0, 0, self.model.right_image.shape[1], self.model.right_image.shape[0]) if self.model.right_image is not None else None
        
        is_out_of_bounds = False
        
        # Add margin to avoid false positives (15% of width/height)
        margin_x = 0
        margin_y = 0
        
        if left_bounds:
            margin_x = int(left_bounds[2] * 0.15)
            margin_y = int(left_bounds[3] * 0.15)
        elif right_bounds:
            margin_x = int(right_bounds[2] * 0.15)
            margin_y = int(right_bounds[3] * 0.15)
        
        # Check left prediction
        left_out = False
        if left_pred is not None and left_bounds is not None:
            px, py, vx, vy = left_pred
            
            # Log current prediction and bounds for debugging
            logging.debug(f"Left prediction: pos=({px}, {py}), vel=({vx}, {vy})")
            
            # Add margin to bounds for checking
            effective_bounds = (-margin_x, -margin_y, left_bounds[2] + margin_x, left_bounds[3] + margin_y)
            
            # Check if current position is outside bounds
            current_out = (px < effective_bounds[0] or px >= effective_bounds[2] or 
                          py < effective_bounds[1] or py >= effective_bounds[3])
                
            # Only consider rapid motion for out-of-bounds prediction
            if abs(vx) > 2.0 or abs(vy) > 2.0:
                future_steps = 3  # Look 3 frames ahead (more conservative)
                future_x = px + vx * future_steps
                future_y = py + vy * future_steps
                
                # Check if future position is outside bounds
                future_out = (future_x < -3*margin_x or future_x >= left_bounds[2] + 3*margin_x or 
                             future_y < -3*margin_y or future_y >= left_bounds[3] + 3*margin_y)
                
                # Consider out of bounds only if both current and future positions are out
                if current_out and future_out:
                    logging.debug(f"Left prediction significantly out of bounds: current=({px}, {py}), future=({future_x}, {future_y})")
                    left_out = True
            
        # Check right prediction (similar to left)
        right_out = False
        if right_pred is not None and right_bounds is not None:
            px, py, vx, vy = right_pred
            
            # Log current prediction and bounds for debugging
            logging.debug(f"Right prediction: pos=({px}, {py}), vel=({vx}, {vy})")
            
            # Add margin to bounds for checking
            effective_bounds = (-margin_x, -margin_y, right_bounds[2] + margin_x, right_bounds[3] + margin_y)
            
            # Check if current position is outside bounds
            current_out = (px < effective_bounds[0] or px >= effective_bounds[2] or 
                          py < effective_bounds[1] or py >= effective_bounds[3])
                
            # Only consider rapid motion for out-of-bounds prediction
            if abs(vx) > 2.0 or abs(vy) > 2.0:
                future_steps = 3  # Look 3 frames ahead (more conservative)
                future_x = px + vx * future_steps
                future_y = py + vy * future_steps
                
                # Check if future position is outside bounds
                future_out = (future_x < -3*margin_x or future_x >= right_bounds[2] + 3*margin_x or 
                             future_y < -3*margin_y or future_y >= right_bounds[3] + 3*margin_y)
                
                # Consider out of bounds only if both current and future positions are out
                if current_out and future_out:
                    logging.debug(f"Right prediction significantly out of bounds: current=({px}, {py}), future=({future_x}, {future_y})")
                    right_out = True
        
        # Use a counter for consecutive out-of-bounds detections
        if not hasattr(self, '_out_of_bounds_counter'):
            self._out_of_bounds_counter = 0
        
        # Consider out of bounds only if both sides are out or a single active side is out
        is_left_active = left_pred is not None and self.kalman_processor.is_filter_ready("left")
        is_right_active = right_pred is not None and self.kalman_processor.is_filter_ready("right")
        
        if ((is_left_active and is_right_active and left_out and right_out) or   # Both sides active and out
            (is_left_active and not is_right_active and left_out) or             # Only left active and out
            (is_right_active and not is_left_active and right_out)):             # Only right active and out
            self._out_of_bounds_counter += 1
            logging.debug(f"Out of bounds counter: {self._out_of_bounds_counter}")
        else:
            # Reset counter if ball is within bounds
            self._out_of_bounds_counter = 0
        
        # Only stop tracking after multiple consecutive out-of-bounds detections
        if self._out_of_bounds_counter >= 3:  # Adjust threshold as needed
            logging.info("Ball predicted out of bounds for multiple frames, stopping tracking")
            self.model.detection_stats["is_tracking"] = False
            self._out_of_bounds_counter = 0  # Reset counter
    
    def _update_detection_signal(self):
        """Emit detection update signal with current state."""
        detection_rate = self.model.get_detection_rate()
        left_coords, right_coords = self.model.get_latest_coordinates()
        self.detection_updated.emit(detection_rate, left_coords, right_coords)
    
    def detect_circles_in_roi(self):
        """
        Manually detect circles in the current ROIs and update the model.
        This is a public wrapper for the internal _detect_circles method.
        
        Returns:
            tuple: (left_processed_image, right_processed_image)
        """
        if self.model.is_enabled and (self.model.left_image is not None or self.model.right_image is not None):
            self._detect_circles()
            
            # Return processed images with circles
            # These are normally emitted via signals, but we return them here for direct access
            left_mask, right_mask = self.get_current_masks()
            
            # Create circle visualizations
            left_viz = None
            right_viz = None
            
            if self.model.left_image is not None and self.model.left_roi is not None:
                left_viz = self.model.left_image.copy()
                
                # Draw ROI if available
                if self.model.left_roi:
                    try:
                        x = int(self.model.left_roi.get("x", 0))
                        y = int(self.model.left_roi.get("y", 0))
                        w = int(self.model.left_roi.get("width", 100))
                        h = int(self.model.left_roi.get("height", 100))
                        cv2.rectangle(left_viz, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    except (ValueError, TypeError, KeyError) as e:
                        logging.error(f"Error drawing left ROI in visualization: {e}")
                
                # Draw circles if available
                if self.model.left_circles:
                    for circle in self.model.left_circles:
                        try:
                            x, y, r = circle
                            cv2.circle(left_viz, (int(x), int(y)), int(r), (0, 0, 255), 2)
                        except (ValueError, TypeError, IndexError) as e:
                            logging.error(f"Error drawing left circle in visualization: {e}")
            
            if self.model.right_image is not None and self.model.right_roi is not None:
                right_viz = self.model.right_image.copy()
                
                # Draw ROI if available
                if self.model.right_roi:
                    try:
                        x = int(self.model.right_roi.get("x", 0))
                        y = int(self.model.right_roi.get("y", 0))
                        w = int(self.model.right_roi.get("width", 100))
                        h = int(self.model.right_roi.get("height", 100))
                        cv2.rectangle(right_viz, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    except (ValueError, TypeError, KeyError) as e:
                        logging.error(f"Error drawing right ROI in visualization: {e}")
                
                # Draw circles if available
                if self.model.right_circles:
                    for circle in self.model.right_circles:
                        try:
                            x, y, r = circle
                            cv2.circle(right_viz, (int(x), int(y)), int(r), (0, 0, 255), 2)
                        except (ValueError, TypeError, IndexError) as e:
                            logging.error(f"Error drawing right circle in visualization: {e}")
            
            return left_viz, right_viz
        
        return None, None
    
    def get_predictions(self):
        """
        Get the current Kalman filter predictions.
        
        Returns:
            tuple: (left_prediction, right_prediction) where each is (x, y, vx, vy) or None
        """
        left_pred = self.kalman_processor.get_prediction("left")
        right_pred = self.kalman_processor.get_prediction("right")
        return left_pred, right_pred
    
    def get_current_masks(self):
        """
        Get the current masks.
            
        Returns:
            tuple: (left_mask, right_mask)
        """
        return self.model.left_mask, self.model.right_mask
    
    def get_hsv_values(self):
        """
        Get the current HSV values.
        
        Returns:
            dict: Current HSV values
        """
        return self.model.hsv_values
    
    def set_roi_settings(self, roi_settings):
        """
        Set ROI settings for ball tracking.
        
        Args:
            roi_settings (dict): Dictionary containing ROI settings
        """
        # Update model and configuration
        self.model.set_roi_settings(roi_settings)
        self.config_manager.set_roi_settings(roi_settings)
        
        # Update service
        self.roi_computer.update_roi_settings(roi_settings)
        
        logging.info(f"ROI settings updated and saved: {roi_settings}")
        
        # Reprocess images if enabled to update ROIs
        if self.model.is_enabled and (self.model.left_image is not None or self.model.right_image is not None):
            self._process_images()
    
    def get_roi_settings(self):
        """
        Get the current ROI settings.
        
        Returns:
            dict: Current ROI settings
        """
        return self.model.roi_settings
    
    def get_current_rois(self):
        """
        Get the current ROIs.
        
        Returns:
            tuple: (left_roi, right_roi)
        """
        return self.model.left_roi, self.model.right_roi
    
    def get_cropped_roi_images(self):
        """
        Get cropped images based on current ROIs.
            
        Returns:
            tuple: (left_cropped_image, right_cropped_image)
        """
        return self.model.cropped_images["left"], self.model.cropped_images["right"]
    
    def get_detection_rate(self):
        """
        Get the current detection rate.
        
        Returns:
            float: Detection rate (0.0 to 1.0) or None if not tracking
        """
        return self.model.get_detection_rate()
    
    def get_latest_coordinates(self):
        """
        Get the latest coordinates from both left and right images.
        
        Returns:
            tuple: (left_coords, right_coords) where each is (x, y, r) or None if not available
        """
        return self.model.get_latest_coordinates()
    
    def get_coordinate_history(self, side="both", count=None):
        """
        Get the coordinate history.
        
        Args:
            side (str): "left", "right" or "both" to indicate which side to return
            count (int, optional): Number of most recent coordinates to return. All if None.
            
        Returns:
            dict or list: Coordinate history for the specified side(s)
        """
        if side == "both":
            if count is None:
                return self.model.coordinate_history
            else:
                return {
                    "left": self.model.coordinate_history["left"][-count:] if self.model.coordinate_history["left"] else [],
                    "right": self.model.coordinate_history["right"][-count:] if self.model.coordinate_history["right"] else []
                }
        elif side in ["left", "right"]:
            if count is None:
                return self.model.coordinate_history[side]
            else:
                return self.model.coordinate_history[side][-count:] if self.model.coordinate_history[side] else []
        else:
            logging.error(f"Invalid side: {side}")
            return None

    def clear_coordinate_history(self):
        """Clear the coordinate history."""
        self.model.clear_coordinate_history()
    
    def reset_tracking(self):
        """
        Reset all tracking data and filters.
        """
        try:
            # 저장 큐 정리
            self.data_saver.cleanup()
            
            # Reset Kalman filters
            self.kalman_processor.reset()
            
            # Reset model data
            self.model.reset()
            
            # Reset timestamp tracking
            self.last_update_time = {"left": None, "right": None}
            
            logging.info("Ball tracking reset complete")
            
        except Exception as e:
            logging.error(f"Error resetting tracking: {e}")
            
        # Return current status
        return self.model.detection_stats

    def save_coordinate_history(self, filename):
        """
        Save the coordinate history to a JSON file.
        
        Args:
            filename (str): Path to the output JSON file
        """
        try:
            history = self.get_coordinate_history()
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'w') as f:
                json.dump(history, f, indent=2)
            logging.info(f"Coordinate history saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving coordinate history: {e}")
            
    def save_tracking_data_to_json(self, folder_path=None, filename=None):
        """
        Save tracking data (original coordinates and Kalman predictions) to a JSON file.
        
        Args:
            folder_path (str, optional): Path to the output folder. Default is 'tracking_data' in current directory.
            filename (str, optional): Base filename without extension. Default uses timestamp.
        
        Returns:
            str: Path to the saved file or None if failed
        """
        try:
            # Prepare data dictionary
            tracking_data = {
                "timestamp": time.time(),
                "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
                "detection_rate": self.model.get_detection_rate(),
                "tracking_active": self.model.detection_stats["is_tracking"],
                "frames_total": self.model.detection_stats["total_frames"],
                "detections_count": self.model.detection_stats["detection_count"],
                "coordinate_data": {
                    "left": {
                        "hsv_centers": [],
                        "hough_centers": [],
                        "kalman_predictions": [],
                        "fused_centers": []
                    },
                    "right": {
                        "hsv_centers": [],
                        "hough_centers": [],
                        "kalman_predictions": [],
                        "fused_centers": []
                    }
                },
                "coordinate_history": {
                    "left": [],
                    "right": []
                }
            }
            
            # Process left side
            if self.model.left_mask is not None:
                # Get HSV mask centroid
                left_hsv_center = self.roi_computer.compute_mask_centroid(self.model.left_mask)
                
                # Get latest Hough circle center if available
                left_hough_center = None
                if self.model.left_circles:
                    left_hough_center = (self.model.left_circles[0][0], self.model.left_circles[0][1])
                
                # Get Kalman prediction if available
                left_kalman_pred = self.kalman_processor.get_prediction("left")
                
                # Add to tracking data
                if left_hsv_center:
                    tracking_data["coordinate_data"]["left"]["hsv_centers"].append({
                        "x": float(left_hsv_center[0]),
                        "y": float(left_hsv_center[1])
                    })
                
                if left_hough_center:
                    tracking_data["coordinate_data"]["left"]["hough_centers"].append({
                        "x": float(left_hough_center[0]),
                        "y": float(left_hough_center[1])
                    })
                
                if left_kalman_pred:
                    tracking_data["coordinate_data"]["left"]["kalman_predictions"].append({
                        "x": float(left_kalman_pred[0]),
                        "y": float(left_kalman_pred[1]),
                        "vx": float(left_kalman_pred[2]),
                        "vy": float(left_kalman_pred[3])
                    })
            
            # Process right side (similar to left)
            if self.model.right_mask is not None:
                # Get HSV mask centroid
                right_hsv_center = self.roi_computer.compute_mask_centroid(self.model.right_mask)
                
                # Get latest Hough circle center if available
                right_hough_center = None
                if self.model.right_circles:
                    right_hough_center = (self.model.right_circles[0][0], self.model.right_circles[0][1])
                
                # Get Kalman prediction if available
                right_kalman_pred = self.kalman_processor.get_prediction("right")
                
                # Add to tracking data
                if right_hsv_center:
                    tracking_data["coordinate_data"]["right"]["hsv_centers"].append({
                        "x": float(right_hsv_center[0]),
                        "y": float(right_hsv_center[1])
                    })
                
                if right_hough_center:
                    tracking_data["coordinate_data"]["right"]["hough_centers"].append({
                        "x": float(right_hough_center[0]),
                        "y": float(right_hough_center[1])
                    })
                
                if right_kalman_pred:
                    tracking_data["coordinate_data"]["right"]["kalman_predictions"].append({
                        "x": float(right_kalman_pred[0]),
                        "y": float(right_kalman_pred[1]),
                        "vx": float(right_kalman_pred[2]),
                        "vy": float(right_kalman_pred[3])
                    })
            
            # Convert coordinate history to serializable format
            for side in ["left", "right"]:
                history = self.model.coordinate_history[side]
                for entry in history:
                    # Each entry is (x, y, r, timestamp)
                    tracking_data["coordinate_history"][side].append({
                        "x": float(entry[0]),
                        "y": float(entry[1]),
                        "radius": float(entry[2]),
                        "timestamp": entry[3]
                    })
            
            # Use data saver service to save the data
            return self.data_saver.save_json_summary(tracking_data, folder_path, filename)
            
        except Exception as e:
            logging.error(f"Error saving tracking data: {e}")
            return None
    
    def save_tracking_data_for_frame(self, frame_number, folder_path=None):
        """
        Save tracking data for a specific frame, using the frame number as part of the filename.
        This allows overwriting data for the same frame when processed again.
        
        Args:
            frame_number (int): Current frame number
            folder_path (str, optional): Path to the output folder
            
        Returns:
            str: Path to the saved file or None if failed
        """
        try:
            # Prepare data dictionary
            frame_data = {
                "frame_number": frame_number,
                "timestamp": time.time(),
                "tracking_active": self.model.detection_stats["is_tracking"],
                "left": {
                    "hsv_center": None,
                    "hough_center": None,
                    "kalman_prediction": None,
                    "fused_center": None
                },
                "right": {
                    "hsv_center": None,
                    "hough_center": None, 
                    "kalman_prediction": None,
                    "fused_center": None
                }
            }
            
            # Process left side
            if self.model.left_mask is not None:
                # Get HSV mask centroid
                left_hsv_center = self.roi_computer.compute_mask_centroid(self.model.left_mask)
                if left_hsv_center:
                    frame_data["left"]["hsv_center"] = {
                        "x": float(left_hsv_center[0]),
                        "y": float(left_hsv_center[1])
                    }
                
                # Get latest Hough circle center if available
                left_hough_center = None
                if self.model.left_circles:
                    left_hough_center = (float(self.model.left_circles[0][0]), float(self.model.left_circles[0][1]))
                    frame_data["left"]["hough_center"] = {
                        "x": left_hough_center[0],
                        "y": left_hough_center[1],
                        "radius": float(self.model.left_circles[0][2])
                    }
                
                # Get Kalman prediction if available
                left_kalman_pred = self.kalman_processor.get_prediction("left")
                if left_kalman_pred is not None:
                    left_kalman_pos = (float(left_kalman_pred[0]), float(left_kalman_pred[1]))
                    frame_data["left"]["kalman_prediction"] = {
                        "x": left_kalman_pos[0],
                        "y": left_kalman_pos[1],
                        "vx": float(left_kalman_pred[2]),
                        "vy": float(left_kalman_pred[3])
                    }
                
                # Calculate fused coordinates
                coords_to_fuse = []
                if left_hsv_center:
                    coords_to_fuse.append(left_hsv_center)
                if left_hough_center:
                    coords_to_fuse.append(left_hough_center)
                if left_kalman_pred:
                    coords_to_fuse.append((left_kalman_pred[0], left_kalman_pred[1]))
                
                if coords_to_fuse:
                    fused_coords = fuse_coordinates(coords_to_fuse)
                    if fused_coords:
                        frame_data["left"]["fused_center"] = {
                            "x": float(fused_coords[0]),
                            "y": float(fused_coords[1])
                        }
                        logging.debug(f"Left fused coordinates: {fused_coords}")
            
            # Process right side (similar to left)
            if self.model.right_mask is not None:
                # Get HSV mask centroid
                right_hsv_center = self.roi_computer.compute_mask_centroid(self.model.right_mask)
                if right_hsv_center:
                    frame_data["right"]["hsv_center"] = {
                        "x": float(right_hsv_center[0]),
                        "y": float(right_hsv_center[1])
                    }
                
                # Get latest Hough circle center if available
                right_hough_center = None
                if self.model.right_circles:
                    right_hough_center = (float(self.model.right_circles[0][0]), float(self.model.right_circles[0][1]))
                    frame_data["right"]["hough_center"] = {
                        "x": right_hough_center[0],
                        "y": right_hough_center[1],
                        "radius": float(self.model.right_circles[0][2])
                    }
                
                # Get Kalman prediction if available
                right_kalman_pred = self.kalman_processor.get_prediction("right")
                if right_kalman_pred is not None:
                    right_kalman_pos = (float(right_kalman_pred[0]), float(right_kalman_pred[1]))
                    frame_data["right"]["kalman_prediction"] = {
                        "x": right_kalman_pos[0],
                        "y": right_kalman_pos[1],
                        "vx": float(right_kalman_pred[2]),
                        "vy": float(right_kalman_pred[3])
                    }
                
                # Calculate fused coordinates
                coords_to_fuse = []
                if right_hsv_center:
                    coords_to_fuse.append(right_hsv_center)
                if right_hough_center:
                    coords_to_fuse.append(right_hough_center)
                if right_kalman_pred:
                    coords_to_fuse.append((right_kalman_pred[0], right_kalman_pred[1]))
                
                if coords_to_fuse:
                    fused_coords = fuse_coordinates(coords_to_fuse)
                    if fused_coords:
                        frame_data["right"]["fused_center"] = {
                            "x": float(fused_coords[0]),
                            "y": float(fused_coords[1])
                        }
                        logging.debug(f"Right fused coordinates: {fused_coords}")
            
            # Use data saver service to save the frame data
            return self.data_saver.save_frame_to_xml(frame_number, frame_data, None)
            
        except Exception as e:
            logging.error(f"Error saving frame tracking data: {e}")
            return None
    
    def initialize_xml_tracking(self, folder_name):
        """
        Initialize XML tracking with the given folder name.
        
        Args:
            folder_name: Name of the folder for tracking data
            
        Returns:
            bool: Success or failure
        """
        try:
            # Initialize data saver if not already done
            if not hasattr(self, 'data_saver') or self.data_saver is None:
                self.data_saver = DataSaver()
            
            # Create output folder path
            output_path = os.path.join(os.getcwd(), "tracking_data", folder_name)
            os.makedirs(output_path, exist_ok=True)
            
            # Initialize XML tracking in the data saver
            result = self.data_saver.initialize_xml_tracking(folder_name)
            
            if result:
                logging.info(f"XML tracking initialized for folder: {folder_name}")
            else:
                logging.warning(f"Failed to initialize XML tracking for folder: {folder_name}")
                
            return result
        except Exception as e:
            logging.error(f"Error initializing XML tracking: {e}")
            return False
            
    def append_frame_xml(self, frame_number, frame_name=None):
        """
        Append the current frame's tracking data to XML file.
        
        Args:
            frame_number: Current frame number
            frame_name: Optional frame filename
            
        Returns:
            Boolean success indicator
        """
        try:
            # Get the current frame data
            frame_data = self.get_frame_data_dict(frame_number)
            
            # Use the data saver to add to XML
            result = self.data_saver.save_frame_to_xml(frame_number, frame_data, frame_name)
            
            if result:
                logging.debug(f"Frame {frame_number} tracking data saved to XML")
            else:
                logging.warning(f"Failed to save frame {frame_number} to XML")
                
            return result
            
        except Exception as e:
            logging.error(f"Error saving frame to XML: {e}")
            return False
            
    def save_xml_tracking_data(self, folder_path=None):
        """
        Save the complete XML tracking data to a file.
        For the incremental logging approach, this creates a full snapshot
        of the in-memory tracking data.
        
        Args:
            folder_path: Optional path to the output folder
            
        Returns:
            Path to the saved file or None if failed
        """
        try:
            # Have the data saver save a snapshot of the current data
            xml_path = self.data_saver.save_xml_tracking_data(folder_path)
            
            if xml_path:
                logging.info(f"XML tracking data snapshot saved to {xml_path}")
            
            return xml_path
            
        except Exception as e:
            logging.error(f"Error saving XML tracking data: {e}")
            return None
    
    def get_frame_data_dict(self, frame_number):
        """
        Create a dictionary with frame tracking data.
        
        Args:
            frame_number: Current frame number
            
        Returns:
            Dictionary with frame data
        """
        data = {
            "tracking_active": self.is_enabled,
            "left": {},
            "right": {}
        }
        
        # Add left camera data
        if self.model.left_circles:
            best_left = self.model.left_circles[0]
            data["left"]["hough_center"] = {
                "x": best_left[0],
                "y": best_left[1],
                "radius": best_left[2]
            }
        
        # Add HSV center for left
        if hasattr(self, 'hsv_mask_generator') and self.hsv_mask_generator:
            left_hsv_center = self.roi_computer.compute_mask_centroid(self.model.left_mask)
            if left_hsv_center:
                data["left"]["hsv_center"] = {
                    "x": left_hsv_center[0],
                    "y": left_hsv_center[1]
                }
        
        # Add left Kalman prediction
        if self.model.left_prediction:
            data["left"]["kalman_prediction"] = {
                "x": self.model.left_prediction[0],
                "y": self.model.left_prediction[1],
                "vx": self.model.left_prediction[2],
                "vy": self.model.left_prediction[3]
            }
        
        # Add right camera data
        if self.model.right_circles:
            best_right = self.model.right_circles[0]
            data["right"]["hough_center"] = {
                "x": best_right[0],
                "y": best_right[1],
                "radius": best_right[2]
            }
        
        # Add HSV center for right
        if hasattr(self, 'hsv_mask_generator') and self.hsv_mask_generator:
            right_hsv_center = self.roi_computer.compute_mask_centroid(self.model.right_mask)
            if right_hsv_center:
                data["right"]["hsv_center"] = {
                    "x": right_hsv_center[0],
                    "y": right_hsv_center[1]
                }
        
        # Add right Kalman prediction
        if self.model.right_prediction:
            data["right"]["kalman_prediction"] = {
                "x": self.model.right_prediction[0],
                "y": self.model.right_prediction[1],
                "vx": self.model.right_prediction[2],
                "vy": self.model.right_prediction[3]
            }
        
        # Add latest 3D world point if available
        world_point = self.model.get_latest_3d_point()
        if world_point:
            data["world"] = world_point
            
        return data
    
    def save_frame_to_json(self, frame_number, folder_path=None):
        """
        Save the current frame's tracking data to JSON file.
        
        Args:
            frame_number: Current frame number
            folder_path: Optional path to the output folder
            
        Returns:
            Path to the saved file or None if failed
        """
        try:
            # Get the current frame data
            frame_data = self.get_frame_data_dict(frame_number)
            
            # Use the data saver to save as JSON
            result = self.data_saver.save_json_frame(frame_number, frame_data, folder_path)
            
            if result:
                logging.debug(f"Frame {frame_number} tracking data saved as JSON to {result}")
            else:
                logging.warning(f"Failed to save frame {frame_number} as JSON")
                
            return result
            
        except Exception as e:
            logging.error(f"Error saving frame to JSON: {e}")
            return None
    
    def process_frame(self, frame_index, frame=None):
        """
        Process a frame and log tracking data to XML.
        
        Args:
            frame_index: Index of the current frame
            frame: Frame object (optional)
        """
        # Process images if tracking is enabled and we have images
        if self.is_enabled and (self.left_image is not None and self.right_image is not None):
            # Always process images when this method is called
            self._process_images()
                
            # Log to XML if tracking data exists
            if hasattr(self, 'data_saver') and self.data_saver is not None:
                # Get frame data dictionary
                frame_data = self.get_frame_data_dict(frame_index)
                
                # Generate frame name
                frame_name = f"frame_{frame_index:06d}.png"
                
                # Log to XML
                result = self.append_frame_xml(frame_index, frame_name)
                if result:
                    logging.debug(f"XML: frame {frame_index} appended")
                else:
                    logging.warning(f"Failed to append frame {frame_index} to XML")
        
        return True

    def update_camera_settings(self, camera_settings: Dict[str, Any]) -> None:
        """
        Update camera settings for 3D triangulation.
        
        Args:
            camera_settings: Camera configuration parameters
        """
        self.camera_settings = camera_settings.copy()
        
        # Update triangulation service with new settings
        if hasattr(self, 'triangulator') and self.triangulator is not None:
            self.triangulator.set_camera(self.camera_settings)
            logging.info("Camera settings updated for triangulation")
            
        # Save settings to configuration
        self.config_manager.set_camera_settings(camera_settings) 