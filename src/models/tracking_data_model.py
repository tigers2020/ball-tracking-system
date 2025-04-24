#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tracking Data Model module.
This module contains the TrackingDataModel class for managing ball tracking data.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class TrackingDataModel:
    """
    Model class for ball tracking data.
    Stores and manages images, masks, ROIs, detected circles, and tracking statistics.
    """
    
    def __init__(self):
        """Initialize the tracking data model."""
        # Images
        self.left_image = None
        self.right_image = None
        
        # Masks
        self.left_mask = None
        self.right_mask = None
        
        # Mask quality indicators
        self.left_mask_too_narrow = False
        self.right_mask_too_narrow = False
        
        # ROIs
        self.left_roi = None
        self.right_roi = None
        
        # Tracking settings
        self.is_enabled = False
        self.hsv_values = {}
        self.roi_settings = {}
        self.hough_settings = {}
        
        # Tracking data
        self.left_circles = None
        self.right_circles = None
        self.left_prediction = None
        self.right_prediction = None
        
        # Cropped images
        self.cropped_images = {
            "left": None,
            "right": None
        }
        
        # Coordinate history
        self.coordinate_history = {
            "left": [],
            "right": []
        }
        
        # Detection statistics
        self.detection_stats = {
            "is_tracking": False,
            "frames_processed": 0,
            "frames_detected": 0,
            "detection_rate": 0.0,
            "lost_frames": 0
        }
        
        logging.info("Tracking data model initialized")
    
    def set_images(self, left_image: np.ndarray, right_image: np.ndarray) -> None:
        """
        Set the current stereo images.
        
        Args:
            left_image: Left OpenCV image
            right_image: Right OpenCV image
        """
        self.left_image = left_image
        self.right_image = right_image
    
    def set_masks(self, left_mask: np.ndarray, right_mask: np.ndarray) -> None:
        """
        Set the current HSV masks.
        
        Args:
            left_mask: Left binary mask
            right_mask: Right binary mask
        """
        self.left_mask = left_mask
        self.right_mask = right_mask
    
    def set_rois(self, left_roi: Optional[Dict[str, int]], right_roi: Optional[Dict[str, int]]) -> None:
        """
        Set the current ROIs.
        
        Args:
            left_roi: Left ROI dictionary
            right_roi: Right ROI dictionary
        """
        self.left_roi = left_roi
        self.right_roi = right_roi
    
    def set_hsv_values(self, hsv_values: Dict[str, Any]) -> None:
        """
        Set HSV threshold values.
        
        Args:
            hsv_values: Dictionary containing HSV threshold values
        """
        self.hsv_values = hsv_values.copy()
        logging.info(f"HSV values updated: {self.hsv_values}")
    
    def set_roi_settings(self, roi_settings: Dict[str, Any]) -> None:
        """
        Set ROI settings.
        
        Args:
            roi_settings: Dictionary containing ROI settings
        """
        self.roi_settings = roi_settings.copy()
        logging.info(f"ROI settings updated: {self.roi_settings}")
    
    def set_hough_settings(self, hough_settings: Dict[str, Any]) -> None:
        """
        Set Hough Circle detection settings.
        
        Args:
            hough_settings: Dictionary containing Hough Circle parameters
        """
        self.hough_settings = hough_settings.copy()
        logging.info(f"Hough settings updated: {self.hough_settings}")
    
    def set_circles(self, left_circles: Optional[List[Tuple[int, int, int]]], 
                   right_circles: Optional[List[Tuple[int, int, int]]]) -> None:
        """
        Set detected circles.
        
        Args:
            left_circles: List of circles (x, y, r) detected in left image
            right_circles: List of circles (x, y, r) detected in right image
        """
        self.left_circles = left_circles
        self.right_circles = right_circles
    
    def set_predictions(self, left_prediction: Optional[Tuple[float, float, float, float]], 
                        right_prediction: Optional[Tuple[float, float, float, float]]) -> None:
        """
        Set Kalman filter predictions.
        
        Args:
            left_prediction: (x, y, vx, vy) prediction for left image
            right_prediction: (x, y, vx, vy) prediction for right image
        """
        self.left_prediction = left_prediction
        self.right_prediction = right_prediction
    
    def add_coordinate(self, side: str, x: float, y: float, radius: float) -> None:
        """
        Add a detected coordinate to the history.
        
        Args:
            side: Which side the coordinate belongs to ("left" or "right")
            x: X coordinate
            y: Y coordinate
            radius: Circle radius
        """
        if side not in ["left", "right"]:
            logging.warning(f"Invalid side: {side}")
            return
        
        self.coordinate_history[side].append((x, y, radius))
        
        # Limit history size to prevent memory issues
        max_history = 1000  # Adjust as needed
        if len(self.coordinate_history[side]) > max_history:
            self.coordinate_history[side] = self.coordinate_history[side][-max_history:]
    
    def clear_coordinate_history(self) -> None:
        """Clear all coordinate history."""
        self.coordinate_history = {
            "left": [],
            "right": []
        }
        logging.info("Coordinate history cleared")
    
    def get_latest_coordinates(self) -> Tuple[Optional[Tuple[float, float, float]], 
                                            Optional[Tuple[float, float, float]]]:
        """
        Get the most recent detected coordinates.
        
        Returns:
            Tuple of (left_coords, right_coords)
        """
        left_coords = self.coordinate_history["left"][-1] if self.coordinate_history["left"] else None
        right_coords = self.coordinate_history["right"][-1] if self.coordinate_history["right"] else None
        return left_coords, right_coords
    
    def get_detection_rate(self) -> float:
        """
        Get the current detection rate.
        
        Returns:
            float: Current detection rate (0.0 to 1.0)
        """
        return self.detection_stats["detection_rate"]
    
    def update_detection_stats(self) -> None:
        """Update detection statistics based on current data."""
        # Check if current frame should be excluded from detection rate calculation
        # due to HSV range being too narrow (resulting in essentially no valid mask)
        if self.left_mask_too_narrow and self.right_mask_too_narrow:
            # If both masks are too narrow, don't count this frame for detection rate
            logging.debug("Skipping detection rate calculation - both masks have too narrow HSV range")
            return
            
        # Increment frame counter
        self.detection_stats["frames_processed"] += 1
        
        # Check if circles were detected in this frame
        if self.left_circles or self.right_circles:
            self.detection_stats["frames_detected"] += 1
            self.detection_stats["lost_frames"] = 0
        else:
            self.detection_stats["lost_frames"] += 1
        
        # Calculate detection rate
        if self.detection_stats["frames_processed"] > 0:
            detection_rate = self.detection_stats["frames_detected"] / self.detection_stats["frames_processed"]
            self.detection_stats["detection_rate"] = detection_rate
    
    def reset(self) -> None:
        """
        Reset the tracking data model to its initial state.
        Clears all images, masks, ROIs, and tracking data.
        Resets detection statistics.
        """
        # Reset images
        self.left_image = None
        self.right_image = None
        
        # Reset masks
        self.left_mask = None
        self.right_mask = None
        
        # Reset mask quality indicators
        self.left_mask_too_narrow = False
        self.right_mask_too_narrow = False
        
        # Reset ROIs
        self.left_roi = None
        self.right_roi = None
        
        # Reset tracking data
        self.left_circles = None
        self.right_circles = None
        self.left_prediction = None
        self.right_prediction = None
        
        # Reset cropped images
        self.cropped_images = {
            "left": None,
            "right": None
        }
        
        # Clear coordinate history
        self.clear_coordinate_history()
        
        # Reset detection statistics
        self.detection_stats = {
            "is_tracking": False,
            "frames_processed": 0,
            "frames_detected": 0,
            "detection_rate": 0.0,
            "lost_frames": 0
        }
        
        logging.info("Tracking data model reset") 