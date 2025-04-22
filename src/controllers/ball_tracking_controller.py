#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ball Tracking Controller module.
This module contains the BallTrackingController class for handling ball tracking functionality.
"""

import logging
import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

from src.utils.config_manager import ConfigManager
from src.utils.ui_constants import ROI


class BallTrackingController(QObject):
    """
    Controller class for ball tracking functionality.
    Handles HSV mask processing and ball detection.
    """
    
    # Signals
    mask_updated = Signal(np.ndarray, np.ndarray)  # left_mask, right_mask
    roi_updated = Signal(dict, dict)  # left_roi, right_roi (each containing x, y, width, height, center_x, center_y)
    
    def __init__(self):
        """Initialize the ball tracking controller."""
        super(BallTrackingController, self).__init__()
        
        # Create configuration manager
        self.config_manager = ConfigManager()
        
        # Load HSV values from configuration
        self.hsv_values = self.config_manager.get_hsv_settings()
        
        # Load ROI settings
        self.roi_settings = self.config_manager.get_roi_settings()
        
        # Store current images
        self.left_image = None
        self.right_image = None
        
        # Store current masks
        self.left_mask = None
        self.right_mask = None
        
        # Store current ROIs
        self.left_roi = None
        self.right_roi = None
        
        # Track enabled state
        self.is_enabled = False
    
    def set_hsv_values(self, hsv_values):
        """
        Set HSV threshold values for ball detection.
        
        Args:
            hsv_values (dict): Dictionary containing HSV min/max values
        """
        # Update HSV values
        for key, value in hsv_values.items():
            if key in self.hsv_values:
                self.hsv_values[key] = value
        
        # Save updated values to configuration
        self.config_manager.set_hsv_settings(self.hsv_values)
        
        logging.info(f"HSV values updated and saved: {self.hsv_values}")
        
        # Apply updated HSV values if enabled
        if self.is_enabled and (self.left_image is not None or self.right_image is not None):
            self._process_images()
    
    def set_images(self, left_image, right_image):
        """
        Set the current stereo images for processing.
        
        Args:
            left_image (numpy.ndarray): Left OpenCV image
            right_image (numpy.ndarray): Right OpenCV image
        """
        self.left_image = left_image
        self.right_image = right_image
        
        # Process images if enabled
        if self.is_enabled:
            self._process_images()
    
    def enable(self, enabled=True):
        """
        Enable or disable ball tracking.
        
        Args:
            enabled (bool): True to enable, False to disable
        """
        if self.is_enabled != enabled:
            self.is_enabled = enabled
            logging.info(f"Ball tracking {'enabled' if enabled else 'disabled'}")
            
            if enabled and (self.left_image is not None or self.right_image is not None):
                self._process_images()
            else:
                # Clear masks
                self.left_mask = None
                self.right_mask = None
                self.mask_updated.emit(None, None)
    
    def _process_images(self):
        """Process the current images to generate HSV masks and ROIs."""
        # Create masks
        self.left_mask = self._create_hsv_mask(self.left_image) if self.left_image is not None else None
        self.right_mask = self._create_hsv_mask(self.right_image) if self.right_image is not None else None
        
        # Calculate ROIs if enabled
        if self.roi_settings["enabled"]:
            self.left_roi = self._calculate_roi(self.left_mask, self.left_image) if self.left_mask is not None else None
            self.right_roi = self._calculate_roi(self.right_mask, self.right_image) if self.right_mask is not None else None
            
            # Emit ROI signal
            self.roi_updated.emit(self.left_roi, self.right_roi)
        else:
            self.left_roi = None
            self.right_roi = None
            self.roi_updated.emit(None, None)
        
        # Emit signal with masks
        self.mask_updated.emit(self.left_mask, self.right_mask)
    
    def _create_hsv_mask(self, image):
        """
        Create an HSV mask for the given image.
        
        Args:
            image (numpy.ndarray): OpenCV BGR image
            
        Returns:
            numpy.ndarray: Binary mask image
        """
        if image is None:
            return None
        
        try:
            # Convert image to HSV
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Create lower and upper HSV boundaries
            lower_bound = np.array([
                self.hsv_values["h_min"],
                self.hsv_values["s_min"],
                self.hsv_values["v_min"]
            ])
            
            upper_bound = np.array([
                self.hsv_values["h_max"],
                self.hsv_values["s_max"],
                self.hsv_values["v_max"]
            ])
            
            # Handle the case when h_min > h_max (for red color that wraps around hue value)
            if self.hsv_values["h_min"] > self.hsv_values["h_max"]:
                # Create two masks and combine them
                lower_mask = cv2.inRange(hsv_image, 
                                        np.array([0, self.hsv_values["s_min"], self.hsv_values["v_min"]]), 
                                        np.array([self.hsv_values["h_max"], self.hsv_values["s_max"], self.hsv_values["v_max"]]))
                
                upper_mask = cv2.inRange(hsv_image, 
                                        np.array([self.hsv_values["h_min"], self.hsv_values["s_min"], self.hsv_values["v_min"]]), 
                                        np.array([179, self.hsv_values["s_max"], self.hsv_values["v_max"]]))
                
                # Combine masks
                mask = cv2.bitwise_or(lower_mask, upper_mask)
            else:
                # Create standard mask
                mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
            
            # Enhanced morphological processing for cleaner mask
            # Create a kernel for morphological operations
            kernel = np.ones((5, 5), np.uint8)
            
            # Noise removal (small specks)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Fill holes in detected objects
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Log HSV ranges and mask statistics for debugging
            non_zero_count = cv2.countNonZero(mask)
            total_pixels = mask.shape[0] * mask.shape[1]
            coverage_percent = (non_zero_count / total_pixels) * 100 if total_pixels > 0 else 0
            
            logging.debug(f"HSV Range: H({self.hsv_values['h_min']}-{self.hsv_values['h_max']}), " +
                         f"S({self.hsv_values['s_min']}-{self.hsv_values['s_max']}), " +
                         f"V({self.hsv_values['v_min']}-{self.hsv_values['v_max']})")
            logging.debug(f"Mask coverage: {coverage_percent:.2f}% ({non_zero_count}/{total_pixels} pixels)")
            
            return mask
        
        except Exception as e:
            logging.error(f"Error creating HSV mask: {e}")
            return None
    
    def _calculate_roi(self, mask, image):
        """
        Calculate ROI based on mask and image.
        
        Args:
            mask (numpy.ndarray): Binary mask
            image (numpy.ndarray): Image
            
        Returns:
            dict: ROI information (x, y, width, height, center_x, center_y)
        """
        if mask is None or image is None:
            return None
            
        try:
            # Get ROI width and height from settings
            roi_width = self.roi_settings["width"]
            roi_height = self.roi_settings["height"]
            
            # Calculate center of the mask
            if self.roi_settings["auto_center"]:
                center_x, center_y = self._compute_mask_centroid(mask)
            else:
                # Use image center if no auto-center
                center_x = image.shape[1] // 2
                center_y = image.shape[0] // 2
            
            # Calculate ROI coordinates
            x = max(0, center_x - roi_width // 2)
            y = max(0, center_y - roi_height // 2)
            
            # Adjust if ROI goes beyond image boundaries
            if x + roi_width > image.shape[1]:
                x = image.shape[1] - roi_width
            if y + roi_height > image.shape[0]:
                y = image.shape[0] - roi_height
                
            # Ensure ROI is within image bounds
            x = max(0, min(x, image.shape[1] - roi_width))
            y = max(0, min(y, image.shape[0] - roi_height))
            
            # Create ROI dict
            roi = {
                "x": x,
                "y": y,
                "width": roi_width,
                "height": roi_height,
                "center_x": center_x,
                "center_y": center_y
            }
            
            return roi
            
        except Exception as e:
            logging.error(f"Error calculating ROI: {e}")
            return None
            
    def _compute_mask_centroid(self, mask):
        """
        Compute the centroid of a binary mask using moments.
        
        Args:
            mask (numpy.ndarray): Binary mask
            
        Returns:
            tuple: (center_x, center_y) coordinates
        """
        try:
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                # If no contours found, use center of the mask
                return mask.shape[1] // 2, mask.shape[0] // 2
                
            # Find the largest contour based on area
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate moments of the largest contour
            M = cv2.moments(largest_contour)
            
            # Calculate centroid
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
            else:
                # If moments calculation fails, use center of the mask
                center_x = mask.shape[1] // 2
                center_y = mask.shape[0] // 2
                
            return center_x, center_y
            
        except Exception as e:
            logging.error(f"Error computing mask centroid: {e}")
            # Return center of the mask as fallback
            return mask.shape[1] // 2, mask.shape[0] // 2
    
    def get_current_masks(self):
        """
        Get the current masks.
        
        Returns:
            tuple: (left_mask, right_mask)
        """
        return self.left_mask, self.right_mask
    
    def get_hsv_values(self):
        """
        Get the current HSV values.
        
        Returns:
            dict: Current HSV values
        """
        return self.hsv_values
    
    def set_roi_settings(self, roi_settings):
        """
        Set ROI settings for ball tracking.
        
        Args:
            roi_settings (dict): Dictionary containing ROI settings
        """
        # Update ROI settings
        for key, value in roi_settings.items():
            if key in self.roi_settings:
                self.roi_settings[key] = value
        
        # Save updated values to configuration
        self.config_manager.set_roi_settings(self.roi_settings)
        
        logging.info(f"ROI settings updated and saved: {self.roi_settings}")
        
        # Reprocess images if enabled to update ROIs
        if self.is_enabled and (self.left_image is not None or self.right_image is not None):
            self._process_images() 
    
    def get_roi_settings(self):
        """
        Get the current ROI settings.
        
        Returns:
            dict: Current ROI settings
        """
        return self.roi_settings
        
    def get_current_rois(self):
        """
        Get the current ROIs.
        
        Returns:
            tuple: (left_roi, right_roi)
        """
        return self.left_roi, self.right_roi 