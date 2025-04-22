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


class BallTrackingController(QObject):
    """
    Controller class for ball tracking functionality.
    Handles HSV mask processing and ball detection.
    """
    
    # Signals
    mask_updated = Signal(np.ndarray, np.ndarray)  # left_mask, right_mask
    
    def __init__(self):
        """Initialize the ball tracking controller."""
        super(BallTrackingController, self).__init__()
        
        # Create configuration manager
        self.config_manager = ConfigManager()
        
        # Load HSV values from configuration
        self.hsv_values = self.config_manager.get_hsv_settings()
        
        # Store current images
        self.left_image = None
        self.right_image = None
        
        # Store current masks
        self.left_mask = None
        self.right_mask = None
        
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
        """Process the current images to generate HSV masks."""
        # Create masks
        self.left_mask = self._create_hsv_mask(self.left_image) if self.left_image is not None else None
        self.right_mask = self._create_hsv_mask(self.right_image) if self.right_image is not None else None
        
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