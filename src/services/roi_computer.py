#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ROI Computer module.
This module contains the ROIComputer class for calculating ROIs from masks.
"""

import logging
import cv2
import numpy as np
from typing import Dict, Tuple, Optional, Any


class ROIComputer:
    """
    Service class for computing Region of Interest (ROI) from masks.
    """
    
    def __init__(self, roi_settings: Dict[str, Any]):
        """
        Initialize the ROI computer.
        
        Args:
            roi_settings: Dictionary containing ROI settings (width, height, auto_center, etc.)
        """
        self.roi_settings = roi_settings
    
    def update_roi_settings(self, roi_settings: Dict[str, Any]) -> None:
        """
        Update ROI settings.
        
        Args:
            roi_settings: Dictionary containing ROI settings
        """
        self.roi_settings = roi_settings.copy()
    
    def compute_roi(self, mask: np.ndarray, image: np.ndarray) -> Optional[Dict[str, int]]:
        """
        Calculate ROI based on mask and image.
        
        Args:
            mask: Binary mask
            image: Image
            
        Returns:
            ROI information (x, y, width, height, center_x, center_y) or None if fails
        """
        if mask is None or image is None:
            logging.debug("Cannot compute ROI: mask or image is None")
            return None
            
        try:
            # Get image dimensions
            img_height, img_width = image.shape[:2]
            
            # Get ROI width and height from settings and ensure they're positive
            roi_width = max(1, abs(self.roi_settings.get("width", 100)))
            roi_height = max(1, abs(self.roi_settings.get("height", 100)))
            
            # Calculate center coordinates
            if self.roi_settings.get("auto_center", True):
                # Use mask centroid
                center_x, center_y = self.compute_mask_centroid(mask)
            else:
                # Use image center
                center_x = img_width // 2
                center_y = img_height // 2
            
            # Calculate ROI coordinates with boundary checking
            x = np.clip(center_x - roi_width // 2, 0, img_width - roi_width)
            y = np.clip(center_y - roi_height // 2, 0, img_height - roi_height)
            
            # Create ROI dict
            roi = {
                "x": int(x),
                "y": int(y),
                "width": int(roi_width),
                "height": int(roi_height),
                "center_x": int(center_x),
                "center_y": int(center_y)
            }
            
            logging.debug(f"ROI computed: x={roi['x']}, y={roi['y']}, w={roi['width']}, h={roi['height']}")
            
            return roi
            
        except Exception as e:
            logging.error(f"Error calculating ROI: {e}")
            return None
    
    def compute_mask_centroid(self, mask: np.ndarray) -> Tuple[int, int]:
        """
        Compute the centroid of a binary mask using moments.
        
        Args:
            mask: Binary mask
            
        Returns:
            (center_x, center_y) coordinates
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
    
    def crop_roi_image(self, image: np.ndarray, roi: Dict[str, int]) -> Optional[np.ndarray]:
        """
        Crop the image based on ROI information.
        
        Args:
            image: Original image
            roi: ROI information with x, y, width, height
            
        Returns:
            Cropped image or None if ROI or image is invalid
        """
        if image is None or roi is None:
            logging.debug("Cannot crop ROI: image or ROI is None")
            return None
            
        try:
            # Extract and validate ROI coordinates
            try:
                x = int(roi.get("x", 0))
                y = int(roi.get("y", 0))
                w = int(roi.get("width", 0))
                h = int(roi.get("height", 0))
            except (ValueError, TypeError) as e:
                logging.error(f"Invalid ROI values: {e}")
                return None
            
            # Ensure width and height are positive
            w = max(1, w)
            h = max(1, h)
            
            # Ensure coordinates are within image bounds
            img_height, img_width = image.shape[:2]
            x = np.clip(x, 0, img_width - 1)
            y = np.clip(y, 0, img_height - 1)
            w = min(w, img_width - x)
            h = min(h, img_height - y)
            
            # Crop the image
            cropped_image = image[y:y+h, x:x+w]
            
            logging.debug(f"Image cropped to ROI: x={x}, y={y}, w={w}, h={h}")
            
            return cropped_image
            
        except Exception as e:
            logging.error(f"Error cropping ROI image: {e}")
            return None 