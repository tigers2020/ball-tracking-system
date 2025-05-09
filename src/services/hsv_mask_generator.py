#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HSV Mask Generator Module
This module contains the HSVMaskGenerator class for creating and processing HSV masks for ball tracking.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List, Any

from src.utils.constants import HSV, MORPHOLOGY, COLOR
from src.utils.logging_utils import log_service_init, log_service_update


class HSVMaskGenerator:
    """
    Service class for generating HSV masks to isolate objects by color.
    """

    def __init__(self, hsv_settings: Dict[str, Any]):
        """
        Initialize the HSV mask generator with provided HSV settings.
        
        Args:
            hsv_settings: Dictionary containing HSV ranges and parameters
                - h_min, h_max: Hue range (0-179)
                - s_min, s_max: Saturation range (0-255)
                - v_min, v_max: Value range (0-255)
                - morph_iterations: Number of iterations for morphological operations
                - blur_size: Size of the Gaussian blur kernel
                - dilation_iterations: Number of iterations for dilation
        """
        self.hsv_settings = hsv_settings.copy()
        
        # Normalize HSV parameter keys to ensure compatibility
        self._normalize_hsv_keys()
        
        # Log missing HSV parameters
        for param in ["h_min", "h_max", "s_min", "s_max", "v_min", "v_max"]:
            if param not in self.hsv_settings:
                logging.warning(f"Missing HSV parameter: {param}, using default")
        
        # Use the new logging utility instead of direct logging
        log_service_init("HSV mask generator", self.hsv_settings)

    def _normalize_hsv_keys(self) -> None:
        """
        Normalize HSV parameter keys to ensure compatibility with different naming conventions.
        Maps legacy keys to standardized format (h_min, h_max, s_min, s_max, v_min, v_max).
        """
        # Map of legacy keys to standardized keys
        key_mapping = {
            # Legacy hue keys
            'hMin': 'h_min', 'hMax': 'h_max',
            'lower_h': 'h_min', 'upper_h': 'h_max',
            # Legacy saturation keys
            'sMin': 's_min', 'sMax': 's_max',
            'lower_s': 's_min', 'upper_s': 's_max',
            # Legacy value keys
            'vMin': 'v_min', 'vMax': 'v_max',
            'lower_v': 'v_min', 'upper_v': 'v_max'
        }
        
        # Map legacy keys to standardized keys
        for legacy_key, standard_key in key_mapping.items():
            if legacy_key in self.hsv_settings and standard_key not in self.hsv_settings:
                self.hsv_settings[standard_key] = self.hsv_settings[legacy_key]
                logging.debug(f"Mapped legacy key '{legacy_key}' to standard key '{standard_key}'")

    def update_settings(self, hsv_settings: Dict[str, Any]) -> None:
        """
        Update the HSV settings.
        
        Args:
            hsv_settings: Dictionary containing HSV ranges and parameters
        """
        self.hsv_settings.update(hsv_settings.copy())
        # Apply key normalization after updating settings
        self._normalize_hsv_keys()
        # Use the new logging utility instead of direct logging
        log_service_update("HSV mask generator", self.hsv_settings)

    def update_hsv_values(self, hsv_values: Dict[str, Any]) -> None:
        """
        Update the HSV values. This is an alias for update_settings for compatibility.
        
        Args:
            hsv_values: Dictionary containing HSV ranges and parameters
        """
        self.update_settings(hsv_values)

    def generate_mask(self, img: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Tuple[int, int]], bool]:
        """
        Generate HSV mask for the given image.
        
        Args:
            img: Input BGR image
            roi: Region of interest (x, y, w, h) or None for the whole image
            
        Returns:
            Tuple of (original_image, masked_image, binary_mask, centroid, mask_too_narrow)
            where binary_mask is the raw binary mask,
            centroid is (x, y) or None if no contours found, and
            mask_too_narrow is a boolean flag indicating if the mask ratio is too small
        """
        try:
            # Create a copy of the original image for output
            output_img = img.copy()
            
            # Extract ROI if provided
            if roi is not None:
                x, y, w, h = roi
                roi_img = img[y:y+h, x:x+w]
            else:
                roi_img = img
                x, y = 0, 0
            
            # Convert image to HSV color space
            hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
            
            # Extract HSV thresholds from settings - use standardized keys (h_min, h_max, etc.)
            lower_h = self.hsv_settings.get('h_min', 0)
            upper_h = self.hsv_settings.get('h_max', 179)
            lower_s = self.hsv_settings.get('s_min', 0)
            upper_s = self.hsv_settings.get('s_max', 255)
            lower_v = self.hsv_settings.get('v_min', 0)
            upper_v = self.hsv_settings.get('v_max', 255)
            
            # Handle Hue wraparound case (e.g., red color which wraps around 0/179)
            if lower_h > upper_h:  # Wraparound case
                # Create two masks for the split Hue range
                mask1 = cv2.inRange(hsv, 
                                   np.array([lower_h, lower_s, lower_v]),
                                   np.array([179, upper_s, upper_v]))
                
                mask2 = cv2.inRange(hsv,
                                   np.array([0, lower_s, lower_v]),
                                   np.array([upper_h, upper_s, upper_v]))
                
                # Combine the two masks
                mask = cv2.bitwise_or(mask1, mask2)
                logging.debug(f"Using hue wraparound masking: {lower_h}-179 and 0-{upper_h}")
            else:
                # Standard case (no wraparound)
                mask = cv2.inRange(hsv,
                                  np.array([lower_h, lower_s, lower_v]),
                                  np.array([upper_h, upper_s, upper_v]))
            
            # Apply combined morphological operations and blur
            if mask.any():
                # Get morph parameters from settings
                morph_iterations = self.hsv_settings.get('morph_iterations', MORPHOLOGY.iterations)
                blur_size = self.hsv_settings.get('blur_size', HSV.blur_size)
                dilation_iterations = self.hsv_settings.get('dilation_iterations', HSV.dilation_iterations)
                
                # Ensure blur_size is a positive odd integer
                if blur_size <= 0:
                    # If blur_size is invalid, use default value
                    blur_size = HSV.blur_size
                    logging.warning(f"Invalid blur_size {self.hsv_settings.get('blur_size')}, using default value {HSV.blur_size}")
                elif blur_size % 2 == 0:
                    # If even, convert to nearest odd number
                    blur_size = blur_size + 1
                    logging.warning(f"Even blur_size {self.hsv_settings.get('blur_size')} converted to odd: {blur_size}")
                
                # Create kernel for morphological operations
                kernel = np.ones(MORPHOLOGY.kernel_size, np.uint8)
                
                # Apply morphological opening to remove small noise
                if morph_iterations > 0:
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
                
                # Apply morphological closing to fill small holes
                if morph_iterations > 0:
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
                
                # Apply Gaussian blur to smooth edges
                if blur_size > 0:
                    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
                    # Threshold again after blur to get binary mask
                    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
                
                # Apply dilation to expand the mask
                if dilation_iterations > 0:
                    mask = cv2.dilate(mask, kernel, iterations=dilation_iterations)
            
            # Store the binary mask for return
            binary_mask = mask.copy()
            
            # Check mask ratio for sanity (warn if too many or too few white pixels)
            mask_ratio = cv2.countNonZero(mask) / mask.size if mask.size > 0 else 0
            mask_too_narrow = False
            
            if mask_ratio > 0.9:
                logging.warning(f"Suspicious mask ratio {mask_ratio:.3f} (>90%) - HSV range may be too broad")
            elif mask_ratio < 0.0001:
                logging.warning(f"Suspicious mask ratio {mask_ratio:.6f} (<0.01%) - HSV range may be too narrow")
                mask_too_narrow = True
            else:
                logging.debug(f"Mask ratio: {mask_ratio:.3f}, non-zero pixels: {cv2.countNonZero(mask)}")
            
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            centroid = None
            
            # Process contours if any were found
            if contours:
                # Find the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Calculate moments of the largest contour
                M = cv2.moments(largest_contour)
                
                # Calculate centroid if possible
                if M["m00"] > 0:
                    # Calculate centroid coordinates
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Adjust for ROI offset
                    cx += x
                    cy += y
                    
                    # Store centroid
                    centroid = (cx, cy)
                    
                    # Draw all contours
                    cv2.drawContours(output_img, [cv2.convexHull(largest_contour)], -1, COLOR.YELLOW, 2)
                    
                    # Draw centroid point
                    cv2.circle(output_img, (cx, cy), 5, COLOR.MAGENTA, -1)
                    
                    logging.debug(f"Found HSV centroid at ({cx}, {cy})")
            else:
                logging.debug("No contours found in HSV mask")
            
            # Apply mask to original image for visualization
            masked_img = cv2.bitwise_and(roi_img, roi_img, mask=mask)
            
            # Draw ROI rectangle if provided
            if roi is not None:
                cv2.rectangle(output_img, (x, y), (x + w, y + h), COLOR.YELLOW, 2)
            
            return img, output_img, binary_mask, centroid, mask_too_narrow
            
        except Exception as e:
            logging.error(f"Error in HSV mask generation: {e}")
            if img is not None:
                empty_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8) if len(img.shape) >= 2 else None
                return img, img, empty_mask, None, False
            else:
                return None, None, None, None, False 