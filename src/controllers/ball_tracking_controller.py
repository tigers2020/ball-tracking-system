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
from pathlib import Path
from PySide6.QtCore import QObject, Signal

from src.utils.config_manager import ConfigManager
from src.utils.ui_constants import ROI
from src.utils.coord_utils import fuse_coordinates
from src.utils.kalman_factory import KalmanFilterFactory


class BallTrackingController(QObject):
    """
    Controller class for ball tracking functionality.
    Handles HSV mask processing and ball detection.
    """
    
    # Signals
    mask_updated = Signal(np.ndarray, np.ndarray)  # left_mask, right_mask
    roi_updated = Signal(dict, dict)  # left_roi, right_roi (each containing x, y, width, height, center_x, center_y)
    detection_updated = Signal(float, tuple, tuple)  # detection_rate, left_coords, right_coords
    circles_processed = Signal(np.ndarray, np.ndarray)  # left_circle_image, right_circle_image
    
    def __init__(self):
        """Initialize the ball tracking controller."""
        super(BallTrackingController, self).__init__()
        
        # Kalman readiness flags per side (must be defined before initializing filters)
        self.kalman_ready = {"left": False, "right": False}

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
        
        # Store circle detection results
        self.left_circles = None
        self.right_circles = None
        
        # For coordinate tracking
        self.coordinate_history = {
            "left": [],  # List of (x, y, r, timestamp) tuples
            "right": []  # List of (x, y, r, timestamp) tuples
        }
        self.max_history_length = 100  # Maximum number of coordinate points to store
        
        # Store predicted coordinates (must be initialized BEFORE Kalman filters)
        self.prediction = {
            "left": None,  # (x, y, vx, vy) predicted state
            "right": None  # (x, y, vx, vy) predicted state
        }
        
        # Counter for number of Kalman updates per side to ignore initial predictions
        self.kalman_update_counter = {"left": 0, "right": 0}
        
        # Store last measurement for velocity estimation
        self.last_measurement = {"left": None, "right": None}
        
        # Initialize Kalman filters
        self.left_kalman = None
        self.right_kalman = None
        self._init_kalman_filters()
        
        # Detection rate tracking
        self.detection_stats = {
            "first_detection_time": None,  # Time of first simultaneous detection
            "detection_count": 0,          # Number of successful detections
            "total_frames": 0,             # Total number of frames processed
            "is_tracking": False           # Whether we're currently tracking
        }
        
        # Store cropped ROI images
        self.cropped_images = {
            "left": None,
            "right": None
        }
        
        # Track enabled state
        self.is_enabled = False
        
        # XML tracking data
        self.xml_root = None
        self.current_folder = None
    
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
        
        # Update frame count for detection rate
        if self.is_enabled and self.detection_stats["is_tracking"]:
            self.detection_stats["total_frames"] += 1
        
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
        
        # Detect circles in the current ROIs
        left_result, right_result = self.detect_circles_in_roi()
        
        # Emit circles processed signal if we have valid results
        if left_result[0] is not None or right_result[0] is not None:
            self.circles_processed.emit(
                left_result[0] if left_result[0] is not None else np.zeros_like(self.left_image) if self.left_image is not None else np.zeros((480, 640, 3), dtype=np.uint8),
                right_result[0] if right_result[0] is not None else np.zeros_like(self.right_image) if self.right_image is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            )
    
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
            
            # Apply Gaussian blur to the mask
            mask = cv2.GaussianBlur(mask, (7, 7), 0)
            
            # Apply threshold to get back binary mask after blur
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            # Extend the mask by 2 pixels using dilation
            extend_kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, extend_kernel, iterations=2)
            
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
        
    def crop_roi_image(self, image, roi):
        """
        Crop the image based on ROI information.
        
        Args:
            image (numpy.ndarray): Original image
            roi (dict): ROI information with x, y, width, height
            
        Returns:
            numpy.ndarray: Cropped image or None if ROI or image is invalid
        """
        if image is None or roi is None:
            return None
            
        try:
            # Extract ROI coordinates
            x = roi["x"]
            y = roi["y"]
            w = roi["width"]
            h = roi["height"]
            
            # Ensure coordinates are within image bounds
            x = max(0, min(x, image.shape[1] - 1))
            y = max(0, min(y, image.shape[0] - 1))
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            # Crop the image
            cropped_image = image[y:y+h, x:x+w]
            
            logging.debug(f"Image cropped to ROI: x={x}, y={y}, w={w}, h={h}")
            
            return cropped_image
            
        except Exception as e:
            logging.error(f"Error cropping ROI image: {e}")
            return None
    
    def get_cropped_roi_images(self):
        """
        Get cropped images based on current ROIs.
        
        Returns:
            tuple: (left_cropped_image, right_cropped_image)
        """
        left_cropped = None
        right_cropped = None
        
        if self.left_image is not None and self.left_roi is not None:
            left_cropped = self.crop_roi_image(self.left_image, self.left_roi)
            
        if self.right_image is not None and self.right_roi is not None:
            right_cropped = self.crop_roi_image(self.right_image, self.right_roi)
        
        # Store cropped images for later use
        self.cropped_images["left"] = left_cropped
        self.cropped_images["right"] = right_cropped
            
        return left_cropped, right_cropped
        
    def detect_circles(self, image, roi=None):
        """
        Detect circles in an image using Hough transform.
        If ROI is provided, circles will be detected within the ROI.
        
        Args:
            image (numpy.ndarray): Image to process
            roi (dict, optional): ROI information to limit detection area
            
        Returns:
            tuple: (circle_image, circles) where:
                - circle_image is the original image with circles drawn
                - circles is a list of circle parameters (x, y, r)
        """
        if image is None:
            return None, None
            
        try:
            # Work with a copy of the image
            circle_image = image.copy()
            
            # Crop to ROI if provided
            if roi is not None:
                roi_image = self.crop_roi_image(image, roi)
            else:
                roi_image = image
                
            # Convert to grayscale
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
            
            # Apply GaussianBlur to reduce noise
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            
            # Get Hough circle parameters from configuration
            hough_settings = self.config_manager.get_hough_circle_settings()
            
            # Adjust Hough parameters based on ROI size
            adaptive = hough_settings.get("adaptive", False)
            if roi is not None and adaptive:
                # Only adjust min_dist to maximum of half ROI size
                min_half = min(roi["width"], roi["height"]) / 2
                hough_settings["min_dist"] = max(hough_settings.get("min_dist", 0), min_half)
                # Scale dp proportionally (smaller ROI -> higher resolution)
                hough_settings["dp"] = max(1.0, hough_settings.get("dp", 1.0) * (hough_settings["min_dist"] / min_half))
                logging.debug(f"Adaptive Hough parameters for ROI: {hough_settings}")
            
            # Detect circles using Hough transform with adjusted parameters
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=hough_settings["dp"],
                minDist=hough_settings["min_dist"],
                param1=hough_settings.get("param1"),
                param2=hough_settings.get("param2"),
                minRadius=hough_settings.get("min_radius"),
                maxRadius=hough_settings.get("max_radius")
            )
            
            detected_circles = []
            
            # Determine which side (left or right) this image represents
            side = None
            if image is self.left_image or (roi is not None and self.left_roi is not None and roi == self.left_roi):
                side = "left"
            elif image is self.right_image or (roi is not None and self.right_roi is not None and roi == self.right_roi):
                side = "right"
                
            # Get HSV mask centroid
            hsv_mask = self.left_mask if side == "left" else self.right_mask
            hsv_center = self._compute_mask_centroid(hsv_mask) if hsv_mask is not None else None
            
            # Get Kalman prediction
            kalman_pred = self.prediction[side] if side and self.prediction[side] is not None and self.detection_stats["is_tracking"] else None
            # Ignore initial Kalman predictions for first 5 updates
            if kalman_pred is not None and self.kalman_update_counter[side] <= 5:
                kalman_pred = None
            
            # Draw detected circles
            if circles is not None:
                circles = np.uint16(np.around(circles))
                
                for circle in circles[0, :]:
                    # Get circle parameters (center x, center y, radius)
                    x, y, r = circle
                    
                    # Adjust coordinates if using ROI
                    if roi is not None:
                        x += roi["x"]
                        y += roi["y"]
                    
                    # Store circle information
                    detected_circles.append((x, y, r))
                    
                    # Draw outer circle (green)
                    cv2.circle(circle_image, (int(x), int(y)), int(r), (0, 255, 0), 2)
                    
                    # Draw center (red)
                    cv2.circle(circle_image, (int(x), int(y)), 2, (0, 0, 255), 3)
                    
                    # Add radius text
                    cv2.putText(
                        circle_image,
                        f"r={r}",
                        (int(x + r + 5), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1
                    )
                    
                logging.debug(f"Detected {len(detected_circles)} circles")
            else:
                logging.debug("No circles detected")
            
            # Get Hough circle center if available
            hough_center = None
            if detected_circles:
                hough_center = (detected_circles[0][0], detected_circles[0][1])
            
            # Draw Kalman prediction for the detected circles
            if kalman_pred is not None and self.kalman_update_counter[side] > 5 and not (kalman_pred[0] == 0.0 and kalman_pred[1] == 0.0):
                px, py, vx, vy = kalman_pred
                
                # Draw predicted position (blue circle)
                # Use the radius of the detected circle if available, otherwise use 15 as default
                pred_radius = detected_circles[0][2] if detected_circles else 15
                cv2.circle(circle_image, (int(px), int(py)), int(pred_radius), (255, 0, 0), 2)  # Blue circle for prediction
                
                # Draw center of prediction (blue dot)
                cv2.circle(circle_image, (int(px), int(py)), 2, (255, 0, 0), 3)
                
                # Add prediction text
                cv2.putText(
                    circle_image,
                    f"Kalman",
                    (int(px + 5), int(py - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1
                )
                
                # Optionally draw velocity vector
                vector_scale = 3
                end_x = int(px + vx * vector_scale)
                end_y = int(py + vy * vector_scale)
                cv2.arrowedLine(circle_image, (int(px), int(py)), (end_x, end_y), (255, 0, 0), 2)
                
                logging.debug(f"Drew Kalman prediction at ({px}, {py}) with velocity ({vx}, {vy})")
            
            # Draw HSV mask centroid if available
            if hsv_center is not None:
                cx, cy = hsv_center
                
                # Draw HSV centroid (magenta circle)
                cv2.circle(circle_image, (int(cx), int(cy)), 5, (255, 0, 255), 2)  # Magenta circle
                
                # Add HSV text
                cv2.putText(
                    circle_image,
                    f"HSV",
                    (int(cx + 5), int(cy - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    1
                )
                
                logging.debug(f"Drew HSV centroid at ({cx}, {cy})")
            
            # Draw fused coordinates
            fused = self._fuse_coordinates(side, hsv_center, hough_center, kalman_pred, self.detection_stats["is_tracking"] == False)
            if fused is not None:
                fx, fy = fused
                
                # Draw fused position (yellow star)
                cv2.drawMarker(
                    circle_image, 
                    (int(fx), int(fy)),
                    (0, 255, 255),  # Yellow
                    markerType=cv2.MARKER_STAR,
                    markerSize=20,
                    thickness=2
                )
                
                # Add fused text
                cv2.putText(
                    circle_image,
                    f"Fused",
                    (int(fx) + 10, int(fy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1
                )
                
                logging.debug(f"Drew fused coordinate at ({fx:.2f}, {fy:.2f})")
                
            return circle_image, detected_circles
            
        except Exception as e:
            logging.error(f"Error detecting circles: {e}")
            return image, None
    
    def _update_detection_rate(self):
        """
        Update detection rate based on current detection status.
        """
        import time
        
        # Initialize Kalman for sides independently if not ready
        if self.left_circles and not self.kalman_ready.get("left", False):
            lx, ly, _ = self.left_circles[0]
            self._set_kalman_initial_state("left", lx, ly)
        if self.right_circles and not self.kalman_ready.get("right", False):
            rx, ry, _ = self.right_circles[0]
            self._set_kalman_initial_state("right", rx, ry)
        # Update tracking flag based on Kalman readiness
        self.detection_stats["is_tracking"] = self.kalman_ready.get("left", False) or self.kalman_ready.get("right", False)

        # Both circles detected
        if self.left_circles and self.right_circles:
            # First simultaneous detection
            if self.detection_stats["first_detection_time"] is None:
                self.detection_stats["first_detection_time"] = time.time()
                self.detection_stats["is_tracking"] = True
                self.detection_stats["detection_count"] = 1
                self.detection_stats["total_frames"] = 1
                
                # Initialize Kalman filters with current detections
                if self.left_circles and self.right_circles:
                    # Get first detected circles from both sides
                    left_x, left_y, left_r = self.left_circles[0]
                    right_x, right_y, right_r = self.right_circles[0]
                    
                    # Initialize Kalman filters with these detections
                    self._set_kalman_initial_state("left", left_x, left_y)
                    self._set_kalman_initial_state("right", right_x, right_y)
                    
                    logging.info("First simultaneous detection, starting tracking and initializing Kalman filters")
                else:
                    logging.info("First simultaneous detection, starting tracking")
            
            # Already tracking
            elif self.detection_stats["is_tracking"]:
                self.detection_stats["detection_count"] += 1
                logging.debug(f"Detection count: {self.detection_stats['detection_count']}/{self.detection_stats['total_frames']}")
        
        # Check if ball is out of bounds (based on predictions)
        elif self.detection_stats["is_tracking"]:
            left_pred, right_pred = self.get_predictions()
            
            # Check if predictions indicate ball is out of bounds
            is_out_of_bounds = self._check_out_of_bounds(left_pred, right_pred)
            
            # If ball is predicted to be out of bounds, stop tracking
            if is_out_of_bounds:
                logging.info("Ball predicted out of bounds, stopping tracking")
                self.detection_stats["is_tracking"] = False
                # Reset Kalman predictions when tracking stops
                self.prediction["left"] = None
                self.prediction["right"] = None
        
        # Get current detection rate
        detection_rate = self.get_detection_rate()
        
        # Get latest coordinates
        left_coords, right_coords = self.get_latest_coordinates()
        
        # Emit detection update signal
        self.detection_updated.emit(detection_rate, left_coords, right_coords)
    
    def _check_out_of_bounds(self, left_pred, right_pred):
        """
        Check if ball is predicted to be out of bounds.
        
        Args:
            left_pred (tuple): Left camera prediction (x, y, vx, vy)
            right_pred (tuple): Right camera prediction (x, y, vx, vy)
            
        Returns:
            bool: True if ball is predicted to be out of bounds
        """
        if left_pred is None and right_pred is None:
            return False
            
        # Define image boundaries
        left_bounds = (0, 0, self.left_image.shape[1], self.left_image.shape[0]) if self.left_image is not None else None
        right_bounds = (0, 0, self.right_image.shape[1], self.right_image.shape[0]) if self.right_image is not None else None
        
        # Check left prediction
        if left_pred is not None and left_bounds is not None:
            px, py, vx, vy = left_pred
            
            # Predicted position is outside image bounds
            if (px < 0 or px >= left_bounds[2] or py < 0 or py >= left_bounds[3]):
                logging.debug(f"Left prediction out of bounds: ({px}, {py})")
                return True
                
            # Predict future positions based on velocity
            future_steps = 5  # Look 5 frames ahead
            future_x = px + vx * future_steps
            future_y = py + vy * future_steps
            
            if (future_x < 0 or future_x >= left_bounds[2] or future_y < 0 or future_y >= left_bounds[3]):
                logging.debug(f"Left future prediction out of bounds: ({future_x}, {future_y})")
                return True
        
        # Check right prediction (similar to left)
        if right_pred is not None and right_bounds is not None:
            px, py, vx, vy = right_pred
            
            if (px < 0 or px >= right_bounds[2] or py < 0 or py >= right_bounds[3]):
                logging.debug(f"Right prediction out of bounds: ({px}, {py})")
                return True
                
            future_steps = 5  # Look 5 frames ahead
            future_x = px + vx * future_steps
            future_y = py + vy * future_steps
            
            if (future_x < 0 or future_x >= right_bounds[2] or future_y < 0 or future_y >= right_bounds[3]):
                logging.debug(f"Right future prediction out of bounds: ({future_x}, {future_y})")
                return True
        
        return False
    
    def get_detection_rate(self):
        """
        Get the current detection rate.
        
        Returns:
            float: Detection rate (0.0 to 1.0) or None if not tracking
        """
        if not self.detection_stats["is_tracking"]:
            if self.detection_stats["total_frames"] > 0:
                # Return final detection rate if tracking has stopped
                return self.detection_stats["detection_count"] / self.detection_stats["total_frames"]
            else:
                return 0.0
                
        # If tracking, return current rate
        elif self.detection_stats["total_frames"] > 0:
            return self.detection_stats["detection_count"] / self.detection_stats["total_frames"]
        else:
            return 0.0
    
    def detect_circles_in_roi(self):
        """
        Detect circles in the current ROIs.
        
        Returns:
            tuple: (left_result, right_result) where each result contains:
                - processed_image: Image with circles drawn
                - circles: List of detected circles (x, y, r)
        """
        try:
            left_result = (None, None)
            right_result = (None, None)
            
            # Process left image
            if self.left_image is not None and self.left_roi is not None:
                # Use already cropped image if available, otherwise crop it
                if "left" not in self.cropped_images or self.cropped_images["left"] is None:
                    self.cropped_images["left"] = self.crop_roi_image(self.left_image, self.left_roi)
                
                left_processed, left_circles = self.detect_circles(self.left_image, self.left_roi)
                left_result = (left_processed, left_circles)
                
                # Store circles for tracking
                self.left_circles = left_circles
                
                # Record circle coordinates
                if left_circles:
                    # Add coordinates to history and update Kalman filter if tracking is active
                    x, y, r = left_circles[0]
                    self._add_coordinate("left", x, y, r)
                
            # Process right image
            if self.right_image is not None and self.right_roi is not None:
                # Use already cropped image if available, otherwise crop it
                if "right" not in self.cropped_images or self.cropped_images["right"] is None:
                    self.cropped_images["right"] = self.crop_roi_image(self.right_image, self.right_roi)
                
                right_processed, right_circles = self.detect_circles(self.right_image, self.right_roi)
                right_result = (right_processed, right_circles)
                
                # Store circles for tracking
                self.right_circles = right_circles
                
                # Record circle coordinates
                if right_circles:
                    # Add coordinates to history and update Kalman filter if tracking is active
                    x, y, r = right_circles[0]
                    self._add_coordinate("right", x, y, r)
            
            # Update detection rate
            self._update_detection_rate()
                
            return left_result, right_result
            
        except Exception as e:
            logging.error(f"Error in detect_circles_in_roi: {e}")
            return (None, None), (None, None)
    
    def _init_kalman_filters(self):
        """Initialize Kalman filters for both left and right camera tracking with factory or fallback."""
        # Ensure readiness flags exist and reset them
        if not hasattr(self, 'kalman_ready'):
            self.kalman_ready = {"left": False, "right": False}
        else:
            self.kalman_ready["left"] = False
            self.kalman_ready["right"] = False
        # Attempt dynamic configuration via factory
        config_path = os.path.join(os.getcwd(), "config", "kalman_config.yaml")
        try:
            factory = KalmanFilterFactory(config_path)
            self.left_kalman, left_config = factory.create_kalman_filter()
            self.right_kalman, right_config = factory.create_kalman_filter()
            self.prediction["left"] = None
            self.prediction["right"] = None
            # Store time step for velocity calculation
            self.kalman_dt = left_config.get('dt', 1.0)
            logging.info(
                f"Kalman filters initialized with dt={left_config['dt']}, "
                f"process_noise={left_config['process_noise']}, "
                f"measurement_noise={left_config['measurement_noise']}"
            )
            return
        except Exception as e:
            logging.error(f"Error initializing Kalman filters with factory: {e}")
        # Fallback to basic initialization if factory fails
        try:
            self.left_kalman = cv2.KalmanFilter(4, 2)
            self.right_kalman = cv2.KalmanFilter(4, 2)
            # State transition matrix
            transition = np.array([
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], np.float32)
            self.left_kalman.transitionMatrix = transition
            self.right_kalman.transitionMatrix = transition
            # Measurement matrix
            measure = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
            self.left_kalman.measurementMatrix = measure
            self.right_kalman.measurementMatrix = measure
            # Process noise covariance
            proc_noise = np.eye(4, dtype=np.float32) * 0.03
            self.left_kalman.processNoiseCov = proc_noise
            self.right_kalman.processNoiseCov = proc_noise
            # Measurement noise covariance
            meas_noise = np.eye(2, dtype=np.float32) * 1.0
            self.left_kalman.measurementNoiseCov = meas_noise
            self.right_kalman.measurementNoiseCov = meas_noise
            # Error covariance post
            err_cov = np.eye(4, dtype=np.float32) * 1.0
            self.left_kalman.errorCovPost = err_cov
            self.right_kalman.errorCovPost = err_cov
            self.prediction["left"] = None
            self.prediction["right"] = None
            # Default time step
            self.kalman_dt = 1.0
            logging.info("Kalman filters initialized with default configuration (fallback mode)")
        except Exception as e:
            logging.error(f"Error initializing Kalman filters fallback: {e}")
        return
    
    def _set_kalman_initial_state(self, side, x, y):
        """
        Set the initial state of the Kalman filter to the specified position.
        This ensures that predictions start from the actual detected position.
        
        Args:
            side (str): "left" or "right" to indicate which Kalman filter to use
            x (int): Initial X coordinate
            y (int): Initial Y coordinate
        """
        try:
            kalman = self.left_kalman if side == "left" else self.right_kalman
            
            # Set initial state (x, y, vx=0, vy=0)
            kalman.statePost = np.array([
                [np.float32(x)],
                [np.float32(y)],
                [np.float32(0)],  # Initial velocity x = 0
                [np.float32(0)]   # Initial velocity y = 0
            ])
            
            # Set initial prediction
            self.prediction[side] = (x, y, 0, 0)
            
            logging.info(f"Kalman filter for {side} initialized with position ({x}, {y})")
            
            # Mark Kalman ready for this side
            self.kalman_ready[side] = True
            
        except Exception as e:
            logging.error(f"Error setting initial Kalman state: {e}")
            
    def _update_kalman(self, side, x, y):
        """
        Update Kalman filter with new measurement and get prediction.
        
        Args:
            side (str): "left" or "right" to indicate which Kalman filter to use
            x (int): X coordinate of the measurement
            y (int): Y coordinate of the measurement
            
        Returns:
            tuple: Predicted (x, y, vx, vy) state as floats
        """
        try:
            kalman = self.left_kalman if side == "left" else self.right_kalman
            
            # Convert to float32
            measurement = np.array([[np.float32(x)], [np.float32(y)]])
            
            # Predict next state before correction (for visualization)
            prediction = kalman.predict()
            # Correct with current measurement
            kalman.correct(measurement)
            
            # Extract predicted state (maintain float precision)
            px = float(prediction[0, 0])
            py = float(prediction[1, 0])
            vx = float(prediction[2, 0])
            vy = float(prediction[3, 0])
            
            # Store prediction
            self.prediction[side] = (px, py, vx, vy)
            
            # Increment counter for excluding initial predictions
            self.kalman_update_counter[side] += 1
            
            logging.debug(f"Kalman prediction for {side}: pos=({px:.2f}, {py:.2f}), vel=({vx:.2f}, {vy:.2f})")
            
            return px, py, vx, vy
            
        except Exception as e:
            logging.error(f"Error updating Kalman filter: {e}")
            return None
    
    def _add_coordinate(self, side, x, y, r):
        """
        Add a circle coordinate to the history and update Kalman prediction.
        
        Args:
            side (str): "left" or "right" to indicate which camera
            x (int): X coordinate
            y (int): Y coordinate
            r (int): Radius
        """
        import time
        
        # Create a coordinate record
        record = (x, y, r, time.time())
        
        # Add to history
        self.coordinate_history[side].append(record)
        
        # Trim history if it exceeds maximum length
        if len(self.coordinate_history[side]) > self.max_history_length:
            self.coordinate_history[side].pop(0)
        
        # Only update Kalman filter if we're actively tracking (both sides detected)
        if self.detection_stats["is_tracking"]:
            # Update Kalman filter prediction
            self._update_kalman(side, x, y)
            logging.debug(f"Added {side} coordinate and updated Kalman: x={x}, y={y}, r={r}")
        else:
            logging.debug(f"Added {side} coordinate (no Kalman update yet): x={x}, y={y}, r={r}")
    
    def get_predictions(self):
        """
        Get the current Kalman predictions for both sides.
        
        Returns:
            tuple: (left_prediction, right_prediction) where each is (x, y, vx, vy) or None
        """
        # Only return predictions if Kalman is ready for each side
        left_pred = self.prediction["left"] if self.kalman_ready.get("left", False) else None
        right_pred = self.prediction["right"] if self.kalman_ready.get("right", False) else None
        return left_pred, right_pred
    
    def get_latest_coordinates(self):
        """
        Get the latest coordinates from both left and right images.
        
        Returns:
            tuple: (left_coords, right_coords) where each is (x, y, r) or None if not available
        """
        left_coords = None
        right_coords = None
        
        # Get latest left coordinates
        if self.coordinate_history["left"]:
            latest = self.coordinate_history["left"][-1]
            left_coords = (latest[0], latest[1], latest[2])
            
        # Get latest right coordinates
        if self.coordinate_history["right"]:
            latest = self.coordinate_history["right"][-1]
            right_coords = (latest[0], latest[1], latest[2])
            
        return left_coords, right_coords
    
    def clear_coordinate_history(self):
        """
        Clear the coordinate history.
        """
        self.coordinate_history["left"] = []
        self.coordinate_history["right"] = []
        logging.info("Coordinate history cleared")
    
    def reset_tracking(self):
        """
        Reset all tracking data including detection rate, coordinate history and predictions.
        Call this when user manually stops tracking or restarts application.
        """
        # Reset coordinate history
        self.clear_coordinate_history()
        
        # Reset detection stats
        self.detection_stats = {
            "first_detection_time": None,  # Time of first simultaneous detection
            "detection_count": 0,          # Number of successful detections
            "total_frames": 0,             # Total number of frames processed
            "is_tracking": False           # Whether we're currently tracking
        }
        
        # Reset circle detection results
        self.left_circles = None
        self.right_circles = None
        
        # Reset prediction data
        self.prediction["left"] = None
        self.prediction["right"] = None
        
        # Re-initialize Kalman filters
        self._init_kalman_filters()
        
        logging.info("Ball tracking data completely reset")
        
        # Emit signal with updated detection rate (0.0) and empty coordinates
        self.detection_updated.emit(0.0, None, None)
    
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
                return self.coordinate_history
            else:
                return {
                    "left": self.coordinate_history["left"][-count:] if self.coordinate_history["left"] else [],
                    "right": self.coordinate_history["right"][-count:] if self.coordinate_history["right"] else []
                }
        elif side in ["left", "right"]:
            if count is None:
                return self.coordinate_history[side]
            else:
                return self.coordinate_history[side][-count:] if self.coordinate_history[side] else []
        else:
            logging.error(f"Invalid side: {side}")
            return None

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
    
    def _fuse_coordinates(self, side, hsv_center=None, hough_center=None, kalman_pred=None, is_first_detection=False):
        """
        Fuse multiple coordinate estimates (HSV mask centroid, Hough circle center, Kalman prediction)
        to produce a more robust 2D pixel coordinate by averaging valid coordinates.
        
        Args:
            side (str): "left" or "right" to indicate which camera
            hsv_center (tuple, optional): HSV mask centroid (x, y)
            hough_center (tuple, optional): Hough circle center (x, y)
            kalman_pred (tuple, optional): Kalman prediction (x, y, vx, vy)
            is_first_detection (bool): Whether this is the first detection frame
            
        Returns:
            tuple: Fused 2D coordinates (x, y) or None if no valid coordinates
        """
        # Collect all valid coordinates
        valid_coords = []
        
        # Add HSV centroid if available
        if hsv_center is not None:
            valid_coords.append((hsv_center[0], hsv_center[1]))
        
        # Add Hough circle center if available
        if hough_center is not None:
            valid_coords.append((hough_center[0], hough_center[1]))
        
        # Add Kalman prediction if available, tracking is active, and not the first detection
        # For the first detection frame, we don't use Kalman prediction to avoid initial offset
        if kalman_pred is not None and self.detection_stats["is_tracking"] and not is_first_detection and self.kalman_update_counter[side] > 5:
            px, py, vx, vy = kalman_pred
            # Validate prediction inside image bounds
            img = self.left_image if side == "left" else self.right_image
            if img is not None:
                h, w = img.shape[:2]
                if 0 <= px < w and 0 <= py < h:
                    valid_coords.append((px, py))
                else:
                    logging.debug(f"Ignored out-of-bounds Kalman prediction for {side}: ({px:.2f}, {py:.2f})")
            else:
                logging.debug(f"No image available to validate Kalman prediction for {side}")
        
        # Use the utility function to fuse coordinates
        fused = fuse_coordinates(valid_coords)
        
        if fused:
            logging.debug(f"Fused {len(valid_coords)} coordinates for {side}: {fused[0]:.2f}, {fused[1]:.2f}")
            
        return fused
            
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
            # Set default folder path if not provided
            if folder_path is None:
                folder_path = os.path.join(os.getcwd(), "tracking_data")
            
            # Create folder if it doesn't exist
            os.makedirs(folder_path, exist_ok=True)
            
            # Set default filename if not provided (use timestamp)
            if filename is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"tracking_data_{timestamp}.json"
            elif not filename.endswith('.json'):
                filename = f"{filename}.json"
            
            # Combine folder path and filename
            file_path = os.path.join(folder_path, filename)
            
            # Prepare data dictionary
            tracking_data = {
                "timestamp": time.time(),
                "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
                "detection_rate": self.get_detection_rate(),
                "tracking_active": self.detection_stats["is_tracking"],
                "frames_total": self.detection_stats["total_frames"],
                "detections_count": self.detection_stats["detection_count"],
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
            if self.left_mask is not None:
                # Get HSV mask centroid
                left_hsv_center = self._compute_mask_centroid(self.left_mask)
                
                # Get latest Hough circle center if available
                left_hough_center = None
                if self.left_circles:
                    left_hough_center = (self.left_circles[0][0], self.left_circles[0][1])
                
                # Get Kalman prediction if available
                left_kalman_pred = self.prediction["left"]
                
                # Fuse coordinates
                left_fused = self._fuse_coordinates(
                    "left", 
                    left_hsv_center, 
                    left_hough_center, 
                    left_kalman_pred,
                    self.detection_stats["is_tracking"] == False
                )
                
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
                
                # Add Kalman predictions only if valid (non-zero)
                if left_kalman_pred and not (left_kalman_pred[0] == 0.0 and left_kalman_pred[1] == 0.0):
                    tracking_data["coordinate_data"]["left"]["kalman_predictions"].append({
                        "x": float(left_kalman_pred[0]),
                        "y": float(left_kalman_pred[1]),
                        "vx": float(left_kalman_pred[2]),
                        "vy": float(left_kalman_pred[3])
                    })
                
                if left_fused:
                    tracking_data["coordinate_data"]["left"]["fused_centers"].append({
                        "x": float(left_fused[0]),
                        "y": float(left_fused[1])
                    })
            
            # Process right side (similar to left)
            if self.right_mask is not None:
                # Get HSV mask centroid
                right_hsv_center = self._compute_mask_centroid(self.right_mask)
                
                # Get latest Hough circle center if available
                right_hough_center = None
                if self.right_circles:
                    right_hough_center = (self.right_circles[0][0], self.right_circles[0][1])
                
                # Get Kalman prediction if available
                right_kalman_pred = self.prediction["right"]
                
                # Fuse coordinates
                right_fused = self._fuse_coordinates(
                    "right", 
                    right_hsv_center, 
                    right_hough_center, 
                    right_kalman_pred,
                    self.detection_stats["is_tracking"] == False
                )
                
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
                
                # Add Kalman predictions only if valid (non-zero)
                if right_kalman_pred and not (right_kalman_pred[0] == 0.0 and right_kalman_pred[1] == 0.0):
                    tracking_data["coordinate_data"]["right"]["kalman_predictions"].append({
                        "x": float(right_kalman_pred[0]),
                        "y": float(right_kalman_pred[1]),
                        "vx": float(right_kalman_pred[2]),
                        "vy": float(right_kalman_pred[3])
                    })
                
                if right_fused:
                    tracking_data["coordinate_data"]["right"]["fused_centers"].append({
                        "x": float(right_fused[0]),
                        "y": float(right_fused[1])
                    })
            
            # Convert coordinate history to serializable format
            for side in ["left", "right"]:
                history = self.coordinate_history[side]
                for entry in history:
                    # Each entry is (x, y, r, timestamp)
                    tracking_data["coordinate_history"][side].append({
                        "x": float(entry[0]),
                        "y": float(entry[1]),
                        "radius": float(entry[2]),
                        "timestamp": entry[3]
                    })
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(tracking_data, f, indent=2)
            
            logging.info(f"Tracking data saved to {file_path}")
            return file_path
            
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
            # Set default folder path if not provided
            if folder_path is None:
                folder_path = os.path.join(os.getcwd(), "tracking_data")
            
            # Create folder if it doesn't exist
            os.makedirs(folder_path, exist_ok=True)
            
            # Create filename using frame number to ensure overwrite of same frame data
            filename = f"frame_{frame_number:06d}.json"
            
            # Combine folder path and filename
            file_path = os.path.join(folder_path, filename)
            
            # Prepare data dictionary
            frame_data = {
                "frame_number": frame_number,
                "timestamp": time.time(),
                "tracking_active": self.detection_stats["is_tracking"],
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
            if self.left_mask is not None:
                # Get HSV mask centroid
                left_hsv_center = self._compute_mask_centroid(self.left_mask)
                if left_hsv_center:
                    frame_data["left"]["hsv_center"] = {
                        "x": float(left_hsv_center[0]),
                        "y": float(left_hsv_center[1])
                    }
                
                # Get latest Hough circle center if available
                if self.left_circles:
                    left_hough_center = (self.left_circles[0][0], self.left_circles[0][1])
                    frame_data["left"]["hough_center"] = {
                        "x": float(left_hough_center[0]),
                        "y": float(left_hough_center[1]),
                        "radius": float(self.left_circles[0][2])
                    }
                
                # Get Kalman prediction if available
                left_pred = self.prediction.get("left")
                if left_pred is not None:
                    px, py, vx, vy = left_pred
                    # Skip invalid zero predictions
                    if not (px == 0.0 and py == 0.0):
                        frame_data["left"]["kalman_prediction"] = {
                            "x": float(px),
                            "y": float(py),
                            "vx": float(vx),
                            "vy": float(vy)
                        }
                    
                # Fuse coordinates
                left_fused = self._fuse_coordinates(
                    "left", 
                    left_hsv_center, 
                    left_hough_center if self.left_circles else None, 
                    left_pred,
                    self.detection_stats["is_tracking"] == False
                )
                
                if left_fused:
                    frame_data["left"]["fused_center"] = {
                        "x": float(left_fused[0]), 
                        "y": float(left_fused[1])
                    }
            
            # Process right side (similar to left)
            if self.right_mask is not None:
                # Get HSV mask centroid
                right_hsv_center = self._compute_mask_centroid(self.right_mask)
                if right_hsv_center:
                    frame_data["right"]["hsv_center"] = {
                        "x": float(right_hsv_center[0]),
                        "y": float(right_hsv_center[1])
                    }
                
                # Get latest Hough circle center if available
                if self.right_circles:
                    right_hough_center = (self.right_circles[0][0], self.right_circles[0][1])
                    frame_data["right"]["hough_center"] = {
                        "x": float(right_hough_center[0]), 
                        "y": float(right_hough_center[1]),
                        "radius": float(self.right_circles[0][2])
                    }
                
                # Get Kalman prediction if available
                right_pred = self.prediction.get("right")
                if right_pred is not None:
                    px, py, vx, vy = right_pred
                    # Skip invalid zero predictions
                    if not (px == 0.0 and py == 0.0):
                        frame_data["right"]["kalman_prediction"] = {
                            "x": float(px),
                            "y": float(py),
                            "vx": float(vx),
                            "vy": float(vy)
                        }
                    
                # Fuse coordinates
                right_fused = self._fuse_coordinates(
                    "right", 
                    right_hsv_center, 
                    right_hough_center if self.right_circles else None, 
                    right_pred,
                    self.detection_stats["is_tracking"] == False
                )
                
                if right_fused:
                    frame_data["right"]["fused_center"] = {
                        "x": float(right_fused[0]), 
                        "y": float(right_fused[1])
                    }
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(frame_data, f, indent=2)
            
            logging.debug(f"Frame {frame_number} tracking data saved to {file_path}")
            return file_path
            
        except Exception as e:
            logging.error(f"Error saving frame tracking data: {e}")
            return None
    
    def load_coordinate_history(self, filename):
        """
        Load the coordinate history from a JSON file.
        
        Args:
            filename (str): Path to the input JSON file
        """
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                history = json.load(f)
            self.coordinate_history = history
            logging.info(f"Coordinate history loaded from {filename}")
        else:
            logging.error(f"File {filename} does not exist")
    
    def initialize_xml_tracking(self, folder_name):
        """
        Initialize XML tracking data structure for a new folder.
        If an existing XML file exists, it will be loaded to continue tracking.
        
        Args:
            folder_name (str): Name of the folder being processed
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Set up file path
            tracking_folder = os.path.join(os.getcwd(), "tracking_data", folder_name)
            os.makedirs(tracking_folder, exist_ok=True)
            xml_file_path = os.path.join(tracking_folder, "tracking_data.xml")
            
            # Check if the file already exists
            if os.path.exists(xml_file_path):
                try:
                    # Try to parse the existing file
                    tree = ET.parse(xml_file_path)
                    self.xml_root = tree.getroot()
                    
                    # Get existing frame count
                    image_elements = self.xml_root.findall("Image")
                    frame_count = len(image_elements)
                    
                    # Store the current folder
                    self.current_folder = folder_name
                    
                    logging.info(f"Loaded existing XML tracking data for folder '{folder_name}' with {frame_count} frames")
                    
                    # Update timestamp to indicate this is a resumed session
                    self.xml_root.set("resumed", str(time.time()))
                    self.xml_root.set("resume_time", time.strftime("%Y-%m-%d %H:%M:%S"))
                    
                    return True
                except Exception as e:
                    logging.warning(f"Failed to load existing XML file, creating new one: {e}")
                    # Fall through to create new file
            
            # Create new root element
            self.xml_root = ET.Element("TrackingData")
            self.xml_root.set("folder", folder_name)
            self.xml_root.set("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
            self.xml_root.set("created", str(time.time()))
            
            # Store the current folder
            self.current_folder = folder_name
            
            # Immediately save to disk to ensure file exists
            self.save_xml_tracking_data(tracking_folder)
            
            logging.info(f"Initialized new XML tracking data for folder: {folder_name}")
            return True
            
        except Exception as e:
            logging.error(f"Error initializing XML tracking: {e}")
            return False
    
    def save_frame_to_xml(self, frame_number, frame_name=None):
        """
        Save the current frame's tracking data to the XML structure.
        
        Args:
            frame_number (int): Frame number
            frame_name (str, optional): Frame filename if available
            
        Returns:
            bool: Success or failure
        """
        if self.xml_root is None:
            logging.error("XML tracking not initialized. Call initialize_xml_tracking first.")
            return False
            
        try:
            # Create frame/image element
            image_elem = ET.SubElement(self.xml_root, "Image")
            image_elem.set("number", str(frame_number))
            if frame_name:
                image_elem.set("name", frame_name)
            image_elem.set("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
            image_elem.set("tracking_active", str(self.detection_stats["is_tracking"]))
            
            # Process left camera data
            left_elem = ET.SubElement(image_elem, "Left")
            
            # HSV mask centroid
            if self.left_mask is not None:
                left_hsv_center = self._compute_mask_centroid(self.left_mask)
                if left_hsv_center:
                    hsv_elem = ET.SubElement(left_elem, "HSV")
                    hsv_elem.set("x", str(float(left_hsv_center[0])))
                    hsv_elem.set("y", str(float(left_hsv_center[1])))
            
            # Hough circle
            if self.left_circles:
                hough_elem = ET.SubElement(left_elem, "Hough")
                hough_elem.set("x", str(float(self.left_circles[0][0])))
                hough_elem.set("y", str(float(self.left_circles[0][1])))
                hough_elem.set("radius", str(float(self.left_circles[0][2])))
            
            # Kalman prediction
            left_pred = self.prediction.get("left")
            if left_pred is not None:
                px, py, vx, vy = left_pred
                # Skip invalid zero predictions
                if not (px == 0.0 and py == 0.0):
                    kalman_elem = ET.SubElement(left_elem, "Kalman")
                    kalman_elem.set("x", str(px))
                    kalman_elem.set("y", str(py))
            
            # Fused coordinate
            left_hsv_center = self._compute_mask_centroid(self.left_mask) if self.left_mask is not None else None
            left_hough_center = (self.left_circles[0][0], self.left_circles[0][1]) if self.left_circles else None
            left_fused = self._fuse_coordinates(
                "left", 
                left_hsv_center, 
                left_hough_center, 
                left_pred,
                self.detection_stats["is_tracking"] == False
            )
            if left_fused:
                fused_elem = ET.SubElement(left_elem, "Fused")
                fused_elem.set("x", str(float(left_fused[0])))
                fused_elem.set("y", str(float(left_fused[1])))
            
            # Process right camera data
            right_elem = ET.SubElement(image_elem, "Right")
            
            # HSV mask centroid
            if self.right_mask is not None:
                right_hsv_center = self._compute_mask_centroid(self.right_mask)
                if right_hsv_center:
                    hsv_elem = ET.SubElement(right_elem, "HSV")
                    hsv_elem.set("x", str(float(right_hsv_center[0])))
                    hsv_elem.set("y", str(float(right_hsv_center[1])))
            
            # Hough circle
            if self.right_circles:
                hough_elem = ET.SubElement(right_elem, "Hough")
                hough_elem.set("x", str(float(self.right_circles[0][0])))
                hough_elem.set("y", str(float(self.right_circles[0][1])))
                hough_elem.set("radius", str(float(self.right_circles[0][2])))
            
            # Kalman prediction
            right_pred = self.prediction.get("right")
            if right_pred is not None:
                px, py, vx, vy = right_pred
                # Skip invalid zero predictions
                if not (px == 0.0 and py == 0.0):
                    kalman_elem = ET.SubElement(right_elem, "Kalman")
                    kalman_elem.set("x", str(px))
                    kalman_elem.set("y", str(py))
            
            # Fused coordinate
            right_hsv_center = self._compute_mask_centroid(self.right_mask) if self.right_mask is not None else None
            right_hough_center = (self.right_circles[0][0], self.right_circles[0][1]) if self.right_circles else None
            right_fused = self._fuse_coordinates(
                "right", 
                right_hsv_center, 
                right_hough_center, 
                right_pred,
                self.detection_stats["is_tracking"] == False
            )
            if right_fused:
                fused_elem = ET.SubElement(right_elem, "Fused")
                fused_elem.set("x", str(float(right_fused[0])))
                fused_elem.set("y", str(float(right_fused[1])))
            
            logging.debug(f"Added frame {frame_number} to XML tracking data")
            # Real-time update: write XML tracking data file after each frame
            self.save_xml_tracking_data()
            return True
            
        except Exception as e:
            logging.error(f"Error saving frame to XML: {e}")
            return False
    
    def save_xml_tracking_data(self, folder_path=None):
        """
        Save the XML tracking data to a file.
        
        Args:
            folder_path (str, optional): Path to the output folder. 
                                         Default is 'tracking_data/{current_folder}'.
            
        Returns:
            str: Path to the saved file or None if failed
        """
        if self.xml_root is None:
            logging.error("XML tracking not initialized. Call initialize_xml_tracking first.")
            return None
            
        try:
            # Set default folder path if not provided
            if folder_path is None:
                if self.current_folder is None:
                    folder_path = os.path.join(os.getcwd(), "tracking_data", "default")
                else:
                    folder_path = os.path.join(os.getcwd(), "tracking_data", self.current_folder)
            
            # Create folder if it doesn't exist
            os.makedirs(folder_path, exist_ok=True)
            
            # Create the output file path
            output_file = os.path.join(folder_path, "tracking_data.xml")
            
            # Remove zero Kalman tags (invalid predictions)
            for image_elem in self.xml_root.findall("Image"):
                for side_tag in ("Left", "Right"):
                    side_elem = image_elem.find(side_tag)
                    if side_elem is not None:
                        for kalman_elem in list(side_elem.findall("Kalman")):
                            try:
                                x_val = float(kalman_elem.get("x", 0))
                                y_val = float(kalman_elem.get("y", 0))
                            except (ValueError, TypeError):
                                continue
                            if x_val == 0.0 and y_val == 0.0:
                                side_elem.remove(kalman_elem)
            
            # Remove any existing Statistics elements to avoid duplicates
            for existing_stats in self.xml_root.findall("Statistics"):
                self.xml_root.remove(existing_stats)
            
            # Add summary statistics
            stats_elem = ET.SubElement(self.xml_root, "Statistics")
            stats_elem.set("detection_rate", str(self.get_detection_rate()))
            stats_elem.set("frames_total", str(self.detection_stats["total_frames"]))
            stats_elem.set("detections_count", str(self.detection_stats["detection_count"]))
            stats_elem.set("tracking_active", str(self.detection_stats["is_tracking"]))
            
            # Create XML tree and write to file
            tree = ET.ElementTree(self.xml_root)
            
            # Use minidom to pretty print the XML
            import xml.dom.minidom
            rough_string = ET.tostring(self.xml_root, 'utf-8')
            reparsed = xml.dom.minidom.parseString(rough_string)
            pretty_xml = reparsed.toprettyxml(indent="  ")
            # Remove blank lines to avoid excessive empty lines
            pretty_lines = [line for line in pretty_xml.split('\n') if line.strip()]
            pretty_xml = '\n'.join(pretty_lines)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(pretty_xml)
            
            logging.info(f"Saved XML tracking data to {output_file}")
            return output_file
            
        except Exception as e:
            logging.error(f"Error saving XML tracking data: {e}")
            return None 