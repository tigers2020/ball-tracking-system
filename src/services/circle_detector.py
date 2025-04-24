#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Circle Detector Module
This module contains the CircleDetector class for detecting circles in images.
"""

import logging
import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional, Any


class CircleDetector:
    """
    Service class for detecting circles in images using Hough Circle Transform.
    """

    def __init__(self, hough_settings: Dict[str, Any]):
        """
        Initialize the circle detector with provided Hough Circle settings.
        
        Args:
            hough_settings: Dictionary containing Hough Circle parameters
                - dp: Inverse ratio of the accumulator resolution to the image resolution
                - min_dist: Minimum distance between the centers of detected circles
                - param1: Higher threshold for edge detection
                - param2: Accumulator threshold for circle detection
                - min_radius: Minimum radius of circles to detect
                - max_radius: Maximum radius of circles to detect
                - adaptive: Whether to use ROI-adaptive parameter scaling
        """
        self.hough_settings = hough_settings.copy()
        self.adaptive = hough_settings.get('adaptive', False)
        logging.info(f"Circle detector initialized with settings: {self.hough_settings}")

    def update_settings(self, hough_settings: Dict[str, Any]) -> None:
        """
        Update the Hough Circle settings.
        
        Args:
            hough_settings: Dictionary containing Hough Circle parameters
        """
        self.hough_settings = hough_settings.copy()
        self.adaptive = hough_settings.get('adaptive', False)
        logging.info(f"Circle detector settings updated: {self.hough_settings}")

    def detect_circles(self, img: np.ndarray, roi: Optional[Dict[str, int]] = None, 
                    hsv_center: Optional[Tuple[int, int]] = None, 
                    kalman_pred: Optional[Tuple[float, float, float, float]] = None, 
                    reset_mode: bool = False, 
                    side: str = None,
                    visualize: bool = False) -> Dict[str, Any]:
        """
        Detect circles in an image using Hough Circle Transform.
        
        Args:
            img: Input grayscale or BGR image
            roi: Region of interest dict with 'x', 'y', 'width', 'height' or None for the whole image
            hsv_center: Optional HSV mask centroid (x, y)
            kalman_pred: Optional Kalman filter prediction (x, y, vx, vy)
            reset_mode: Whether to ignore previous tracking (reset mode)
            side: Which side the detection is for ('left' or 'right')
            visualize: Whether to include visualization in the output (deprecated, use visualization module instead)
            
        Returns:
            Dictionary containing detection results:
            - 'circles': List of (x, y, radius) tuples or None
            - 'roi': ROI dictionary or None
            - 'hsv_center': HSV center point (x, y) or None
            - 'kalman_pred': Kalman prediction (x, y, vx, vy) or None
            - 'image': Output image with visualizations (only if visualize=True)
        """
        try:
            # Convert to grayscale if it's a color image
            if len(img.shape) > 2:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Create output data dictionary
            result = {
                'circles': None,
                'roi': None,
                'hsv_center': hsv_center,
                'kalman_pred': kalman_pred
            }
            
            # Create a copy of the original image for drawing if visualization is requested
            output_img = None
            if visualize:
                output_img = img.copy() if len(img.shape) > 2 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            # Extract ROI if provided
            if roi is not None:
                # Extract ROI coordinates from dictionary
                x = roi.get('x', 0)
                y = roi.get('y', 0)
                w = roi.get('width', gray.shape[1])
                h = roi.get('height', gray.shape[0])
                
                # Ensure within image bounds
                x = max(0, min(x, gray.shape[1] - 1))
                y = max(0, min(y, gray.shape[0] - 1))
                w = max(1, min(w, gray.shape[1] - x))
                h = max(1, min(h, gray.shape[0] - y))
                
                roi_img = gray[y:y+h, x:x+w]
                
                # Store adjusted ROI in results
                result['roi'] = {'x': x, 'y': y, 'width': w, 'height': h}
                logging.debug(f"Using ROI at x={x}, y={y}, w={w}, h={h}")
            else:
                roi_img = gray
                x, y = 0, 0
                w, h = gray.shape[1], gray.shape[0]
            
            # Copy current settings to avoid modifying the original
            settings = self.hough_settings.copy()
            
            # Apply ROI-adaptive parameter scaling if enabled
            if self.adaptive and roi is not None:
                # Calculate ROI size ratio compared to full image
                roi_area_ratio = (w * h) / (gray.shape[0] * gray.shape[1])
                
                # Scale parameters based on ROI size
                # For smaller ROIs: increase dp (more precise), decrease min_dist (closer circles)
                # For larger ROIs: keep dp closer to original, keep min_dist larger
                
                # Scale dp inversely with ROI area ratio (smaller ROI = higher dp)
                settings['dp'] = max(1.0, settings['dp'] * (1.0 + (1.0 - roi_area_ratio) * 0.5))
                
                # Scale min_dist proportional to ROI dimensions
                settings['min_dist'] = max(5, int(settings['min_dist'] * np.sqrt(roi_area_ratio) * 1.2))
                
                # Scale param2 (accumulator threshold) based on ROI size
                # For smaller ROIs, lower the threshold to detect circles more easily
                settings['param2'] = max(5, int(settings['param2'] * (0.8 + roi_area_ratio * 0.4)))
                
                logging.debug(f"ROI-adaptive parameters: dp={settings['dp']:.2f}, min_dist={settings['min_dist']}, "
                             f"param2={settings['param2']}, roi_ratio={roi_area_ratio:.3f}")
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(roi_img, (5, 5), 0)
            
            # Detect circles using Hough Circle Transform
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=settings['dp'],
                minDist=settings['min_dist'],
                param1=settings['param1'],
                param2=settings['param2'],
                minRadius=settings['min_radius'],
                maxRadius=settings['max_radius']
            )
            
            # Process detected circles
            circles_list = None
            
            if circles is not None:
                # Convert to integer coordinates
                circles = np.uint16(np.around(circles))
                
                # Prepare circles list
                circles_list = []
                
                # Process all circles
                for i in range(len(circles[0])):
                    circle = circles[0][i]
                    center_x = int(circle[0]) + x
                    center_y = int(circle[1]) + y
                    radius = int(circle[2])
                    
                    # Store circle data
                    circles_list.append((center_x, center_y, radius))
                    
                    if i == 0:
                        logging.debug(f"Detected circle: center=({center_x}, {center_y}), radius={radius}")
                
                if side:
                    logging.debug(f"Detected {len(circles_list)} circles on {side} side")
                
                # Store circles in result
                result['circles'] = circles_list
            else:
                logging.debug(f"No circles detected{' on ' + side if side else ''}")
            
            # If visualization is requested (legacy behavior), add visualization to result
            if visualize and output_img is not None:
                # This is deprecated behavior, use visualization modules instead
                logging.warning("Visualization in CircleDetector is deprecated. Use visualization modules instead.")
                
                # Legacy visualization code left for backward compatibility
                if result['roi'] is not None:
                    roi_dict = result['roi']
                    cv2.rectangle(output_img, 
                                 (roi_dict['x'], roi_dict['y']), 
                                 (roi_dict['x'] + roi_dict['width'], roi_dict['y'] + roi_dict['height']), 
                                 (255, 255, 0), 2)
                
                if hsv_center is not None:
                    cv2.circle(output_img, hsv_center, 5, (0, 255, 255), -1)
                
                if kalman_pred is not None:
                    pred_x, pred_y = int(kalman_pred[0]), int(kalman_pred[1])
                    if 0 <= pred_x < output_img.shape[1] and 0 <= pred_y < output_img.shape[0]:
                        cv2.circle(output_img, (pred_x, pred_y), 5, (255, 0, 255), -1)
                        # Draw velocity vector
                        vel_length = np.sqrt(kalman_pred[2]**2 + kalman_pred[3]**2)
                        if vel_length > 0.5:  # Only draw if significant velocity
                            end_x = int(pred_x + kalman_pred[2] * 3)  # 3 frames prediction
                            end_y = int(pred_y + kalman_pred[3] * 3)
                            cv2.arrowedLine(output_img, (pred_x, pred_y), (end_x, end_y), (255, 0, 255), 2)
                
                if circles_list:
                    for i, (center_x, center_y, radius) in enumerate(circles_list):
                        # Use different colors for the best circle (first one) and others
                        color = (0, 255, 0) if i == 0 else (255, 0, 0)
                        
                        # Draw the circle on the output image
                        cv2.circle(output_img, (center_x, center_y), radius, color, 2)
                        cv2.circle(output_img, (center_x, center_y), 2, (0, 0, 255), 3)
                
                result['image'] = output_img
            
            return result
            
        except Exception as e:
            logging.error(f"Error in circle detection{' on ' + side if side else ''}: {e}")
            return {'circles': None, 'roi': roi, 'hsv_center': hsv_center, 'kalman_pred': kalman_pred} 