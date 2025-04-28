#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Coordinate Combiner Service.
This module contains the CoordinateCombiner class for combining detection results
from different sources (HSV, Hough, Kalman) and calculating 3D coordinates.
"""

import logging
import numpy as np
from typing import Tuple, Dict, List, Optional, Any, Union

from src.utils.constants import STEREO


class CoordinateCombiner:
    """
    Service for combining 2D coordinates and calculating 3D coordinates.
    Combines results from different detection methods (HSV, Hough, Kalman)
    to produce more stable tracking results.
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the coordinate combiner.
        
        Args:
            config_manager: ConfigManager instance for accessing camera settings
        """
        self.config_manager = config_manager
        self.camera_settings = {}
        self.coordinate_settings = {}
        
        # Update settings if config manager is provided
        if config_manager:
            self.update_settings()
    
    def update_settings(self):
        """Update settings from the config manager."""
        if self.config_manager:
            self.camera_settings = self.config_manager.get_camera_settings()
            self.coordinate_settings = self.config_manager.get_coordinate_settings()
            logging.debug("Updated coordinate combiner settings")
    
    def combine_2d_coordinates(self, 
                              hsv_point: Optional[Tuple[float, float]] = None,
                              hough_point: Optional[Tuple[float, float]] = None,
                              kalman_point: Optional[Tuple[float, float]] = None,
                              weights: Optional[Dict[str, float]] = None
                              ) -> Optional[Tuple[float, float]]:
        """
        Combine 2D coordinates from different sources using weighted average.
        
        Args:
            hsv_point: (x, y) coordinates from HSV detection
            hough_point: (x, y) coordinates from Hough Circle detection
            kalman_point: (x, y) coordinates from Kalman filter prediction
            weights: Dictionary of weights for each source
                {'hsv': 0.2, 'hough': 0.3, 'kalman': 0.5}
                Default weights are 0.2, 0.3, 0.5 respectively
                
        Returns:
            Combined (x, y) coordinates or None if no valid inputs
        """
        # Default weights
        if weights is None:
            weights = {'hsv': 0.2, 'hough': 0.3, 'kalman': 0.5}
        
        # Points and their corresponding weights
        points = []
        point_weights = []
        
        # Add each valid point with its weight
        if hsv_point is not None:
            points.append(hsv_point)
            point_weights.append(weights.get('hsv', 0.2))
        
        if hough_point is not None:
            points.append(hough_point)
            point_weights.append(weights.get('hough', 0.3))
        
        if kalman_point is not None:
            points.append(kalman_point)
            point_weights.append(weights.get('kalman', 0.5))
        
        # If no valid points, return None
        if not points:
            return None
        
        # If only one point, return it directly
        if len(points) == 1:
            return points[0]
        
        # Normalize weights
        total_weight = sum(point_weights)
        if total_weight > 0:
            point_weights = [w / total_weight for w in point_weights]
        else:
            # If all weights are zero, use equal weights
            point_weights = [1.0 / len(points)] * len(points)
        
        # Calculate weighted average
        x = sum(p[0] * w for p, w in zip(points, point_weights))
        y = sum(p[1] * w for p, w in zip(points, point_weights))
        
        return (x, y)
    
    def triangulate_3d_position(self,
                               left_point: Optional[Tuple[float, float]],
                               right_point: Optional[Tuple[float, float]],
                               camera_settings: Optional[Dict[str, Any]] = None
                               ) -> Optional[Tuple[float, float, float]]:
        """
        Calculate 3D position using stereo triangulation.
        
        Args:
            left_point: (x, y) coordinates in left image
            right_point: (x, y) coordinates in right image
            camera_settings: Override camera settings
                
        Returns:
            (x, y, z) coordinates in world space (meters) or None if triangulation fails
        """
        # Check if both points are valid
        if left_point is None or right_point is None:
            return None
        
        # Use provided camera settings or instance settings
        settings = camera_settings if camera_settings else self.camera_settings
        if not settings:
            logging.warning("No camera settings available for triangulation")
            return None
        
        try:
            # Extract camera parameters
            baseline = settings.get('baseline_m', STEREO.DEFAULT_BASELINE_M)
            focal_length_mm = settings.get('focal_length_mm', 50.0)
            sensor_width_mm = settings.get('sensor_width_mm', 36.0)
            
            # Get image dimensions from calibration points if available
            if self.config_manager:
                calib_points = self.config_manager.get_calibration_points()
                image_dims = calib_points.get('left_image_size', {'width': 640, 'height': 480})
                image_width = image_dims.get('width', 640)
            else:
                # Default to standard dimensions
                image_width = 640
            
            # Calculate disparity
            xl, yl = left_point
            xr, yr = right_point
            
            # Calculate pixel-to-mm conversion factor
            # This could be more accurately derived from camera calibration
            if sensor_width_mm > 0 and image_width > 0:
                pixel_to_mm = sensor_width_mm / image_width
            else:
                pixel_to_mm = 0.01  # Default fallback
            
            # Calculate disparity in mm
            disparity_px = xl - xr
            disparity_mm = disparity_px * pixel_to_mm
            
            # Basic triangulation formula: Z = (baseline * focal_length) / disparity
            if abs(disparity_mm) > 0.1:  # Avoid division by zero or very small disparities
                # Z is depth (distance from camera baseline)
                z = (baseline * focal_length_mm) / disparity_mm
                
                # X is horizontal position (right is positive)
                x = (xl - image_width/2) * pixel_to_mm * z / focal_length_mm
                
                # Y is vertical position (up is positive)
                y = (image_width/2 - yl) * pixel_to_mm * z / focal_length_mm
                
                # Apply coordinate rotation if needed
                # This would be a more complex transformation based on camera orientation
                rotation = self.coordinate_settings.get('rotation', {'x': 0.0, 'y': 0.0, 'z': 0.0})
                
                # For now, just return the basic coordinates
                return (x, y, z)
            else:
                logging.warning(f"Disparity too small for reliable triangulation: {disparity_mm}mm")
                return None
            
        except Exception as e:
            logging.error(f"Error in triangulation: {e}")
            return None
    
    def combine_and_triangulate(self,
                               left_hsv: Optional[Tuple[float, float]] = None,
                               left_hough: Optional[Tuple[float, float]] = None,
                               left_kalman: Optional[Tuple[float, float]] = None,
                               right_hsv: Optional[Tuple[float, float]] = None,
                               right_hough: Optional[Tuple[float, float]] = None,
                               right_kalman: Optional[Tuple[float, float]] = None,
                               weights: Optional[Dict[str, float]] = None
                               ) -> Dict[str, Any]:
        """
        Combine 2D coordinates and calculate 3D position.
        
        Args:
            left_hsv: HSV detection point in left image
            left_hough: Hough detection point in left image
            left_kalman: Kalman prediction point in left image
            right_hsv: HSV detection point in right image
            right_hough: Hough detection point in right image
            right_kalman: Kalman prediction point in right image
            weights: Dictionary of weights for each source
            
        Returns:
            Dictionary with combined results:
            {
                'left_2d': (x, y) or None,
                'right_2d': (x, y) or None,
                'world_3d': (x, y, z) or None,
                'status': 'tracking' or 'lost' or 'partial',
                'confidence': float between 0 and 1
            }
        """
        # Combine left and right 2D coordinates
        left_combined = self.combine_2d_coordinates(left_hsv, left_hough, left_kalman, weights)
        right_combined = self.combine_2d_coordinates(right_hsv, right_hough, right_kalman, weights)
        
        # Calculate 3D position
        world_position = self.triangulate_3d_position(left_combined, right_combined)
        
        # Determine status and confidence
        if left_combined and right_combined:
            status = 'tracking'
            confidence = 1.0
        elif left_combined or right_combined:
            status = 'partial'
            confidence = 0.5
        else:
            status = 'lost'
            confidence = 0.0
        
        # Return combined results
        return {
            'left_2d': left_combined,
            'right_2d': right_combined,
            'world_3d': world_position,
            'status': status,
            'confidence': confidence
        } 