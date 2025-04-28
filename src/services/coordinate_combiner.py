#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Coordinate Combiner service module.
This module contains the CoordinateCombiner class for combining coordinates from different sources.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import time


class CoordinateCombiner:
    """
    Service class for combining and processing coordinates from different detection methods.
    Combines HSV, Hough Circle, and Kalman filter coordinates to produce final 2D and 3D positions.
    """
    
    def __init__(self, triangulation_service=None):
        """
        Initialize the coordinate combiner.
        
        Args:
            triangulation_service: Service for performing 3D triangulation
        """
        self.triangulation_service = triangulation_service
        
        # Initialize tracking performance metrics
        self.processing_start_time = 0
        self.processing_time = 0
        self.fps_history = []
        self.max_fps_history = 30  # Number of frames to average FPS over
        
        # Initialize coordinate weights (can be adjusted based on confidence)
        self.weights = {
            'hsv': 0.3,      # Weight for HSV detection
            'hough': 0.4,    # Weight for Hough Circle detection
            'kalman': 0.3    # Weight for Kalman filter prediction
        }
        
        # Tracking status
        self.tracking_status = "Waiting"
        self.confidence = 0.0
        self.last_3d_position = (0.0, 0.0, 0.0)
    
    def set_triangulation_service(self, service):
        """
        Set the triangulation service.
        
        Args:
            service: Service for performing 3D triangulation
        """
        self.triangulation_service = service
    
    def set_detection_weights(self, hsv_weight: float, hough_weight: float, kalman_weight: float):
        """
        Set weights for different detection methods.
        
        Args:
            hsv_weight (float): Weight for HSV detection (0.0-1.0)
            hough_weight (float): Weight for Hough Circle detection (0.0-1.0)
            kalman_weight (float): Weight for Kalman filter prediction (0.0-1.0)
        """
        # Ensure weights sum to 1.0
        total = hsv_weight + hough_weight + kalman_weight
        if total <= 0:
            # Default to equal weights if all are zero or negative
            self.weights = {'hsv': 0.33, 'hough': 0.34, 'kalman': 0.33}
            return
            
        # Normalize weights to sum to 1.0
        self.weights = {
            'hsv': hsv_weight / total,
            'hough': hough_weight / total,
            'kalman': kalman_weight / total
        }
        
        logging.info(f"Updated detection weights: HSV={self.weights['hsv']:.2f}, "
                    f"Hough={self.weights['hough']:.2f}, Kalman={self.weights['kalman']:.2f}")
    
    def start_processing(self):
        """Start the processing timer."""
        self.processing_start_time = time.time()
    
    def end_processing(self):
        """
        End the processing timer and update performance metrics.
        
        Returns:
            float: Processing time in milliseconds
        """
        if self.processing_start_time > 0:
            self.processing_time = (time.time() - self.processing_start_time) * 1000  # ms
            
            # Update FPS history (1000 ms / processing_time)
            if self.processing_time > 0:
                fps = 1000.0 / self.processing_time
                self.fps_history.append(fps)
                if len(self.fps_history) > self.max_fps_history:
                    self.fps_history.pop(0)
            
            self.processing_start_time = 0
            return self.processing_time
        
        return 0.0
    
    def get_fps(self):
        """
        Calculate the average FPS over recent frames.
        
        Returns:
            float: Current FPS (frames per second)
        """
        if not self.fps_history:
            return 0.0
        
        return sum(self.fps_history) / len(self.fps_history)
    
    def combine_coordinates(self, hsv_coords, hough_coords, kalman_coords):
        """
        Combine coordinates from different detection methods using weighted average.
        
        Args:
            hsv_coords (tuple or None): (x, y) from HSV detection
            hough_coords (tuple or None): (x, y) from Hough Circle detection
            kalman_coords (tuple or None): (x, y) from Kalman filter prediction
            
        Returns:
            tuple or None: Combined (x, y) coordinates or None if no valid coordinates
        """
        # Initialize lists to hold valid x and y values
        valid_x = []
        valid_y = []
        valid_weights = []
        
        # Add HSV coordinates if valid
        if hsv_coords is not None:
            valid_x.append(hsv_coords[0] * self.weights['hsv'])
            valid_y.append(hsv_coords[1] * self.weights['hsv'])
            valid_weights.append(self.weights['hsv'])
        
        # Add Hough Circle coordinates if valid
        if hough_coords is not None:
            valid_x.append(hough_coords[0] * self.weights['hough'])
            valid_y.append(hough_coords[1] * self.weights['hough'])
            valid_weights.append(self.weights['hough'])
        
        # Add Kalman filter coordinates if valid
        if kalman_coords is not None:
            valid_x.append(kalman_coords[0] * self.weights['kalman'])
            valid_y.append(kalman_coords[1] * self.weights['kalman'])
            valid_weights.append(self.weights['kalman'])
        
        # If no valid coordinates, return None
        if not valid_x:
            self.tracking_status = "Lost"
            self.confidence = 0.0
            return None
        
        # Calculate weighted average
        sum_weights = sum(valid_weights)
        if sum_weights > 0:
            x = sum(valid_x) / sum_weights
            y = sum(valid_y) / sum_weights
            
            self.tracking_status = "Tracking"
            self.confidence = 100.0 * (sum_weights / sum(self.weights.values()))
            
            return (x, y)
        
        # Should not reach here if valid_x is not empty
        self.tracking_status = "Lost"
        self.confidence = 0.0
        return None
    
    def calculate_3d_position(self, left_coords, right_coords):
        """
        Calculate 3D position from stereo coordinates using triangulation.
        
        Args:
            left_coords (tuple or None): (x, y) from left image
            right_coords (tuple or None): (x, y) from right image
            
        Returns:
            tuple or None: (x, y, z) 3D coordinates in meters or None if triangulation fails
        """
        # Check if both coordinates are valid
        if left_coords is None or right_coords is None:
            logging.error("Cannot calculate 3D position: One or both 2D coordinates are None")
            return None
        
        # Verify triangulation service
        if self.triangulation_service is None:
            logging.critical("Cannot calculate 3D position: No triangulation service provided")
            # Always return a fallback position instead of None
            return self._calculate_fallback_position(left_coords, right_coords)
            
        # Check if triangulation method exists
        if not hasattr(self.triangulation_service, 'triangulate'):
            logging.critical("Triangulation service is missing the 'triangulate' method - cannot perform triangulation")
            # Always return a fallback position instead of None
            return self._calculate_fallback_position(left_coords, right_coords)
            
        # Check if triangulation method is callable
        if not callable(getattr(self.triangulation_service, 'triangulate')):
            logging.critical("Triangulation service's 'triangulate' method is not callable - implementation error")
            # Always return a fallback position instead of None
            return self._calculate_fallback_position(left_coords, right_coords)
        
        try:
            # Clean and validate coordinates to ensure they're properly formatted
            try:
                x_left, y_left = float(left_coords[0]), float(left_coords[1])
                x_right, y_right = float(right_coords[0]), float(right_coords[1])
            except (TypeError, ValueError, IndexError) as e:
                logging.critical(f"Invalid coordinate format: {e}")
                logging.critical(f"left_coords={left_coords}, right_coords={right_coords}")
                return self._calculate_fallback_position(left_coords, right_coords)
            
            # Log input coordinates for debugging
            logging.critical(f"Triangulation input coordinates: left=({x_left}, {y_left}), right=({x_right}, {y_right})")
            
            # Calculate disparity - useful for debugging
            disparity_x = abs(x_left - x_right)
            disparity_y = abs(y_left - y_right)
            logging.critical(f"Coordinate disparities: x={disparity_x}, y={disparity_y}")
            
            # Check for large y-disparity which could indicate poor calibration or noise
            if disparity_y > 30:  # Pixel threshold
                logging.warning(f"Large y-disparity ({disparity_y} px) in stereo coordinates - triangulation may be unreliable")
                self.confidence *= max(0.5, 1.0 - (disparity_y - 30) / 100.0)  # Reduce confidence
            
            # Perform triangulation
            logging.critical(f"Calling triangulation_service.triangulate({x_left}, {y_left}, {x_right}, {y_right})")
            
            # Try triangulation
            world_coords = self.triangulation_service.triangulate(x_left, y_left, x_right, y_right)
            
            # Log the raw result
            logging.critical(f"Triangulation raw result: {world_coords}")
            
            # Check for invalid results
            if world_coords is None:
                logging.critical("Triangulation returned None - fallback to default values")
                return self._calculate_fallback_position(left_coords, right_coords)
                
            if isinstance(world_coords, np.ndarray) and np.isnan(world_coords).any():
                logging.critical("Triangulation returned NaN values - fallback to default values")
                return self._calculate_fallback_position(left_coords, right_coords)
            
            # Validate the result is a 3D point
            if isinstance(world_coords, np.ndarray):
                # Check array dimensions
                if len(world_coords) < 3:
                    logging.critical(f"Triangulation result has wrong dimensions: {world_coords}")
                    return self._calculate_fallback_position(left_coords, right_coords)
                    
                # Convert numpy array to tuple
                position = (float(world_coords[0]), float(world_coords[1]), float(world_coords[2]))
            else:
                # Check if it's a sequence with at least 3 elements
                if not hasattr(world_coords, '__len__') or len(world_coords) < 3:
                    logging.critical(f"Triangulation result has wrong format: {world_coords}")
                    return self._calculate_fallback_position(left_coords, right_coords)
                    
                # Assume triangulation returned a tuple or list
                position = tuple(map(float, world_coords[:3]))
            
            # Store the result as the last valid 3D position
            self.last_3d_position = position
            
            # Return the 3D position
            logging.critical(f"Final calculated 3D position: {position}")
            return position
        
        except Exception as e:
            logging.critical(f"Error during 3D position calculation: {str(e)}")
            import traceback
            logging.critical(traceback.format_exc())
            return self._calculate_fallback_position(left_coords, right_coords)
            
    def _calculate_fallback_position(self, left_coords, right_coords):
        """
        Calculate a fallback 3D position based on disparity when triangulation fails.
        This is a simplified approximation of depth from stereo vision.
        
        Args:
            left_coords (tuple): (x, y) coordinates in left image
            right_coords (tuple): (x, y) coordinates in right image
            
        Returns:
            tuple: (x, y, z) approximate 3D position
        """
        try:
            # Extract coordinates
            try:
                x_left, y_left = float(left_coords[0]), float(left_coords[1])
                x_right, y_right = float(right_coords[0]), float(right_coords[1])
            except (TypeError, ValueError, IndexError):
                # If parsing fails, return zeros
                return (0.0, 0.0, 0.0)
                
            # Calculate average x, y position
            x_avg = (x_left + x_right) / 2.0
            y_avg = (y_left + y_right) / 2.0
            
            # Calculate disparity (x-difference between images)
            disparity = abs(x_left - x_right)
            
            # Simple depth calculation: Z = focal_length * baseline / disparity
            # Since we don't know focal length and baseline, use a scaling factor
            # Larger disparity = closer object
            depth_scale = 100.0  # Arbitrary scale factor
            
            if disparity < 0.1:  # Avoid division by very small numbers
                z_approx = depth_scale  # Default depth if disparity is too small
            else:
                z_approx = depth_scale / disparity
                
            # Limit z to reasonable range (adjust as needed)
            z_approx = min(max(z_approx, 1.0), 1000.0)
            
            # Create fallback position
            position = (x_avg, y_avg, z_approx)
            
            # Store as last valid position
            self.last_3d_position = position
            
            logging.critical(f"Using fallback 3D position based on disparity: {position}")
            return position
            
        except Exception as e:
            logging.critical(f"Error calculating fallback position: {e}")
            # Last resort: return zeros
            return (0.0, 0.0, 0.0)
    
    def process_frame(self, frame_idx, hsv_left, hsv_right, hough_left, hough_right, 
                      kalman_left, kalman_right):
        """
        Process a frame to combine coordinates and calculate 3D position.
        
        Args:
            frame_idx (int): Current frame index
            hsv_left (tuple or None): HSV detection coordinates in left image
            hsv_right (tuple or None): HSV detection coordinates in right image
            hough_left (tuple or None): Hough detection coordinates in left image
            hough_right (tuple or None): Hough detection coordinates in right image
            kalman_left (tuple or None): Kalman prediction coordinates in left image
            kalman_right (tuple or None): Kalman prediction coordinates in right image
            
        Returns:
            dict: Processing results containing:
                - frame_idx (int): Frame index
                - left_2d (tuple): Combined left 2D coordinates
                - right_2d (tuple): Combined right 2D coordinates
                - world_3d (tuple): Calculated 3D coordinates
                - process_time (float): Processing time in milliseconds
                - fps (float): Current FPS
                - status (str): Tracking status
                - confidence (float): Detection confidence (0-100)
        """
        # Start processing timer
        self.start_processing()
        
        # Combine left coordinates
        left_combined = self.combine_coordinates(hsv_left, hough_left, kalman_left)
        
        # Combine right coordinates
        right_combined = self.combine_coordinates(hsv_right, hough_right, kalman_right)
        
        # Calculate 3D position
        position_3d = self.calculate_3d_position(left_combined, right_combined)
        
        # If 3D calculation failed but we have previous position, use it with reduced confidence
        if position_3d is None and self.last_3d_position is not None:
            position_3d = self.last_3d_position
            self.confidence *= 0.5  # Reduce confidence for using old position
        
        # End processing and get time
        process_time = self.end_processing()
        
        # Prepare and return results
        results = {
            'frame_idx': frame_idx,
            'left_2d': left_combined,
            'right_2d': right_combined,
            'world_3d': position_3d if position_3d is not None else (0.0, 0.0, 0.0),
            'process_time': process_time,
            'fps': self.get_fps(),
            'status': self.tracking_status,
            'confidence': self.confidence
        }
        
        return results 