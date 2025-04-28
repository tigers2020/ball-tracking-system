#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Coordinate Combiner service module.
This module contains the CoordinateCombiner class for combining coordinates from different sources.
"""

import logging
import numpy as np
import cv2
import math
from typing import Dict, List, Tuple, Optional, Any, Union
import time


class CoordinateCombiner:
    """
    Service class for combining and processing coordinates from different detection methods.
    Combines HSV, Hough Circle, and Kalman filter coordinates to produce final 2D and 3D positions.
    """
    
    def __init__(self, triangulation_service=None, camera_settings=None):
        """
        Initialize the coordinate combiner.
        
        Args:
            triangulation_service: Service for performing 3D triangulation
            camera_settings: Camera calibration settings
        """
        self.triangulation_service = triangulation_service
        self.camera_settings = camera_settings
        
        # Initialize camera matrices
        self.camera_matrix_left = None
        self.camera_matrix_right = None
        self.dist_coeffs_left = np.zeros((4, 1))  # Assuming no distortion for simplicity
        self.dist_coeffs_right = np.zeros((4, 1))  # Assuming no distortion for simplicity
        self.projection_matrix_left = None
        self.projection_matrix_right = None
        
        # Initialize if camera settings are provided
        if camera_settings:
            self.initialize_camera_parameters(camera_settings)
        
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
    
    def initialize_camera_parameters(self, camera_settings):
        """
        Initialize camera matrices from settings.
        
        Args:
            camera_settings: Dictionary with camera calibration parameters
        """
        logging.info("Initializing camera parameters for triangulation")
        
        try:
            # Extract camera parameters
            focal_length_mm = float(camera_settings.get('focal_length_mm', 50.0))
            sensor_width_mm = float(camera_settings.get('sensor_width_mm', 36.0))
            sensor_height_mm = float(camera_settings.get('sensor_height_mm', 24.0))
            baseline_m = float(camera_settings.get('baseline_m', 1.0))
            
            # Get image dimensions
            image_width_px = int(camera_settings.get('image_width_px', 640))
            image_height_px = int(camera_settings.get('image_height_px', 480))
            
            # Get principal point
            cx = float(camera_settings.get('principal_point_x', image_width_px / 2))
            cy = float(camera_settings.get('principal_point_y', image_height_px / 2))
            
            # Calculate focal length in pixels
            fx = (focal_length_mm / sensor_width_mm) * image_width_px
            fy = (focal_length_mm / sensor_height_mm) * image_height_px
            
            # Create camera matrix (intrinsic parameters)
            self.camera_matrix_left = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Right camera has the same intrinsic parameters
            self.camera_matrix_right = self.camera_matrix_left.copy()
            
            # Get camera location and rotation
            camera_location_x = float(camera_settings.get('camera_location_x', 0.0))
            camera_location_y = float(camera_settings.get('camera_location_y', 0.0))
            camera_location_z = float(camera_settings.get('camera_location_z', 0.0))
            camera_rotation_x = float(camera_settings.get('camera_rotation_x', 0.0))
            camera_rotation_y = float(camera_settings.get('camera_rotation_y', 0.0))
            camera_rotation_z = float(camera_settings.get('camera_rotation_z', 0.0))
            
            # Create rotation matrix
            R = self.create_rotation_matrix(
                math.radians(camera_rotation_x),
                math.radians(camera_rotation_y),
                math.radians(camera_rotation_z)
            )
            
            # Create translation vector for left camera (assumed at origin of stereo rig)
            T_left = np.zeros(3, dtype=np.float32)
            
            # Create translation vector for right camera (shifted by baseline along X)
            T_right = np.array([baseline_m, 0, 0], dtype=np.float32)
            
            # Create projection matrices
            self.projection_matrix_left = np.zeros((3, 4), dtype=np.float32)
            self.projection_matrix_right = np.zeros((3, 4), dtype=np.float32)
            
            # Set rotation part (first 3 columns)
            self.projection_matrix_left[:, :3] = R
            self.projection_matrix_right[:, :3] = R
            
            # Set translation part (4th column)
            self.projection_matrix_left[:, 3] = T_left
            self.projection_matrix_right[:, 3] = T_right
            
            # Multiply by camera matrix to get final projection matrices
            self.projection_matrix_left = self.camera_matrix_left @ self.projection_matrix_left
            self.projection_matrix_right = self.camera_matrix_right @ self.projection_matrix_right
            
            logging.info("Camera parameters initialized successfully")
            logging.debug(f"Camera matrix left: {self.camera_matrix_left}")
            logging.debug(f"Projection matrix left: {self.projection_matrix_left}")
            logging.debug(f"Projection matrix right: {self.projection_matrix_right}")
            
        except Exception as e:
            logging.error(f"Error initializing camera parameters: {e}")
            import traceback
            logging.error(traceback.format_exc())
    
    def create_rotation_matrix(self, pitch, roll, yaw):
        """
        Create a 3D rotation matrix from pitch, roll, and yaw angles.
        
        Args:
            pitch (float): X-axis rotation in radians
            roll (float): Y-axis rotation in radians
            yaw (float): Z-axis rotation in radians
            
        Returns:
            numpy.ndarray: 3x3 rotation matrix
        """
        # X-axis rotation (pitch)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ], dtype=np.float32)
        
        # Y-axis rotation (roll)
        Ry = np.array([
            [np.cos(roll), 0, np.sin(roll)],
            [0, 1, 0],
            [-np.sin(roll), 0, np.cos(roll)]
        ], dtype=np.float32)
        
        # Z-axis rotation (yaw)
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Combine rotations: R = Rz * Ry * Rx
        R = Rz @ Ry @ Rx
        
        return R
    
    def set_triangulation_service(self, service):
        """
        Set the triangulation service.
        
        Args:
            service: Service for performing 3D triangulation
        """
        self.triangulation_service = service
    
    def set_camera_settings(self, camera_settings):
        """
        Set camera calibration settings and reinitialize parameters.
        
        Args:
            camera_settings: Dictionary with camera calibration parameters
        """
        self.camera_settings = camera_settings
        self.initialize_camera_parameters(camera_settings)
    
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
        
        # If we have a triangulation service, use it first
        if self.triangulation_service is not None:
            try:
                if hasattr(self.triangulation_service, 'triangulate'):
                    # Extract coordinates
                    x_left, y_left = float(left_coords[0]), float(left_coords[1])
                    x_right, y_right = float(right_coords[0]), float(right_coords[1])
                    
                    # Log input coordinates for debugging
                    logging.debug(f"Triangulation input coordinates: left=({x_left}, {y_left}), right=({x_right}, {y_right})")
                    
                    # Try triangulation through the service
                    world_coords = self.triangulation_service.triangulate(x_left, y_left, x_right, y_right)
                    
                    # If successful, return the result
                    if world_coords is not None and not (isinstance(world_coords, np.ndarray) and np.isnan(world_coords).any()):
                        # Convert to tuple if it's an array
                        if isinstance(world_coords, np.ndarray):
                            position = tuple(float(x) for x in world_coords[:3])
                        else:
                            position = tuple(map(float, world_coords[:3]))
                        
                        # Update last valid position
                        self.last_3d_position = position
                        logging.debug(f"Triangulation service returned 3D position: {position}")
                        return position
                    
                    logging.debug("Triangulation service failed, falling back to OpenCV triangulation")
            except Exception as e:
                logging.error(f"Error using triangulation service: {e}")
                logging.debug("Falling back to OpenCV triangulation")
        
        # If we have camera matrices, use OpenCV triangulation
        if self.projection_matrix_left is not None and self.projection_matrix_right is not None:
            try:
                # Extract coordinates
                x_left, y_left = float(left_coords[0]), float(left_coords[1])
                x_right, y_right = float(right_coords[0]), float(right_coords[1])
                
                # Convert to numpy arrays
                left_point = np.array([[x_left, y_left]], dtype=np.float32)
                right_point = np.array([[x_right, y_right]], dtype=np.float32)
                
                # Undistort points (convert to normalized camera coordinates)
                left_point_norm = cv2.undistortPoints(left_point, self.camera_matrix_left, self.dist_coeffs_left)
                right_point_norm = cv2.undistortPoints(right_point, self.camera_matrix_right, self.dist_coeffs_right)
                
                # Triangulate points
                points_4d = cv2.triangulatePoints(
                    self.projection_matrix_left,
                    self.projection_matrix_right,
                    left_point_norm,
                    right_point_norm
                )
                
                # Convert from homogeneous coordinates to 3D
                points_3d = cv2.convertPointsFromHomogeneous(points_4d.T)
                
                # Extract the 3D point
                x3d, y3d, z3d = points_3d[0][0]
                
                # Validate the result (check for NaN or unreasonable values)
                if np.isnan(x3d) or np.isnan(y3d) or np.isnan(z3d) or z3d <= 0 or z3d > 1000:
                    logging.warning(f"Invalid triangulation result: ({x3d}, {y3d}, {z3d})")
                    return self._calculate_opencv_fallback(left_coords, right_coords)
                
                # Create position tuple
                position = (float(x3d), float(y3d), float(z3d))
                
                # Update last valid position
                self.last_3d_position = position
                logging.debug(f"OpenCV triangulation returned 3D position: {position}")
                
                return position
                
            except Exception as e:
                logging.error(f"Error during OpenCV triangulation: {e}")
                import traceback
                logging.error(traceback.format_exc())
                return self._calculate_opencv_fallback(left_coords, right_coords)
        
        # If neither triangulation service nor camera matrices are available,
        # use fallback method
        return self._calculate_opencv_fallback(left_coords, right_coords)
    
    def _calculate_opencv_fallback(self, left_coords, right_coords):
        """
        Calculate a 3D position fallback using simplified stereo equations
        when proper triangulation fails.
        
        Args:
            left_coords (tuple): (x, y) coordinates in left image
            right_coords (tuple): (x, y) coordinates in right image
            
        Returns:
            tuple: (x, y, z) approximate 3D position
        """
        try:
            # Extract coordinates
            x_left, y_left = float(left_coords[0]), float(left_coords[1])
            x_right, y_right = float(right_coords[0]), float(right_coords[1])
            
            # Get camera parameters (or use defaults if not available)
            if self.camera_settings:
                focal_length_px = (float(self.camera_settings.get('focal_length_mm', 50.0)) / 
                                 float(self.camera_settings.get('sensor_width_mm', 36.0)) * 
                                 float(self.camera_settings.get('image_width_px', 640)))
                baseline = float(self.camera_settings.get('baseline_m', 1.0))
            else:
                # Default values if no camera settings are available
                focal_length_px = 800.0  # Common approximate value for 640x480 images
                baseline = 1.0  # 1 meter default baseline
            
            # Calculate disparity
            disparity = abs(x_left - x_right)
            
            # Avoid division by zero or very small values
            if disparity < 0.1:
                z = 10.0  # Default 10 meters if disparity is too small
            else:
                # Basic stereo formula: Z = baseline * focal_length / disparity
                z = baseline * focal_length_px / disparity
                
                # Limit to reasonable range (0.1 to 100 meters)
                z = min(max(z, 0.1), 100.0)
            
            # Calculate X and Y world coordinates using average image position and Z
            # Assuming camera is centered and aligned with world coordinates
            if self.camera_matrix_left is not None:
                # If we have camera matrix, use it for more accurate X,Y calculation
                cx = self.camera_matrix_left[0, 2]
                cy = self.camera_matrix_left[1, 2]
                fx = self.camera_matrix_left[0, 0]
                fy = self.camera_matrix_left[1, 1]
                
                # Use average of left and right X coordinates
                x_avg = (x_left + x_right) / 2.0
                
                # Calculate X and Y in world coordinates
                x = (x_avg - cx) * z / fx
                y = (y_left - cy) * z / fy
            else:
                # Simple approximation if camera matrix is not available
                x_avg = (x_left + x_right) / 2.0
                y_avg = (y_left + y_right) / 2.0
                
                # Use average pixel coordinates (approximate center of image is origin)
                image_width = float(self.camera_settings.get('image_width_px', 640)) if self.camera_settings else 640
                image_height = float(self.camera_settings.get('image_height_px', 480)) if self.camera_settings else 480
                x = (x_avg - image_width/2) * z / focal_length_px
                y = (y_avg - image_height/2) * z / focal_length_px
            
            # Create position tuple
            position = (float(x), float(y), float(z))
            
            # Update last valid position
            self.last_3d_position = position
            logging.warning(f"Using fallback stereo calculation: {position}")
            
            return position
            
        except Exception as e:
            logging.error(f"Error calculating fallback position: {e}")
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