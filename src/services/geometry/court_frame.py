#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tennis Court Frame module.
This module contains the CourtFrame class, which handles coordinate transformations
between camera and court frames.
"""

import numpy as np
from typing import Tuple


class CourtFrame:
    """
    Tennis court frame class.
    
    This class handles coordinate transformations between camera and court frames.
    The court frame is defined as follows:
    - Origin at the center of the court
    - X-axis along the width of the court (right is positive)
    - Y-axis along the length of the court (away from the camera is positive)
    - Z-axis pointing upward (up is positive)
    """
    
    def __init__(self):
        """Initialize the court frame."""
        # Transformation from camera to court frame
        self.origin = np.array([0.0, 0.0, 0.0])  # Origin of court in camera frame
        self.rotation = np.eye(3)  # Rotation matrix from court to camera frame
        
        # Inverse transformation
        self.inv_rotation = np.eye(3)  # Inverse rotation matrix
        
    def set_transform(self, origin_x: float, origin_y: float, origin_z: float,
                     rotation_matrix: np.ndarray):
        """
        Set the transformation from camera to court frame.
        
        Args:
            origin_x: X coordinate of court origin in camera frame
            origin_y: Y coordinate of court origin in camera frame
            origin_z: Z coordinate of court origin in camera frame
            rotation_matrix: 3x3 rotation matrix from court to camera frame
        """
        # Set origin
        self.origin = np.array([origin_x, origin_y, origin_z])
        
        # Set rotation matrix
        self.rotation = rotation_matrix
        
        # Compute inverse rotation matrix
        self.inv_rotation = np.linalg.inv(rotation_matrix)
        
    def camera_to_court(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Transform a point from camera frame to court frame.
        
        Args:
            x: X coordinate in camera frame
            y: Y coordinate in camera frame
            z: Z coordinate in camera frame
            
        Returns:
            Tuple[float, float, float]: Point in court frame (x, y, z)
        """
        # Convert to numpy array
        point_camera = np.array([x, y, z])
        
        # Translate to court origin
        point_translated = point_camera - self.origin
        
        # Rotate to court frame
        point_court = self.inv_rotation @ point_translated
        
        return tuple(point_court)
    
    def court_to_camera(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Transform a point from court frame to camera frame.
        
        Args:
            x: X coordinate in court frame
            y: Y coordinate in court frame
            z: Z coordinate in court frame
            
        Returns:
            Tuple[float, float, float]: Point in camera frame (x, y, z)
        """
        # Convert to numpy array
        point_court = np.array([x, y, z])
        
        # Rotate to camera frame
        point_rotated = self.rotation @ point_court
        
        # Translate to camera origin
        point_camera = point_rotated + self.origin
        
        return tuple(point_camera)
    
    def camera_to_court_vector(self, vx: float, vy: float, vz: float) -> Tuple[float, float, float]:
        """
        Transform a vector from camera frame to court frame.
        Vectors only need rotation, no translation.
        
        Args:
            vx: X component in camera frame
            vy: Y component in camera frame
            vz: Z component in camera frame
            
        Returns:
            Tuple[float, float, float]: Vector in court frame (vx, vy, vz)
        """
        # Convert to numpy array
        vector_camera = np.array([vx, vy, vz])
        
        # Rotate to court frame
        vector_court = self.inv_rotation @ vector_camera
        
        return tuple(vector_court)
    
    def court_to_camera_vector(self, vx: float, vy: float, vz: float) -> Tuple[float, float, float]:
        """
        Transform a vector from court frame to camera frame.
        Vectors only need rotation, no translation.
        
        Args:
            vx: X component in court frame
            vy: Y component in court frame
            vz: Z component in court frame
            
        Returns:
            Tuple[float, float, float]: Vector in camera frame (vx, vy, vz)
        """
        # Convert to numpy array
        vector_court = np.array([vx, vy, vz])
        
        # Rotate to camera frame
        vector_camera = self.rotation @ vector_court
        
        return tuple(vector_camera)
    
    def get_origin(self) -> Tuple[float, float, float]:
        """
        Get the origin of the court in camera frame.
        
        Returns:
            Tuple[float, float, float]: Origin in camera frame (x, y, z)
        """
        return tuple(self.origin)
    
    def get_rotation(self) -> np.ndarray:
        """
        Get the rotation matrix from court to camera frame.
        
        Returns:
            np.ndarray: 3x3 rotation matrix
        """
        return self.rotation.copy()
    
    def get_inverse_rotation(self) -> np.ndarray:
        """
        Get the rotation matrix from camera to court frame.
        
        Returns:
            np.ndarray: 3x3 rotation matrix
        """
        return self.inv_rotation.copy()
    
    def get_court_axes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the court axes in camera frame.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: X, Y, Z axes in camera frame
        """
        # X axis is the first column of the rotation matrix
        x_axis = self.rotation[:, 0]
        
        # Y axis is the second column of the rotation matrix
        y_axis = self.rotation[:, 1]
        
        # Z axis is the third column of the rotation matrix
        z_axis = self.rotation[:, 2]
        
        return x_axis, y_axis, z_axis 