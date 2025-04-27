#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Direct Linear Transform (DLT) triangulation implementation.
"""

import numpy as np
import cv2
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

from src.core.geometry.triangulation.base import AbstractTriangulator

logger = logging.getLogger(__name__)

class DLTTriangulator(AbstractTriangulator):
    """
    Direct Linear Transform (DLT) triangulator implementation.
    
    This class implements triangulation using the DLT algorithm,
    which is a simple and efficient method for triangulating 3D points
    from corresponding 2D points in multiple views.
    """
    
    def __init__(self, camera_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the DLT triangulator.
        
        Args:
            camera_params: Dictionary containing camera parameters
        """
        super().__init__(camera_params)
        
    def set_camera_parameters(self, camera_params: Dict[str, Any]) -> None:
        """
        Set camera parameters for triangulation.
        
        Args:
            camera_params: Dictionary containing camera parameters including:
                          - camera_matrix_left: Intrinsic parameters for left camera
                          - camera_matrix_right: Intrinsic parameters for right camera
                          - R_left: Rotation matrix for left camera
                          - R_right: Rotation matrix for right camera
                          - t_left: Translation vector for left camera
                          - t_right: Translation vector for right camera
        """
        if not camera_params:
            logger.error("No camera parameters provided")
            return
            
        # Check if required parameters are present
        required_params = [
            "camera_matrix_left", "camera_matrix_right",
            "R_left", "R_right", "t_left", "t_right"
        ]
        
        if not all(param in camera_params for param in required_params):
            missing = [p for p in required_params if p not in camera_params]
            logger.error(f"Missing required camera parameters: {missing}")
            return
            
        # Store camera parameters
        self.camera_params = camera_params
        
        # Set camera matrices
        self.camera_matrix = np.array(camera_params["camera_matrix_left"], dtype=np.float64)
        
        # Calculate projection matrices
        K_left = np.array(camera_params["camera_matrix_left"], dtype=np.float64)
        R_left = np.array(camera_params["R_left"], dtype=np.float64)
        t_left = np.array(camera_params["t_left"], dtype=np.float64).reshape(3, 1)
        
        K_right = np.array(camera_params["camera_matrix_right"], dtype=np.float64)
        R_right = np.array(camera_params["R_right"], dtype=np.float64)
        t_right = np.array(camera_params["t_right"], dtype=np.float64).reshape(3, 1)
        
        # Calculate projection matrices P = K[R|t]
        self.P_left = np.dot(K_left, np.hstack((R_left, t_left)))
        self.P_right = np.dot(K_right, np.hstack((R_right, t_right)))
        
        # Set calibrated flag
        self.is_calibrated = True
        logger.info("DLT triangulator calibrated")
        
    def triangulate_points(self, points_2d_list: List[np.ndarray], **kwargs) -> np.ndarray:
        """
        Triangulate multiple 3D points from corresponding points in two views.
        
        Args:
            points_2d_list: List containing two arrays of 2D points, [points_left, points_right]
            **kwargs: Additional parameters (not used in this implementation)
            
        Returns:
            Array of triangulated 3D points
        """
        if not self.is_calibrated:
            logger.error("Triangulator not calibrated. Call set_camera_parameters first.")
            return np.array([])
            
        # Validate inputs
        if len(points_2d_list) != 2:
            logger.error(f"DLT triangulation requires exactly 2 point sets, got {len(points_2d_list)}")
            return np.array([])
            
        points_left = points_2d_list[0]
        points_right = points_2d_list[1]
        
        if points_left.shape[0] != points_right.shape[0]:
            logger.error("Point sets must have the same number of points")
            return np.array([])
            
        if points_left.shape[1] < 2 or points_right.shape[1] < 2:
            logger.error("Point sets must have at least 2 coordinates (x, y)")
            return np.array([])
            
        # Triangulate the points
        points_3d = np.zeros((points_left.shape[0], 3), dtype=np.float64)
        
        for i in range(points_left.shape[0]):
            point_left = points_left[i, :2]
            point_right = points_right[i, :2]
            
            # Skip if any point contains NaN
            if np.isnan(point_left).any() or np.isnan(point_right).any():
                points_3d[i] = np.array([np.nan, np.nan, np.nan])
                continue
                
            # Triangulate this point
            point_3d = self.triangulate_point([(point_left[0], point_left[1]), 
                                              (point_right[0], point_right[1])])
            
            if point_3d is not None:
                points_3d[i] = point_3d
            else:
                points_3d[i] = np.array([np.nan, np.nan, np.nan])
                
        return points_3d
        
    def triangulate_point(self, points_2d: Union[List[Tuple[float, float]], 
                                               List[np.ndarray]]) -> Optional[np.ndarray]:
        """
        Triangulate a single 3D point from two corresponding 2D points.
        
        Args:
            points_2d: List of two 2D points [(x1, y1), (x2, y2)] from left and right cameras
            
        Returns:
            3D point as numpy array [X, Y, Z] or None if triangulation fails
        """
        if not self.is_calibrated:
            logger.error("Triangulator not calibrated. Call set_camera_parameters first.")
            return None
            
        # Validate inputs
        if len(points_2d) != 2:
            logger.error(f"DLT triangulation requires exactly 2 points, got {len(points_2d)}")
            return None
            
        # Extract points
        try:
            point_left = np.array(points_2d[0], dtype=np.float64)
            point_right = np.array(points_2d[1], dtype=np.float64)
            
            if point_left.size < 2 or point_right.size < 2:
                logger.error("Each point must have at least (x, y) coordinates")
                return None
                
            point_left = point_left[:2]  # Ensure we only have (x, y)
            point_right = point_right[:2]
            
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid point format: {e}")
            return None
            
        # Triangulate using DLT method
        return self._triangulate_point_dlt(point_left, point_right)
        
    def _triangulate_point_dlt(self, point_left: np.ndarray, point_right: np.ndarray) -> np.ndarray:
        """
        Triangulate a 3D point using Direct Linear Transform (DLT) method.
        
        Args:
            point_left: 2D point from left camera view (x, y)
            point_right: 2D point from right camera view (x, y)
            
        Returns:
            Triangulated 3D point (X, Y, Z)
        """
        # Construct the DLT matrix (4 equations from 2 points)
        A = np.zeros((4, 4), dtype=np.float64)
        
        # Left camera equations
        A[0, :] = point_left[0] * self.P_left[2, :] - self.P_left[0, :]
        A[1, :] = point_left[1] * self.P_left[2, :] - self.P_left[1, :]
        
        # Right camera equations
        A[2, :] = point_right[0] * self.P_right[2, :] - self.P_right[0, :]
        A[3, :] = point_right[1] * self.P_right[2, :] - self.P_right[1, :]
        
        # Solve the system of equations using SVD
        _, _, Vt = np.linalg.svd(A)
        
        # The solution is the last row of Vt (corresponding to smallest singular value)
        X = Vt[-1, :]
        
        # Convert from homogeneous to Euclidean coordinates
        X = X / X[3]
        
        return X[:3]
        
    def triangulate_opencv(self, points_left: np.ndarray, points_right: np.ndarray) -> np.ndarray:
        """
        Triangulate points using OpenCV's triangulatePoints function.
        This is an alternative to the custom DLT implementation.
        
        Args:
            points_left: Array of 2D points from left camera (Nx2)
            points_right: Array of 2D points from right camera (Nx2)
            
        Returns:
            Array of triangulated 3D points (Nx3)
        """
        if not self.is_calibrated:
            logger.error("Triangulator not calibrated")
            return np.array([])
            
        # Ensure points are in the correct format for OpenCV
        if len(points_left) != len(points_right):
            logger.error("Point sets must have the same number of points")
            return np.array([])
            
        # Reshape points for OpenCV: from Nx2 to 2xN
        pts_left = points_left.T[:2, :]  # Ensure only x,y coordinates
        pts_right = points_right.T[:2, :]
        
        # Triangulate points using OpenCV
        points_4d = cv2.triangulatePoints(self.P_left, self.P_right, pts_left, pts_right)
        
        # Convert from homogeneous to Euclidean coordinates (4xN to Nx3)
        points_3d = points_4d[:3, :].T
        for i in range(points_3d.shape[0]):
            points_3d[i] = points_3d[i] / points_4d[3, i]
            
        return points_3d
        
    def calculate_reprojection_error(self, 
                                   point_3d: np.ndarray, 
                                   points_2d: List[Tuple[float, float]]) -> float:
        """
        Calculate the reprojection error for a triangulated point.
        
        Args:
            point_3d: Triangulated 3D point
            points_2d: List of 2D points used for triangulation [left_point, right_point]
            
        Returns:
            Mean squared reprojection error
        """
        if not self.is_calibrated or point_3d is None:
            return float('inf')
            
        if len(points_2d) != 2:
            logger.error("Reprojection error calculation requires exactly 2 points")
            return float('inf')
            
        # Convert 3D point to homogeneous coordinates
        point_3d_h = np.append(point_3d, 1.0)
        
        # Project the 3D point onto image planes
        point_left_proj_h = np.dot(self.P_left, point_3d_h)
        point_right_proj_h = np.dot(self.P_right, point_3d_h)
        
        # Convert to Euclidean coordinates
        point_left_proj = point_left_proj_h[:2] / point_left_proj_h[2]
        point_right_proj = point_right_proj_h[:2] / point_right_proj_h[2]
        
        # Calculate squared errors
        error_left = np.sum((np.array(points_2d[0]) - point_left_proj) ** 2)
        error_right = np.sum((np.array(points_2d[1]) - point_right_proj) ** 2)
        
        # Return mean squared error
        return (error_left + error_right) / 2.0 