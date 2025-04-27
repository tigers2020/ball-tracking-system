#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OpenCV-based triangulation implementation module.
"""

import numpy as np
import cv2
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

from src.core.geometry.triangulation.base import AbstractTriangulator

logger = logging.getLogger(__name__)

class CVTriangulator(AbstractTriangulator):
    """
    OpenCV-based triangulator implementation.
    
    This class implements triangulation using OpenCV's triangulation functions,
    which are optimized for stereo camera setups.
    """
    
    def __init__(self, camera_params: Optional[Dict[str, Any]] = None,
                 method: int = cv2.TRIANGULATE_POINTS):
        """
        Initialize the OpenCV triangulator.
        
        Args:
            camera_params: Dictionary containing camera parameters
            method: OpenCV triangulation method to use
                    - cv2.TRIANGULATE_POINTS: Linear triangulation
                    - cv2.TRIANGULATE_EIG: Uses eigenvalue decomposition
        """
        super().__init__(camera_params)
        self.method = method
        self.valid_methods = {
            "linear": cv2.TRIANGULATE_POINTS,
            "eigen": getattr(cv2, "TRIANGULATE_EIG", cv2.TRIANGULATE_POINTS)  # Fallback for older OpenCV versions
        }
        
    def set_method(self, method: Union[str, int]) -> None:
        """
        Set the triangulation method to use.
        
        Args:
            method: OpenCV triangulation method (string or constant)
        """
        if isinstance(method, str):
            if method not in self.valid_methods:
                logger.error(f"Unknown triangulation method: {method}. "
                           f"Available methods: {list(self.valid_methods.keys())}")
                return
                
            self.method = self.valid_methods[method]
        else:
            self.method = method
            
        logger.info(f"OpenCV triangulation method set to {self.method}")
        
    def set_camera_parameters(self, camera_params: Dict[str, Any]) -> None:
        """
        Set camera parameters for triangulation.
        
        Args:
            camera_params: Dictionary containing camera parameters including:
                          - camera_matrices: List of camera matrices (3x4 projection matrices)
                          - R: List of rotation matrices
                          - t: List of translation vectors
                          - K: List of intrinsic parameter matrices
        """
        if not camera_params:
            logger.error("No camera parameters provided")
            return
            
        # Check the required parameters for OpenCV triangulation
        # OpenCV triangulation works with projection matrices
        if "camera_matrices" in camera_params:
            # We have projection matrices directly
            P = camera_params["camera_matrices"]
            if not isinstance(P, list) or len(P) < 2:
                logger.error("camera_matrices must be a list of at least 2 projection matrices")
                return
                
            self.P = [np.array(p, dtype=np.float64) for p in P]
        else:
            # We need to compute projection matrices from K, R, t
            required_params = ["K", "R", "t"]
            
            if not all(param in camera_params for param in required_params):
                missing = [p for p in required_params if p not in camera_params]
                logger.error(f"Missing required camera parameters: {missing}")
                return
                
            K = camera_params["K"]
            R = camera_params["R"]
            t = camera_params["t"]
            
            if not all(isinstance(x, list) for x in [K, R, t]):
                logger.error("K, R, and t must be lists of matrices/vectors")
                return
                
            if not (len(K) == len(R) == len(t)):
                logger.error("K, R, and t must have the same length")
                return
                
            # Compute projection matrices P = K[R|t]
            self.P = []
            for i in range(len(K)):
                K_i = np.array(K[i], dtype=np.float64)
                R_i = np.array(R[i], dtype=np.float64)
                t_i = np.array(t[i], dtype=np.float64).reshape(3, 1)
                
                P_i = np.dot(K_i, np.hstack((R_i, t_i)))
                self.P.append(P_i)
                
        # Store camera parameters
        self.camera_params = camera_params
        
        # OpenCV triangulation works best with 2 cameras
        if len(self.P) >= 2:
            self.P1 = self.P[0]
            self.P2 = self.P[1]
            self.is_calibrated = True
            logger.info("CV triangulator calibrated with primary stereo pair")
        else:
            logger.error("At least 2 cameras are required for OpenCV triangulation")
            return
            
        # For compatibility with reprojection, calculate camera centers
        if "R" in camera_params and "t" in camera_params:
            self.camera_centers = []
            R = camera_params["R"]
            t = camera_params["t"]
            
            for i in range(len(R)):
                R_i = np.array(R[i], dtype=np.float64)
                t_i = np.array(t[i], dtype=np.float64).reshape(3, 1)
                
                # Camera center C = -R^T * t
                C_i = -np.dot(R_i.T, t_i).flatten()
                self.camera_centers.append(C_i)
                
            if len(self.camera_centers) >= 2:
                self.C1 = self.camera_centers[0]
                self.C2 = self.camera_centers[1]
                
        if self.is_calibrated:
            logger.info("OpenCV triangulator calibrated successfully")
            
    def triangulate_points(self, points_2d_list: List[np.ndarray], **kwargs) -> np.ndarray:
        """
        Triangulate multiple 3D points from corresponding points in multiple views.
        
        Args:
            points_2d_list: List containing arrays of 2D points from multiple views
            **kwargs: Additional parameters
            
        Returns:
            Array of triangulated 3D points
        """
        if not self.is_calibrated:
            logger.error("Triangulator not calibrated. Call set_camera_parameters first.")
            return np.array([])
            
        # For OpenCV triangulation, we need at least 2 sets of points
        if len(points_2d_list) < 2:
            logger.error("Need at least two sets of points for triangulation")
            return np.array([])
            
        # OpenCV triangulation primarily works with 2 cameras
        # If more cameras are provided, we'll use the first two
        if len(points_2d_list) > 2:
            logger.warning(f"Using only the first 2 cameras for triangulation "
                         f"(out of {len(points_2d_list)} provided)")
            points_2d_list = points_2d_list[:2]
            
        # Make sure we have the same number of points in both sets
        n_points = points_2d_list[0].shape[0]
        if points_2d_list[1].shape[0] != n_points:
            logger.error(f"Point sets have different numbers of points: "
                       f"{n_points} vs {points_2d_list[1].shape[0]}")
            return np.array([])
            
        # Extract points for OpenCV triangulation
        points1 = points_2d_list[0][:, :2].T  # OpenCV expects points as 2xN
        points2 = points_2d_list[1][:, :2].T
        
        # Skip points with NaN values
        valid_mask = ~(np.isnan(points1).any(axis=0) | np.isnan(points2).any(axis=0))
        
        # Initialize output array
        points_3d = np.zeros((n_points, 3), dtype=np.float64)
        points_3d.fill(np.nan)  # Fill with NaN as default
        
        # Only triangulate points that don't have NaN coordinates
        if np.any(valid_mask):
            valid_points1 = points1[:, valid_mask]
            valid_points2 = points2[:, valid_mask]
            
            # Triangulate points using OpenCV's method
            # Returns homogeneous 4xN array
            points_4d_h = cv2.triangulatePoints(
                self.P1, self.P2, valid_points1, valid_points2, self.method)
            
            # Convert to Euclidean coordinates (3xN)
            points_3d_valid = (points_4d_h[:3] / points_4d_h[3:4]).T
            
            # Update valid points in the output array
            valid_indices = np.where(valid_mask)[0]
            points_3d[valid_indices] = points_3d_valid
            
        return points_3d
        
    def triangulate_point(self, points_2d: List[Union[Tuple[float, float], np.ndarray]]) -> Optional[np.ndarray]:
        """
        Triangulate a single 3D point from corresponding 2D points.
        
        Args:
            points_2d: List of 2D points from multiple camera views
            
        Returns:
            3D point as numpy array [X, Y, Z] or None if triangulation fails
        """
        if not self.is_calibrated:
            logger.error("Triangulator not calibrated. Call set_camera_parameters first.")
            return None
            
        # For OpenCV triangulation, we need at least 2 cameras
        if len(points_2d) < 2:
            logger.error("Need at least two points for triangulation")
            return None
            
        # OpenCV triangulation primarily works with 2 cameras
        # If more cameras are provided, we'll use the first two
        if len(points_2d) > 2:
            logger.warning(f"Using only the first 2 cameras for triangulation "
                         f"(out of {len(points_2d)} provided)")
            points_2d = points_2d[:2]
            
        # Extract points
        try:
            point1 = np.array(points_2d[0][:2], dtype=np.float64).reshape(2, 1)
            point2 = np.array(points_2d[1][:2], dtype=np.float64).reshape(2, 1)
            
            # Check for NaN values
            if np.isnan(point1).any() or np.isnan(point2).any():
                logger.warning("Cannot triangulate points with NaN coordinates")
                return None
                
        except (ValueError, TypeError, IndexError) as e:
            logger.error(f"Invalid point format: {e}")
            return None
            
        # Triangulate using OpenCV's triangulatePoints function
        point_4d_h = cv2.triangulatePoints(self.P1, self.P2, point1, point2, self.method)
        
        # Convert to Euclidean coordinates
        if abs(point_4d_h[3, 0]) < 1e-8:
            logger.warning("Homogeneous coordinate is close to zero")
            return None
            
        point_3d = (point_4d_h[:3, 0] / point_4d_h[3, 0]).flatten()
        
        return point_3d
        
    def calculate_reprojection_error(self, 
                                   point_3d: np.ndarray, 
                                   points_2d: List[Union[Tuple[float, float], np.ndarray]]) -> float:
        """
        Calculate the reprojection error for a triangulated point.
        
        Args:
            point_3d: Triangulated 3D point
            points_2d: List of 2D points used for triangulation
            
        Returns:
            Mean squared reprojection error
        """
        if not self.is_calibrated or point_3d is None:
            return float('inf')
            
        # For consistency with triangulation, only use the first two cameras
        n_cameras = min(len(points_2d), 2)
        if n_cameras < 2:
            logger.error("Need at least two points for calculating reprojection error")
            return float('inf')
            
        # Convert 3D point to homogeneous coordinates
        point_3d_h = np.append(point_3d, 1.0)
        
        # Calculate reprojection error for the first two cameras
        total_error = 0.0
        
        for i in range(n_cameras):
            # Project the 3D point onto the image plane
            projected_h = np.dot(self.P[i], point_3d_h)
            
            # Convert to Euclidean coordinates
            if abs(projected_h[2]) < 1e-8:
                logger.warning(f"Homogeneous coordinate close to zero in camera {i}")
                return float('inf')
                
            projected = projected_h[:2] / projected_h[2]
            
            # Calculate squared error
            point_2d = np.array(points_2d[i][:2])
            error = np.sum((point_2d - projected) ** 2)
            total_error += error
            
        # Return mean squared error
        return total_error / n_cameras 