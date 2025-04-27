#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Linear triangulation implementation module.
"""

import numpy as np
import cv2
import logging
from typing import List, Dict, Any, Optional, Union, Tuple, Callable

from src.core.geometry.triangulation.base import AbstractTriangulator

logger = logging.getLogger(__name__)

class LinearTriangulator(AbstractTriangulator):
    """
    Linear triangulator implementation.
    
    This class implements triangulation using linear methods such as DLT
    (Direct Linear Transform) to recover 3D coordinates from multiple 2D points.
    """
    
    def __init__(self, camera_params: Optional[Dict[str, Any]] = None,
                 method: str = "dlt"):
        """
        Initialize the linear triangulator.
        
        Args:
            camera_params: Dictionary containing camera parameters
            method: Triangulation method to use
                   - "dlt": Direct Linear Transform
                   - "midpoint": Midpoint method (only for 2 views)
                   - "eigendecomp": Eigenvalue decomposition method
        """
        super().__init__(camera_params)
        
        self.method = method
        self.n_cameras = 0
        self.P = []  # Projection matrices (3x4)
        
    def set_method(self, method: str) -> None:
        """
        Set the triangulation method to use.
        
        Args:
            method: Triangulation method name
        """
        valid_methods = ["dlt", "midpoint", "eigendecomp"]
        if method not in valid_methods:
            logger.error(f"Unknown triangulation method: {method}. "
                       f"Available methods: {valid_methods}")
            return
            
        self.method = method
        logger.info(f"Linear triangulation method set to {method}")
        
    def set_camera_parameters(self, camera_params: Dict[str, Any]) -> None:
        """
        Set camera parameters for triangulation.
        
        Args:
            camera_params: Dictionary containing camera parameters including:
                          - camera_matrices: List of projection matrices (3x4)
                          - K: List of intrinsic parameter matrices
                          - R: List of rotation matrices
                          - t: List of translation vectors
        """
        if not camera_params:
            logger.error("No camera parameters provided")
            return
            
        # Store camera parameters
        self.camera_params = camera_params
        
        # Check for projection matrices
        required_params = []
        if "camera_matrices" in camera_params:
            required_params = ["camera_matrices"]
        else:
            required_params = ["K", "R", "t"]
            
        if not all(param in camera_params for param in required_params):
            missing = [p for p in required_params if p not in camera_params]
            logger.error(f"Missing required camera parameters: {missing}")
            return
            
        # Initialize parameters for projection
        if "camera_matrices" in camera_params:
            self.P = camera_params["camera_matrices"]
            if not isinstance(self.P, list) or len(self.P) < 2:
                logger.error("camera_matrices must be a list of at least 2 projection matrices")
                return
                
            # Convert to numpy arrays
            self.P = [np.array(p, dtype=np.float64) for p in self.P]
            self.n_cameras = len(self.P)
            
        else:
            # Compute projection matrices from K, R, t
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
                
            self.n_cameras = len(self.P)
            
        self.is_calibrated = True
        logger.info(f"Linear triangulator calibrated with {self.n_cameras} cameras")
        
    def _triangulate_dlt(self, points_2d: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Triangulate a point using Direct Linear Transform (DLT).
        
        Args:
            points_2d: List of 2D points from multiple camera views
            
        Returns:
            3D point as numpy array [X, Y, Z] or None if triangulation fails
        """
        if len(points_2d) < 2 or len(self.P) < 2:
            return None
            
        # Construct the DLT matrix A
        A = np.zeros((2 * len(points_2d), 4))
        
        for i, (point, P) in enumerate(zip(points_2d, self.P)):
            if point is None or np.isnan(np.array(point)).any():
                continue
                
            x, y = point[:2]
            
            # Add two rows to A for each point
            A[2*i] = x * P[2] - P[0]
            A[2*i+1] = y * P[2] - P[1]
            
        # Solve the system using SVD
        _, _, Vt = np.linalg.svd(A)
        
        # The solution is the last row of Vt
        point_h = Vt[-1]
        
        # Convert to non-homogeneous coordinates
        if abs(point_h[3]) < 1e-8:
            logger.warning("DLT produced point at infinity")
            return None
            
        point_3d = point_h[:3] / point_h[3]
        
        return point_3d
        
    def _triangulate_midpoint(self, points_2d: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Triangulate a point using the midpoint method (only for 2 views).
        
        Args:
            points_2d: List of 2D points from 2 camera views
            
        Returns:
            3D point as numpy array [X, Y, Z] or None if triangulation fails
        """
        if len(points_2d) != 2 or len(self.P) != 2:
            logger.error("Midpoint method requires exactly 2 camera views")
            return None
            
        # Extract camera matrices components
        # P = [R | t]
        R1 = self.P[0][:, :3]
        t1 = self.P[0][:, 3]
        
        R2 = self.P[1][:, :3]
        t2 = self.P[1][:, 3]
        
        # Get camera centers
        C1 = -np.linalg.inv(R1) @ t1
        C2 = -np.linalg.inv(R2) @ t2
        
        # Get normalized image points
        p1 = np.array(points_2d[0][:2], dtype=np.float64)
        p2 = np.array(points_2d[1][:2], dtype=np.float64)
        
        # Get ray directions
        v1 = np.linalg.inv(R1) @ np.append(p1, 1.0)
        v1 = v1 / np.linalg.norm(v1)
        
        v2 = np.linalg.inv(R2) @ np.append(p2, 1.0)
        v2 = v2 / np.linalg.norm(v2)
        
        # Compute the midpoint
        # The closest point to both lines is computed
        # by finding parameters s and t that minimize |C1 + s*v1 - (C2 + t*v2)|^2
        
        # Build system of equations
        A = np.array([
            [np.dot(v1, v1), -np.dot(v1, v2)],
            [-np.dot(v1, v2), np.dot(v2, v2)]
        ])
        
        b = np.array([
            np.dot(v1, C2 - C1),
            np.dot(v2, C1 - C2)
        ])
        
        try:
            # Solve for s and t
            s, t = np.linalg.solve(A, b)
            
            # Compute the 3D points on each ray
            point1 = C1 + s * v1
            point2 = C2 + t * v2
            
            # Midpoint is the average
            point_3d = (point1 + point2) / 2
            
            return point_3d
            
        except np.linalg.LinAlgError:
            logger.warning("Midpoint method failed: parallel rays")
            return None
            
    def _triangulate_eigendecomp(self, points_2d: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Triangulate a point using eigenvalue decomposition.
        
        Args:
            points_2d: List of 2D points from multiple camera views
            
        Returns:
            3D point as numpy array [X, Y, Z] or None if triangulation fails
        """
        if len(points_2d) < 2 or len(self.P) < 2:
            return None
            
        # Construct the DLT matrix A
        A = np.zeros((3 * len(points_2d), 4))
        
        for i, (point, P) in enumerate(zip(points_2d, self.P)):
            if point is None or np.isnan(np.array(point)).any():
                continue
                
            # Add the projection matrix rows to A
            A[3*i:3*i+3] = P
            
        # Compute A^T A
        AtA = A.T @ A
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(AtA)
        
        # The solution is the eigenvector corresponding to the smallest eigenvalue
        point_h = eigenvectors[:, 0]
        
        # Convert to non-homogeneous coordinates
        if abs(point_h[3]) < 1e-8:
            logger.warning("Eigendecomposition produced point at infinity")
            return None
            
        point_3d = point_h[:3] / point_h[3]
        
        return point_3d
        
    def triangulate_points(self, points_2d_list: List[np.ndarray]) -> np.ndarray:
        """
        Triangulate multiple 3D points from corresponding points in multiple views.
        
        Args:
            points_2d_list: List containing arrays of 2D points from multiple views
            
        Returns:
            Array of triangulated 3D points
        """
        if not self.is_calibrated:
            logger.error("Triangulator not calibrated. Call set_camera_parameters first.")
            return np.array([])
            
        # For triangulation, we need at least 2 sets of points
        if len(points_2d_list) < 2:
            logger.error("Need at least two sets of points for triangulation")
            return np.array([])
            
        # Make sure we have the same number of points in all sets
        n_points = points_2d_list[0].shape[0]
        for i in range(1, len(points_2d_list)):
            if points_2d_list[i].shape[0] != n_points:
                logger.error(f"Point set {i} has {points_2d_list[i].shape[0]} points, "
                           f"expected {n_points}")
                return np.array([])
                
        # Initialize output array
        points_3d = np.zeros((n_points, 3), dtype=np.float64)
        points_3d.fill(np.nan)  # Fill with NaN as default
        
        # Triangulate each point
        for i in range(n_points):
            # Extract 2D points for this 3D point across all views
            points_2d = [points[i] if i < points.shape[0] else None for points in points_2d_list]
            
            # Skip points with NaN values
            if any(point is None or (point is not None and np.isnan(point).any()) 
                   for point in points_2d):
                continue
                
            # Triangulate individual point
            point_3d = self.triangulate_point(points_2d)
            
            if point_3d is not None:
                points_3d[i] = point_3d
                
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
            
        # For triangulation, we need at least 2 camera views
        if len(points_2d) < 2:
            logger.error("Need at least two points for triangulation")
            return None
            
        # Check for NaN values
        if any(np.isnan(np.array(p)).any() if p is not None else True for p in points_2d):
            logger.warning("Cannot triangulate points with NaN coordinates")
            return None
            
        # Use the corresponding method
        if self.method == "dlt":
            return self._triangulate_dlt(points_2d)
        elif self.method == "midpoint" and len(points_2d) == 2 and len(self.P) == 2:
            return self._triangulate_midpoint(points_2d)
        elif self.method == "eigendecomp":
            return self._triangulate_eigendecomp(points_2d)
        else:
            # Default to DLT
            return self._triangulate_dlt(points_2d)
            
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
            
        # Convert to homogeneous coordinates
        point_h = np.append(point_3d, 1.0)
        
        # Calculate reprojection errors for all cameras
        squared_errors = []
        
        for i, P in enumerate(self.P):
            if i >= len(points_2d) or points_2d[i] is None:
                continue
                
            # Project 3D point to image
            projected_h = np.dot(P, point_h)
            
            # Check for division by zero
            if abs(projected_h[2]) < 1e-8:
                continue
                
            # Convert to image coordinates
            projected = projected_h[:2] / projected_h[2]
            
            # Calculate error
            point_2d = np.array(points_2d[i][:2])
            error = point_2d - projected
            squared_error = np.sum(error ** 2)
            
            squared_errors.append(squared_error)
            
        if not squared_errors:
            return float('inf')
            
        # Return mean squared error
        return np.mean(squared_errors) 