#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-view triangulation module.
Implements triangulation using multiple camera views.
"""

import numpy as np
import cv2
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from scipy.optimize import least_squares

from src.core.geometry.triangulation.base import AbstractTriangulator

logger = logging.getLogger(__name__)

class MultiViewTriangulator(AbstractTriangulator):
    """
    Multi-view triangulator class.
    Implements triangulation using multiple camera views for more robust 3D reconstruction.
    """
    
    def __init__(self, camera_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the multi-view triangulator.
        
        Args:
            camera_params: Dictionary containing camera parameters for multiple cameras
        """
        super().__init__(camera_params)
        self.camera_indices = []
        self.projection_matrices = {}
        self.camera_matrices = {}
        self.distortion_coeffs = {}
        self.rotation_matrices = {}
        self.translation_vectors = {}
        self.min_cameras_required = 2
        
    def set_camera_parameters(self, camera_params: Dict[str, Any]) -> None:
        """
        Set camera parameters for triangulation.
        
        The camera_params dictionary should have camera indices as keys and
        individual camera parameters as values. Each camera entry should contain:
        - camera_matrix (intrinsic parameters)
        - distortion_coefficients
        - rotation_matrix (extrinsic parameters)
        - translation_vector (extrinsic parameters)
        
        Args:
            camera_params: Dictionary containing parameters for multiple cameras
        """
        if not camera_params:
            logger.error("No camera parameters provided")
            return
            
        # Extract camera indices and check minimum requirement
        self.camera_indices = sorted(list(camera_params.keys()))
        if len(self.camera_indices) < self.min_cameras_required:
            logger.error(f"At least {self.min_cameras_required} cameras are required for triangulation")
            return
            
        self.projection_matrices = {}
        self.camera_matrices = {}
        self.distortion_coeffs = {}
        self.rotation_matrices = {}
        self.translation_vectors = {}
        
        # Process each camera's parameters
        for cam_idx in self.camera_indices:
            cam_data = camera_params[cam_idx]
            
            # Check for required parameters
            required_params = ['camera_matrix', 'distortion_coefficients', 
                              'rotation_matrix', 'translation_vector']
            if not all(param in cam_data for param in required_params):
                logger.error(f"Missing required parameters for camera {cam_idx}")
                continue
                
            # Store camera parameters
            self.camera_matrices[cam_idx] = np.array(cam_data['camera_matrix'], dtype=np.float64)
            self.distortion_coeffs[cam_idx] = np.array(cam_data['distortion_coefficients'], dtype=np.float64)
            self.rotation_matrices[cam_idx] = np.array(cam_data['rotation_matrix'], dtype=np.float64)
            self.translation_vectors[cam_idx] = np.array(cam_data['translation_vector'], dtype=np.float64).reshape(3, 1)
            
            # Calculate projection matrix for this camera
            R = self.rotation_matrices[cam_idx]
            t = self.translation_vectors[cam_idx]
            K = self.camera_matrices[cam_idx]
            
            # P = K[R|t]
            Rt = np.hstack((R, t))
            P = np.dot(K, Rt)
            self.projection_matrices[cam_idx] = P
            
            logger.debug(f"Set projection matrix for camera {cam_idx}")
            
        # Set calibration status
        if len(self.projection_matrices) >= self.min_cameras_required:
            self.is_calibrated = True
            logger.info(f"Triangulator calibrated with {len(self.projection_matrices)} cameras")
        else:
            logger.warning("Failed to calibrate the triangulator with sufficient cameras")
            
    def triangulate_points(self, points_2d_list: List[np.ndarray], 
                           camera_indices: Optional[List[int]] = None, 
                           **kwargs) -> np.ndarray:
        """
        Triangulate multiple 3D points from multiple 2D points across different camera views.
        
        Args:
            points_2d_list: List of 2D point arrays, one array per camera
            camera_indices: List of camera indices corresponding to points_2d_list
                           If None, uses cameras in the order they were configured
            **kwargs: Additional parameters:
                      - bundle_adjust: Boolean indicating whether to refine points using bundle adjustment
                      
        Returns:
            Array of triangulated 3D points
        """
        if not self.is_calibrated:
            logger.error("Triangulator not calibrated. Call set_camera_parameters first.")
            return np.array([])
            
        # Validate inputs
        if camera_indices is None:
            # Use first N cameras from configured cameras
            camera_indices = self.camera_indices[:len(points_2d_list)]
        else:
            # Verify all camera indices are valid
            for cam_idx in camera_indices:
                if cam_idx not in self.projection_matrices:
                    logger.error(f"Camera index {cam_idx} not found in configured cameras")
                    return np.array([])
                    
        if len(points_2d_list) != len(camera_indices):
            logger.error("Number of point sets does not match number of camera indices")
            return np.array([])
            
        # Check that all point sets have the same number of points
        num_points = points_2d_list[0].shape[0]
        for i, points in enumerate(points_2d_list):
            if points.shape[0] != num_points:
                logger.error(f"Point set for camera {camera_indices[i]} has different number of points")
                return np.array([])
                
        # Triangulate each point
        points_3d = np.zeros((num_points, 3), dtype=np.float64)
        
        for i in range(num_points):
            # Extract corresponding 2D points from each camera
            points_2d = []
            for j, points in enumerate(points_2d_list):
                cam_idx = camera_indices[j]
                point = (float(points[i, 0]), float(points[i, 1]))
                points_2d.append((cam_idx, point))
                
            # Triangulate this point
            point_3d = self.triangulate_point(points_2d)
            if point_3d is not None:
                points_3d[i] = point_3d
                
        # Perform bundle adjustment if requested
        if kwargs.get('bundle_adjust', False):
            points_3d = self.bundle_adjustment(points_3d, points_2d_list, camera_indices)
                
        return points_3d
        
    def _triangulate_point_dlt(self, points_2d: List[Tuple[int, Tuple[float, float]]]) -> np.ndarray:
        """
        Triangulate a single 3D point using the Direct Linear Transform (DLT) method.
        
        Args:
            points_2d: List of (camera_idx, point) pairs
            
        Returns:
            Triangulated 3D point
        """
        # Build the A matrix for the DLT method
        A = np.zeros((len(points_2d) * 2, 4), dtype=np.float64)
        
        for i, (cam_idx, point) in enumerate(points_2d):
            x, y = point
            P = self.projection_matrices[cam_idx]
            
            # Fill A matrix based on the DLT method equations
            A[i*2, :] = x * P[2, :] - P[0, :]
            A[i*2+1, :] = y * P[2, :] - P[1, :]
            
        # Solve the system of equations
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1, :]
        
        # Convert to homogeneous coordinates
        X = X / X[3]
        
        return X[:3]
        
    def triangulate_point(self, points_2d: Union[List[Tuple[float, float]], 
                                               List[Tuple[int, Tuple[float, float]]]]) -> Optional[np.ndarray]:
        """
        Triangulate a single 3D point from multiple 2D points.
        
        Args:
            points_2d: Either:
                      - List of 2D points [(x1, y1), (x2, y2), ...] (uses cameras in order)
                      - List of (camera_idx, point) pairs [(cam1, (x1, y1)), (cam2, (x2, y2)), ...]
            
        Returns:
            3D point as numpy array [X, Y, Z] or None if triangulation fails
        """
        if not self.is_calibrated:
            logger.error("Triangulator not calibrated. Call set_camera_parameters first.")
            return None
            
        # Check input format and convert if necessary
        if len(points_2d) < self.min_cameras_required:
            logger.error(f"At least {self.min_cameras_required} points are required for triangulation")
            return None
            
        # Determine input format and standardize
        if isinstance(points_2d[0], tuple) and len(points_2d[0]) == 2:
            # Format is [(x1, y1), (x2, y2), ...]
            if len(points_2d) > len(self.camera_indices):
                logger.warning("More points provided than configured cameras. Using first points.")
                points_2d = points_2d[:len(self.camera_indices)]
                
            # Convert to [(cam1, (x1, y1)), (cam2, (x2, y2)), ...]
            standardized_points = []
            for i, point in enumerate(points_2d):
                cam_idx = self.camera_indices[i]
                standardized_points.append((cam_idx, point))
                
            points_2d = standardized_points
            
        # Now the format is [(cam1, (x1, y1)), (cam2, (x2, y2)), ...]
        # Verify all camera indices are valid
        for cam_idx, _ in points_2d:
            if cam_idx not in self.projection_matrices:
                logger.error(f"Camera index {cam_idx} not found in configured cameras")
                return None
                
        try:
            # Use DLT method for triangulation
            point_3d = self._triangulate_point_dlt(points_2d)
            return point_3d
        except Exception as e:
            logger.error(f"Error during triangulation: {str(e)}")
            return None
            
    def calculate_reprojection_error(self, 
                                    point_3d: np.ndarray, 
                                    points_2d: List[Tuple[int, Tuple[float, float]]]) -> float:
        """
        Calculate the reprojection error for a triangulated point.
        
        Args:
            point_3d: Triangulated 3D point
            points_2d: List of (camera_idx, point) pairs used for triangulation
            
        Returns:
            Mean squared reprojection error across all views
        """
        if not self.is_calibrated:
            logger.error("Triangulator not calibrated")
            return float('inf')
            
        if point_3d is None:
            return float('inf')
            
        total_error = 0.0
        
        # Create homogeneous 3D point
        point_3d_h = np.append(point_3d, 1.0)
        
        for cam_idx, point_2d in points_2d:
            if cam_idx not in self.projection_matrices:
                logger.warning(f"Camera {cam_idx} not found in configured cameras")
                continue
                
            # Project the 3D point onto this camera's image plane
            P = self.projection_matrices[cam_idx]
            point_proj_h = np.dot(P, point_3d_h)
            
            # Convert from homogeneous coordinates
            point_proj = point_proj_h[:2] / point_proj_h[2]
            
            # Calculate squared Euclidean distance
            error = np.sum((np.array(point_2d) - point_proj) ** 2)
            total_error += error
            
        # Return mean squared error
        return total_error / len(points_2d)
        
    def bundle_adjustment(self, 
                         points_3d: np.ndarray, 
                         points_2d_list: List[np.ndarray], 
                         camera_indices: List[int]) -> np.ndarray:
        """
        Refine triangulated points using bundle adjustment.
        This is a simplified version that only optimizes 3D point positions while
        keeping camera parameters fixed.
        
        Args:
            points_3d: Initial triangulated 3D points
            points_2d_list: List of 2D point arrays, one array per camera
            camera_indices: List of camera indices corresponding to points_2d_list
            
        Returns:
            Refined 3D points
        """
        def objective_function(params, points_2d_list, camera_indices, projection_matrices):
            """
            Objective function for bundle adjustment.
            
            Args:
                params: Flattened array of 3D points (3*n)
                points_2d_list: List of 2D point arrays, one array per camera
                camera_indices: List of camera indices
                projection_matrices: Dictionary of projection matrices
                
            Returns:
                Flattened array of reprojection errors
            """
            num_points = len(params) // 3
            points_3d = params.reshape(num_points, 3)
            
            # Calculate reprojection errors
            errors = []
            
            for i in range(num_points):
                point_3d = points_3d[i]
                point_3d_h = np.append(point_3d, 1.0)
                
                for j, cam_idx in enumerate(camera_indices):
                    if points_2d_list[j][i, 0] < 0 or points_2d_list[j][i, 1] < 0:
                        # Skip invalid points (negative coordinates often indicate masked/invalid points)
                        continue
                        
                    P = projection_matrices[cam_idx]
                    point_proj_h = np.dot(P, point_3d_h)
                    point_proj = point_proj_h[:2] / point_proj_h[2]
                    
                    error = points_2d_list[j][i, :2] - point_proj
                    errors.append(error)
            
            return np.array(errors).flatten()
            
        # Skip if no points to refine
        if len(points_3d) == 0:
            return points_3d
            
        try:
            # Optimize using least squares
            params_initial = points_3d.flatten()
            
            result = least_squares(
                objective_function,
                params_initial,
                method='lm',  # Levenberg-Marquardt algorithm
                args=(points_2d_list, camera_indices, self.projection_matrices),
                max_nfev=100  # Limit number of function evaluations
            )
            
            # Extract optimized points
            points_3d_refined = result.x.reshape(-1, 3)
            
            logger.info(f"Bundle adjustment completed: initial error={result.cost_initial}, final error={result.cost}")
            
            return points_3d_refined
            
        except Exception as e:
            logger.error(f"Error during bundle adjustment: {str(e)}")
            return points_3d 