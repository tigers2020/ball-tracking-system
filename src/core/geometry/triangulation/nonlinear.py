#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Nonlinear triangulation module.

This module provides an implementation of nonlinear triangulation that refines
linear triangulation results using optimization techniques to minimize reprojection error.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple, Callable

import numpy as np
import cv2
from scipy import optimize

from src.core.geometry.triangulation.base import AbstractTriangulator
from src.core.geometry.triangulation.linear import LinearTriangulator
from src.core.geometry.triangulation.utils import calculate_reprojection_error

logger = logging.getLogger(__name__)

class NonlinearTriangulator(AbstractTriangulator):
    """
    Nonlinear triangulator class that refines 3D points using optimization techniques.
    
    This triangulator first uses a linear method to get an initial estimate, then
    refines it using nonlinear optimization to minimize reprojection error.
    """
    
    def __init__(self, camera_params: Optional[Dict[str, Any]] = None, 
                linear_method: str = 'dlt', 
                optimization_method: str = 'lm'):
        """
        Initialize the nonlinear triangulator.
        
        Args:
            camera_params: Dictionary containing camera parameters
            linear_method: Linear method to use for initial estimate ('dlt', 'midpoint', 'eigen')
            optimization_method: Optimization method to use ('lm', 'trf', 'dogbox')
        """
        super().__init__(camera_params)
        self.linear_method = linear_method
        self.optimization_method = optimization_method
        self.linear_triangulator = LinearTriangulator(camera_params, method=linear_method)
        
        # Camera parameters
        self.projection_matrices = []
        self.camera_matrices = []
        self.distortion_coeffs = []
        
        # Initialize with camera parameters if provided
        if camera_params is not None:
            self.set_camera_parameters(camera_params)
    
    def set_camera_parameters(self, camera_params: Dict[str, Any]) -> None:
        """
        Set camera parameters for triangulation.
        
        Args:
            camera_params: Dictionary containing camera parameters
        """
        self.camera_params = camera_params
        self.linear_triangulator.set_camera_parameters(camera_params)
        
        # Extract projection matrices and other parameters
        if 'projection_matrices' in camera_params:
            self.projection_matrices = camera_params['projection_matrices']
        
        if 'camera_matrices' in camera_params:
            self.camera_matrices = camera_params['camera_matrices']
        
        if 'distortion_coeffs' in camera_params:
            self.distortion_coeffs = camera_params['distortion_coeffs']
            
        self.is_calibrated = True
        logger.info("Nonlinear triangulator configured with camera parameters")
    
    def triangulate_points(self, points_2d_list: List[np.ndarray]) -> np.ndarray:
        """
        Triangulate multiple 3D points from corresponding 2D points in multiple views.
        
        Args:
            points_2d_list: List containing arrays of 2D points from multiple views
            
        Returns:
            Array of triangulated 3D points
        """
        if not self.is_calibrated:
            raise RuntimeError("Triangulator not calibrated. Set camera parameters first.")
        
        # Use linear triangulation for initial estimate
        initial_points_3d = self.linear_triangulator.triangulate_points(points_2d_list)
        
        # Refine each point using nonlinear optimization
        refined_points_3d = np.zeros_like(initial_points_3d)
        
        for i, point_3d in enumerate(initial_points_3d):
            # Extract corresponding 2D points for this 3D point
            current_points_2d = [points[i] for points in points_2d_list]
            
            # Refine the point
            refined_point = self._refine_point(point_3d[:3], current_points_2d)
            refined_points_3d[i] = refined_point
        
        return refined_points_3d
    
    def triangulate_point(self, points_2d: List[Union[Tuple[float, float], np.ndarray]]) -> Optional[np.ndarray]:
        """
        Triangulate a single 3D point from corresponding 2D points.
        
        Args:
            points_2d: List of 2D points from multiple camera views
            
        Returns:
            3D point as numpy array [X, Y, Z] or None if triangulation fails
        """
        if not self.is_calibrated:
            raise RuntimeError("Triangulator not calibrated. Set camera parameters first.")
        
        # Convert points to numpy arrays if they're not already
        points_2d_np = [np.array(p) if not isinstance(p, np.ndarray) else p for p in points_2d]
        
        # Use linear triangulation for initial estimate
        initial_point_3d = self.linear_triangulator.triangulate_point(points_2d_np)
        
        if initial_point_3d is None:
            logger.warning("Linear triangulation failed, cannot perform nonlinear refinement")
            return None
        
        # Refine using nonlinear optimization
        refined_point = self._refine_point(initial_point_3d[:3], points_2d_np)
        
        return refined_point
    
    def _refine_point(self, point_3d_initial: np.ndarray, 
                     points_2d: List[np.ndarray]) -> np.ndarray:
        """
        Refine a 3D point using nonlinear optimization.
        
        Args:
            point_3d_initial: Initial estimate of 3D point
            points_2d: List of 2D points from multiple views
            
        Returns:
            Refined 3D point
        """
        # Define the cost function for optimization (reprojection error)
        def cost_function(point):
            return self._reprojection_error_cost(point, points_2d)
        
        # Run the optimization
        result = optimize.least_squares(
            cost_function,
            point_3d_initial,
            method=self.optimization_method,
            ftol=1e-5,
            xtol=1e-5,
            max_nfev=100
        )
        
        # Return the optimized point
        return result.x
    
    def _reprojection_error_cost(self, point_3d: np.ndarray, 
                               points_2d: List[np.ndarray]) -> np.ndarray:
        """
        Compute reprojection error for all points as a cost function for optimization.
        
        Args:
            point_3d: 3D point to evaluate
            points_2d: List of observed 2D points
            
        Returns:
            Array of reprojection errors (flattened)
        """
        # Ensure we have projection matrices
        if len(self.projection_matrices) == 0:
            raise ValueError("Projection matrices not set")
        
        # Convert point to homogeneous coordinates
        point_3d_h = np.append(point_3d, 1.0)
        
        # Calculate reprojection errors
        errors = []
        
        for i, (point_2d, projection_matrix) in enumerate(zip(points_2d, self.projection_matrices)):
            # Project 3D point to 2D
            projected = projection_matrix.dot(point_3d_h)
            projected = projected[:2] / projected[2]  # Normalize and keep only x,y
            
            # Compute error (x and y separately for optimization)
            error = point_2d - projected
            errors.extend(error)
        
        return np.array(errors)
    
    def calculate_reprojection_error(self, point_3d: np.ndarray, 
                                    points_2d: List[Union[Tuple[float, float], np.ndarray]]) -> float:
        """
        Calculate the reprojection error for a triangulated point.
        
        Args:
            point_3d: Triangulated 3D point
            points_2d: List of 2D points used for triangulation
            
        Returns:
            Mean reprojection error in pixels
        """
        if not self.is_calibrated:
            raise RuntimeError("Triangulator not calibrated. Set camera parameters first.")
        
        # Convert points to numpy arrays if needed
        points_2d_np = [np.array(p) if not isinstance(p, np.ndarray) else p for p in points_2d]
        
        # Calculate error
        error_vector = self._reprojection_error_cost(point_3d, points_2d_np)
        
        # Compute the root mean square error
        rmse = np.sqrt(np.mean(np.square(error_vector)))
        
        return rmse 