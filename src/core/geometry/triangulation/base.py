#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Abstract triangulation base module.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np

logger = logging.getLogger(__name__)

class AbstractTriangulator(ABC):
    """
    Abstract base class for triangulation implementations.
    
    This class defines the interface that all triangulator classes must implement.
    Triangulation is the process of determining 3D point coordinates from multiple
    2D point correspondences across different camera views.
    """
    
    def __init__(self, camera_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the triangulator.
        
        Args:
            camera_params: Dictionary containing camera parameters
        """
        self.camera_params = camera_params
        self.is_calibrated = False
        
    @abstractmethod
    def set_camera_parameters(self, camera_params: Dict[str, Any]) -> None:
        """
        Set camera parameters for triangulation.
        
        Args:
            camera_params: Dictionary containing camera parameters
        """
        pass
        
    @abstractmethod
    def triangulate_points(self, points_2d_list: List[np.ndarray]) -> np.ndarray:
        """
        Triangulate multiple 3D points from corresponding points in multiple views.
        
        Args:
            points_2d_list: List containing arrays of 2D points from multiple views
            
        Returns:
            Array of triangulated 3D points
        """
        pass
        
    @abstractmethod
    def triangulate_point(self, 
                         points_2d: List[Union[Tuple[float, float], np.ndarray]]) -> Optional[np.ndarray]:
        """
        Triangulate a single 3D point from corresponding 2D points.
        
        Args:
            points_2d: List of 2D points from multiple camera views
            
        Returns:
            3D point as numpy array [X, Y, Z] or None if triangulation fails
        """
        pass
        
    @abstractmethod
    def calculate_reprojection_error(self, 
                                    point_3d: np.ndarray, 
                                    points_2d: List[Union[Tuple[float, float], np.ndarray]]) -> float:
        """
        Calculate the reprojection error for a triangulated point.
        
        Args:
            point_3d: Triangulated 3D point
            points_2d: List of 2D points used for triangulation
            
        Returns:
            Reprojection error metric
        """
        pass
    
    def is_ready(self) -> bool:
        """
        Check if the triangulator is calibrated and ready to use.
        
        Returns:
            True if calibrated, False otherwise
        """
        return self.is_calibrated
    
    def get_camera_matrix(self) -> Optional[np.ndarray]:
        """
        Get the camera matrix.
        
        Returns:
            Camera matrix or None if not available
        """
        return self.camera_matrix
    
    def get_projection_matrices(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get the projection matrices for left and right cameras.
        
        Returns:
            Tuple of (left_projection_matrix, right_projection_matrix)
        """
        return self.P_left, self.P_right
    
    def reset(self) -> None:
        """
        Reset the triangulator state.
        """
        self.is_calibrated = False
        self.camera_matrix = None
        self.P_left = None
        self.P_right = None
    
    def is_valid_point(self, point_3d: np.ndarray, max_depth: float = 100.0, min_depth: float = 0.1) -> bool:
        """
        Check if a triangulated 3D point is valid.
        
        Args:
            point_3d: 3D point [X, Y, Z]
            max_depth: Maximum allowed depth (Z) value
            min_depth: Minimum allowed depth (Z) value
            
        Returns:
            True if point is valid, False otherwise
        """
        if point_3d is None:
            return False
            
        # Check for NaN or infinity
        if not np.all(np.isfinite(point_3d)):
            return False
            
        # Check depth (Z) value
        z = point_3d[2]
        if z < min_depth or z > max_depth:
            return False
            
        return True
    
    def get_calibration_status(self) -> Dict[str, Any]:
        """
        Get the current calibration status.
        
        Returns:
            Dictionary with calibration status
        """
        status = {
            "is_calibrated": self.is_calibrated,
        }
        
        if self.is_calibrated:
            status.update({
                "camera_matrix": self.camera_matrix.tolist() if self.camera_matrix is not None else None,
            })
            
        return status 