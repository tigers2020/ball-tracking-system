#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base coordinate transformation module.
Defines the abstract base class for all coordinate transformation implementations.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union

class AbstractCoordinateTransformer(ABC):
    """
    Abstract base class for coordinate transformation implementations.
    
    This class defines the interface that all coordinate transformers must implement.
    It provides a consistent API for converting between different coordinate systems.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the coordinate transformer with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config or {}
        
        # Camera intrinsics
        self.camera_matrix = None
        
        # Rotation and translation for camera-to-world transformation
        self.R = np.eye(3)  # Default to identity
        self.T = np.zeros(3)  # Default to origin
        
        # Scale factor (e.g., meters per pixel)
        self.scale = 1.0
        
        # Set parameters if provided
        if config:
            self.set_parameters(config)
    
    @abstractmethod
    def set_parameters(self, config: Dict[str, Any]) -> None:
        """
        Set parameters for coordinate transformation.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        pass
    
    @abstractmethod
    def to_world(self, u: float, v: float, depth: float) -> Tuple[float, float, float]:
        """
        Convert a point from image space to world space.
        
        Args:
            u: x-coordinate in image space (pixels)
            v: y-coordinate in image space (pixels)
            depth: depth value (could be disparity or actual depth)
            
        Returns:
            3D point in world coordinates (X, Y, Z)
        """
        pass
    
    @abstractmethod
    def to_image(self, X: float, Y: float, Z: float) -> Tuple[float, float]:
        """
        Convert a point from world space to image space.
        
        Args:
            X: x-coordinate in world space
            Y: y-coordinate in world space
            Z: z-coordinate in world space
            
        Returns:
            2D point in image coordinates (u, v)
        """
        pass
    
    def transform_point(self, point: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """
        Transform a point using rotation and translation.
        
        Args:
            point: 3D point [x, y, z]
            rotation: 3x3 rotation matrix
            translation: 3x1 or 3-element translation vector
            
        Returns:
            Transformed 3D point
        """
        # Ensure point is a vector
        point = np.array(point).flatten()
        
        # Ensure translation is a vector
        translation = np.array(translation).flatten()
        
        # Apply rotation and translation
        transformed_point = rotation @ point + translation
        
        return transformed_point
    
    def pixel_to_ray(self, u: float, v: float) -> np.ndarray:
        """
        Convert a pixel coordinate to a ray in camera space.
        
        Args:
            u: x-coordinate in image space (pixels)
            v: y-coordinate in image space (pixels)
            
        Returns:
            Ray direction vector [dx, dy, dz] in camera space
        """
        if self.camera_matrix is None:
            raise ValueError("Camera matrix not set")
            
        # Get camera intrinsics
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        # Calculate ray direction in camera space
        x = (u - cx) / fx
        y = (v - cy) / fy
        z = 1.0
        
        # Normalize ray direction
        ray = np.array([x, y, z])
        ray = ray / np.linalg.norm(ray)
        
        return ray
    
    def ray_plane_intersection(self, camera_center: np.ndarray, ray_direction: np.ndarray, 
                              plane_point: np.ndarray = np.array([0, 0, 0]), 
                              plane_normal: np.ndarray = np.array([0, 0, 1])) -> Optional[np.ndarray]:
        """
        Calculate the intersection of a ray with a plane.
        
        Args:
            camera_center: Camera center in world coordinates [X, Y, Z]
            ray_direction: Ray direction vector [dx, dy, dz]
            plane_point: A point on the plane [X, Y, Z], default is origin
            plane_normal: Normal vector to the plane [nx, ny, nz], default is Z=0 plane
            
        Returns:
            Intersection point in world coordinates [X, Y, Z] or None if no intersection
        """
        # Normalize vectors
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        
        # Check if ray is parallel to the plane
        denom = np.dot(ray_direction, plane_normal)
        if abs(denom) < 1e-6:
            return None  # Ray is parallel to the plane
        
        # Calculate distance along the ray to intersection
        t = np.dot(plane_point - camera_center, plane_normal) / denom
        
        # Ray points away from the plane
        if t < 0:
            return None
        
        # Calculate intersection point
        intersection = camera_center + t * ray_direction
        
        return intersection 