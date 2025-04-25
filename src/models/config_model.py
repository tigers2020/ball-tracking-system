#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Config Model module.
This module contains classes for configuration-related data models.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class CalibrationConfig:
    """
    Model class for calibration configuration.
    Stores and manages calibration data including points and image dimensions.
    """
    
    def __init__(self):
        """Initialize the calibration configuration model."""
        # Calibration points for left and right images
        self.left_points = {}  # Dict[int, Tuple[float, float]]
        self.right_points = {}  # Dict[int, Tuple[float, float]]
        
        # Image dimensions
        self.left_image_width = 0
        self.left_image_height = 0
        self.right_image_width = 0
        self.right_image_height = 0
        
        # Image paths
        self.left_image_path = None
        self.right_image_path = None
        
        # Version
        self.version = 1.2
    
    def clear(self):
        """Clear all calibration data."""
        self.left_points.clear()
        self.right_points.clear()
        self.left_image_width = 0
        self.left_image_height = 0
        self.right_image_width = 0
        self.right_image_height = 0
        self.left_image_path = None
        self.right_image_path = None
    
    def set_image_dimensions(self, side: str, width: int, height: int):
        """
        Set image dimensions for a specific side.
        
        Args:
            side (str): 'left' or 'right'
            width (int): Image width
            height (int): Image height
        """
        if side not in ['left', 'right']:
            logger.error(f"Invalid side: {side}")
            return
            
        if side == 'left':
            self.left_image_width = width
            self.left_image_height = height
        else:
            self.right_image_width = width
            self.right_image_height = height
    
    def set_image_path(self, side: str, path: str):
        """
        Set the image path for a specific side.
        
        Args:
            side (str): 'left' or 'right'
            path (str): Path to the image file
        """
        if side not in ['left', 'right']:
            logger.error(f"Invalid side: {side}")
            return
            
        if side == 'left':
            self.left_image_path = path
        else:
            self.right_image_path = path
    
    def add_point(self, side: str, point_id: int, x: float, y: float):
        """
        Add or update a calibration point.
        
        Args:
            side (str): 'left' or 'right'
            point_id (int): Point identifier
            x (float): X coordinate
            y (float): Y coordinate
        """
        if side not in ['left', 'right']:
            logger.error(f"Invalid side: {side}")
            return
            
        if side == 'left':
            self.left_points[point_id] = (x, y)
        else:
            self.right_points[point_id] = (x, y)
    
    def remove_point(self, side: str, point_id: int):
        """
        Remove a calibration point.
        
        Args:
            side (str): 'left' or 'right'
            point_id (int): Point identifier
        """
        if side not in ['left', 'right']:
            logger.error(f"Invalid side: {side}")
            return
            
        if side == 'left':
            if point_id in self.left_points:
                del self.left_points[point_id]
        else:
            if point_id in self.right_points:
                del self.right_points[point_id]
    
    def get_points(self, side: str) -> Dict[int, Tuple[float, float]]:
        """
        Get all points for a specific side.
        
        Args:
            side (str): 'left' or 'right'
            
        Returns:
            Dict[int, Tuple[float, float]]: Dictionary of points (point_id: (x, y))
        """
        if side not in ['left', 'right']:
            logger.error(f"Invalid side: {side}")
            return {}
            
        return self.left_points if side == 'left' else self.right_points
    
    def get_point(self, side: str, point_id: int) -> Optional[Tuple[float, float]]:
        """
        Get a specific point.
        
        Args:
            side (str): 'left' or 'right'
            point_id (int): Point identifier
            
        Returns:
            Optional[Tuple[float, float]]: Point coordinates (x, y) or None if not found
        """
        if side not in ['left', 'right']:
            logger.error(f"Invalid side: {side}")
            return None
            
        points = self.left_points if side == 'left' else self.right_points
        return points.get(point_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert calibration configuration to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the calibration configuration
        """
        return {
            "left": self.left_points,
            "right": self.right_points,
            "left_image_size": {"width": self.left_image_width, "height": self.left_image_height},
            "right_image_size": {"width": self.right_image_width, "height": self.right_image_height},
            "left_image_path": self.left_image_path,
            "right_image_path": self.right_image_path,
            "calib_ver": self.version
        }
    
    def from_dict(self, data: Dict[str, Any]):
        """
        Update calibration configuration from a dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary with calibration configuration
        """
        if not data:
            return
            
        if "left" in data:
            self.left_points = data["left"]
            
        if "right" in data:
            self.right_points = data["right"]
            
        if "left_image_size" in data:
            size = data["left_image_size"]
            self.left_image_width = size.get("width", 0)
            self.left_image_height = size.get("height", 0)
            
        if "right_image_size" in data:
            size = data["right_image_size"]
            self.right_image_width = size.get("width", 0)
            self.right_image_height = size.get("height", 0)
            
        if "left_image_path" in data:
            self.left_image_path = data["left_image_path"]
            
        if "right_image_path" in data:
            self.right_image_path = data["right_image_path"]
            
        if "calib_ver" in data:
            self.version = data["calib_ver"] 