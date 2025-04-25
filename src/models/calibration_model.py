#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calibration Model module.
This module contains the CalibrationModel class that stores the calibration points
for left and right images in a stereo setup.
"""

import logging
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

class CalibrationModel:
    """
    Model class for court calibration points.
    Stores and manages calibration points for left and right stereo images.
    """
    
    def __init__(self):
        """Initialize the calibration model with empty point lists."""
        # Initialize empty lists for left and right calibration points
        self.left_pts: List[Tuple[float, float]] = []
        self.right_pts: List[Tuple[float, float]] = []
        
        # Maximum number of calibration points
        self.max_points = 14
        
        logger.debug("CalibrationModel initialized")
    
    def add_point(self, side: str, point: Tuple[float, float]) -> bool:
        """
        Add a calibration point to the specified side.
        
        Args:
            side (str): 'left' or 'right' side
            point (Tuple[float, float]): (x, y) coordinates of the point
            
        Returns:
            bool: True if point was added, False if maximum points reached
        """
        if side not in ['left', 'right']:
            logger.error(f"Invalid side specified: {side}")
            return False
        
        points_list = self.left_pts if side == 'left' else self.right_pts
        
        if len(points_list) >= self.max_points:
            logger.warning(f"Maximum number of points ({self.max_points}) reached for {side} side")
            return False
        
        # Add the point to the appropriate list
        if side == 'left':
            self.left_pts.append(point)
            logger.debug(f"Added point {point} to left side (total: {len(self.left_pts)})")
        else:
            self.right_pts.append(point)
            logger.debug(f"Added point {point} to right side (total: {len(self.right_pts)})")
        
        return True
    
    def update_point(self, side: str, index: int, point: Tuple[float, float]) -> bool:
        """
        Update an existing calibration point.
        
        Args:
            side (str): 'left' or 'right' side
            index (int): Index of the point to update
            point (Tuple[float, float]): New (x, y) coordinates
            
        Returns:
            bool: True if point was updated, False otherwise
        """
        if side not in ['left', 'right']:
            logger.error(f"Invalid side specified: {side}")
            return False
        
        points_list = self.left_pts if side == 'left' else self.right_pts
        
        if index < 0 or index >= len(points_list):
            logger.error(f"Invalid point index {index} for {side} side (size: {len(points_list)})")
            return False
        
        # Update the point
        if side == 'left':
            self.left_pts[index] = point
            logger.debug(f"Updated point at index {index} to {point} on left side")
        else:
            self.right_pts[index] = point
            logger.debug(f"Updated point at index {index} to {point} on right side")
        
        return True
    
    def remove_point(self, side: str, index: int) -> bool:
        """
        Remove a calibration point.
        
        Args:
            side (str): 'left' or 'right' side
            index (int): Index of the point to remove
            
        Returns:
            bool: True if point was removed, False otherwise
        """
        if side not in ['left', 'right']:
            logger.error(f"Invalid side specified: {side}")
            return False
        
        points_list = self.left_pts if side == 'left' else self.right_pts
        
        if index < 0 or index >= len(points_list):
            logger.error(f"Invalid point index {index} for {side} side (size: {len(points_list)})")
            return False
        
        # Remove the point
        if side == 'left':
            removed = self.left_pts.pop(index)
            logger.debug(f"Removed point {removed} at index {index} from left side")
        else:
            removed = self.right_pts.pop(index)
            logger.debug(f"Removed point {removed} at index {index} from right side")
        
        return True
    
    def clear_points(self, side: str = None) -> None:
        """
        Clear all calibration points for the specified side or both sides.
        
        Args:
            side (str, optional): 'left', 'right', or None for both sides
        """
        if side is None:
            self.left_pts.clear()
            self.right_pts.clear()
            logger.debug("Cleared all points on both sides")
        elif side == "left":
            self.left_pts.clear()
            logger.debug("Cleared all points on left side")
        elif side == "right":
            self.right_pts.clear()
            logger.debug("Cleared all points on right side")
        else:
            logger.warning(f"Invalid side specified for clearing: {side}")
    
    def get_points(self, side: str) -> List[Tuple[float, float]]:
        """
        Get all calibration points for the specified side.
        
        Args:
            side (str): 'left' or 'right' side
            
        Returns:
            List[Tuple[float, float]]: List of (x, y) coordinates
        """
        if side == 'left':
            return self.left_pts.copy()
        elif side == 'right':
            return self.right_pts.copy()
        else:
            logger.error(f"Invalid side specified: {side}")
            return []
    
    def normalize_points_to_1080p(self, points: List[Tuple[float, float]], current_width: int, current_height: int) -> List[Tuple[float, float]]:
        """
        Normalize points from current resolution to 1080p standard coordinates.
        
        Args:
            points: List of (x, y) coordinates in current resolution
            current_width: Current image width
            current_height: Current image height
            
        Returns:
            List of points normalized to 1080p standard
        """
        normalized_points = []
        target_width, target_height = 1920, 1080  # 1080p standard resolution
        
        for x, y in points:
            # Convert from current resolution to 1080p
            normalized_x = (x / current_width) * target_width
            normalized_y = (y / current_height) * target_height
            normalized_points.append((normalized_x, normalized_y))
            
        logger.debug(f"Normalized {len(points)} points from resolution {current_width}x{current_height} to 1080p standard")
        return normalized_points

    def denormalize_points_from_1080p(self, normalized_points: List[Tuple[float, float]], target_width: int, target_height: int) -> List[Tuple[float, float]]:
        """
        Denormalize points from 1080p standard to the current screen resolution.
        
        Args:
            normalized_points: List of points in 1080p standard
            target_width: Target image width
            target_height: Target image height
            
        Returns:
            List of points converted to current screen resolution
        """
        points = []
        source_width, source_height = 1920, 1080  # 1080p standard resolution
        
        for x, y in normalized_points:
            # Convert from 1080p to current resolution
            screen_x = (x / source_width) * target_width
            screen_y = (y / source_height) * target_height
            points.append((screen_x, screen_y))
            
        logger.debug(f"Denormalized {len(normalized_points)} points from 1080p standard to resolution {target_width}x{target_height}")
        return points
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the model
        """
        return {
            "points": {
                "left": self.left_pts,
                "right": self.right_pts
            },
            "calib_ver": 1.0
        }
    
    def from_dict(self, data: Dict[str, Any]) -> bool:
        """
        Load the model from a dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary containing calibration data
            
        Returns:
            bool: True if data was loaded successfully, False otherwise
        """
        try:
            # Check for old format
            if "raw_points" in data:
                # Handle migration from old format
                if "left" in data["raw_points"] and "right" in data["raw_points"]:
                    self.left_pts = data["raw_points"]["left"]
                    self.right_pts = data["raw_points"]["right"]
                    logger.info("Migrated from old format ('raw_points') to new format ('points')")
                    return True
            
            # New format
            if "points" in data:
                if "left" in data["points"] and "right" in data["points"]:
                    self.left_pts = data["points"]["left"]
                    self.right_pts = data["points"]["right"]
                    logger.info("Loaded calibration points from dict")
                    return True
            
            logger.error("Invalid data format for calibration model")
            return False
        except Exception as e:
            logger.error(f"Error loading calibration data: {e}")
            return False 