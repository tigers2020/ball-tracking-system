"""
CalibrationModel module.
This module defines the CalibrationModel class for storing and managing calibration points.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class CalibrationModel:
    """
    Model for storing left and right calibration points.
    """

    # Maximum number of calibration points (fixed at 14)
    MAX_POINTS = 14

    def __init__(self):
        """
        Initialize empty lists for left and right points.
        """
        self.left_pts = []  # list of (x, y) tuples for left image
        self.right_pts = []  # list of (x, y) tuples for right image
        
        # Image dimensions for normalization
        self.left_image_width = 0
        self.left_image_height = 0
        self.right_image_width = 0
        self.right_image_height = 0

    def add_point(self, side: str, point: tuple[float, float]) -> None:
        """
        Add a calibration point to the specified side.

        Args:
            side (str): 'left' or 'right'
            point (tuple[float, float]): (x, y) coordinates
        """
        if side == 'left':
            # Only add if we haven't reached the maximum
            if len(self.left_pts) < self.MAX_POINTS:
                self.left_pts.append(point)
            else:
                logger.warning(f"Maximum number of points ({self.MAX_POINTS}) reached for left side. Point not added.")
        elif side == 'right':
            # Only add if we haven't reached the maximum
            if len(self.right_pts) < self.MAX_POINTS:
                self.right_pts.append(point)
            else:
                logger.warning(f"Maximum number of points ({self.MAX_POINTS}) reached for right side. Point not added.")
        else:
            raise ValueError(f"Invalid side: {side}")

    def update_point(self, side: str, index: int, point: tuple[float, float]) -> None:
        """
        Update an existing calibration point.

        Args:
            side (str): 'left' or 'right'
            index (int): Index of the point to update
            point (tuple[float, float]): New (x, y) coordinates
        """
        if side == 'left':
            if 0 <= index < len(self.left_pts):
                self.left_pts[index] = point
            else:
                raise IndexError(f"Invalid index {index} for left points (length: {len(self.left_pts)})")
        elif side == 'right':
            if 0 <= index < len(self.right_pts):
                self.right_pts[index] = point
            else:
                raise IndexError(f"Invalid index {index} for right points (length: {len(self.right_pts)})")
        else:
            raise ValueError(f"Invalid side: {side}")

    def get_points(self, side: str) -> list[tuple[float, float]]:
        """
        Retrieve all points for the specified side.

        Args:
            side (str): 'left' or 'right'

        Returns:
            list[tuple[float, float]]: List of points
        """
        if side == 'left':
            return self.left_pts
        elif side == 'right':
            return self.right_pts
        else:
            raise ValueError(f"Invalid side: {side}")

    def clear_points(self) -> None:
        """
        Clear all points from both sides.
        """
        self.left_pts.clear()
        self.right_pts.clear()
        
    def set_image_dimensions(self, side: str, width: int, height: int) -> None:
        """
        Set the image dimensions for the specified side.
        
        Args:
            side (str): 'left' or 'right'
            width (int): Image width in pixels
            height (int): Image height in pixels
        """
        if side == 'left':
            self.left_image_width = width
            self.left_image_height = height
        elif side == 'right':
            self.right_image_width = width
            self.right_image_height = height
        else:
            raise ValueError(f"Invalid side: {side}")
        
        logger.info(f"Set {side} image dimensions to {width}x{height}")
        
    def get_image_dimensions(self, side: str) -> tuple[int, int]:
        """
        Get the image dimensions for the specified side.
        
        Args:
            side (str): 'left' or 'right'
            
        Returns:
            tuple[int, int]: (width, height) of the image
        """
        if side == 'left':
            return (self.left_image_width, self.left_image_height)
        elif side == 'right':
            return (self.right_image_width, self.right_image_height)
        else:
            raise ValueError(f"Invalid side: {side}")
    
    def normalize_point(self, side: str, point: tuple[float, float]) -> tuple[float, float]:
        """
        Normalize a point to 0-1 range based on image dimensions.
        
        Args:
            side (str): 'left' or 'right'
            point (tuple[float, float]): Point coordinates in pixels
            
        Returns:
            tuple[float, float]: Normalized point coordinates (0-1 range)
        """
        x, y = point
        
        if side == 'left':
            if self.left_image_width == 0 or self.left_image_height == 0:
                logger.warning("Left image dimensions not set, returning unnormalized coordinates")
                return point
                
            norm_x = x / self.left_image_width
            norm_y = y / self.left_image_height
        elif side == 'right':
            if self.right_image_width == 0 or self.right_image_height == 0:
                logger.warning("Right image dimensions not set, returning unnormalized coordinates")
                return point
                
            norm_x = x / self.right_image_width
            norm_y = y / self.right_image_height
        else:
            raise ValueError(f"Invalid side: {side}")
            
        return (norm_x, norm_y)
    
    def denormalize_point(self, side: str, norm_point: tuple[float, float]) -> tuple[float, float]:
        """
        Convert a normalized point (0-1) to pixel coordinates.
        
        Args:
            side (str): 'left' or 'right'
            norm_point (tuple[float, float]): Normalized point coordinates (0-1 range)
            
        Returns:
            tuple[float, float]: Point coordinates in pixels
        """
        norm_x, norm_y = norm_point
        
        if side == 'left':
            if self.left_image_width == 0 or self.left_image_height == 0:
                logger.warning("Left image dimensions not set, returning normalized coordinates")
                return norm_point
                
            x = norm_x * self.left_image_width
            y = norm_y * self.left_image_height
        elif side == 'right':
            if self.right_image_width == 0 or self.right_image_height == 0:
                logger.warning("Right image dimensions not set, returning normalized coordinates")
                return norm_point
                
            x = norm_x * self.right_image_width
            y = norm_y * self.right_image_height
        else:
            raise ValueError(f"Invalid side: {side}")
            
        return (x, y)
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the calibration points to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of calibration data
        """
        return {
            "points": {
                "left": self.left_pts,
                "right": self.right_pts
            },
            "left_image_size": {
                "width": self.left_image_width,
                "height": self.left_image_height
            },
            "right_image_size": {
                "width": self.right_image_width,
                "height": self.right_image_height
            }
        }
    
    def to_normalized_dict(self) -> Dict[str, Any]:
        """
        Convert the calibration points to a normalized dictionary format (0-1 range)
        with point IDs (p00, p01, etc.) as keys.
        
        Returns:
            Dict[str, Any]: Dictionary with normalized calibration data
        """
        left_points = {}
        right_points = {}
        
        # Normalize left points
        for i, point in enumerate(self.left_pts):
            if i >= self.MAX_POINTS:
                break
                
            point_id = f"p{i:02d}"
            norm_x, norm_y = self.normalize_point('left', point)
            left_points[point_id] = {
                "x": norm_x,
                "y": norm_y,
                "is_fine_tuned": False  # Default to false
            }
        
        # Normalize right points
        for i, point in enumerate(self.right_pts):
            if i >= self.MAX_POINTS:
                break
                
            point_id = f"p{i:02d}"
            norm_x, norm_y = self.normalize_point('right', point)
            right_points[point_id] = {
                "x": norm_x,
                "y": norm_y,
                "is_fine_tuned": False  # Default to false
            }
        
        return {
            "left": left_points,
            "right": right_points,
            "left_image_size": {
                "width": self.left_image_width,
                "height": self.left_image_height
            },
            "right_image_size": {
                "width": self.right_image_width,
                "height": self.right_image_height
            },
            "left_image_path": None,  # Could be added in the future
            "right_image_path": None,  # Could be added in the future
            "calib_ver": 1.2
        }
    
    def from_normalized_dict(self, data: Dict[str, Any]) -> None:
        """
        Load calibration points from a normalized dictionary format.
        
        Args:
            data (Dict[str, Any]): Dictionary with normalized calibration data
        """
        # Clear existing points
        self.clear_points()
        
        # Set image dimensions
        if "left_image_size" in data:
            self.left_image_width = data["left_image_size"].get("width", 0)
            self.left_image_height = data["left_image_size"].get("height", 0)
            
        if "right_image_size" in data:
            self.right_image_width = data["right_image_size"].get("width", 0)
            self.right_image_height = data["right_image_size"].get("height", 0)
        
        # Load left points
        if "left" in data:
            left_data = data["left"]
            
            # Sort point IDs to ensure order is maintained
            point_ids = sorted(left_data.keys())
            
            for point_id in point_ids:
                point_data = left_data[point_id]
                if "x" in point_data and "y" in point_data:
                    norm_x = point_data["x"]
                    norm_y = point_data["y"]
                    
                    # Denormalize to pixel coordinates
                    x, y = self.denormalize_point('left', (norm_x, norm_y))
                    self.left_pts.append((x, y))
        
        # Load right points
        if "right" in data:
            right_data = data["right"]
            
            # Sort point IDs to ensure order is maintained
            point_ids = sorted(right_data.keys())
            
            for point_id in point_ids:
                point_data = right_data[point_id]
                if "x" in point_data and "y" in point_data:
                    norm_x = point_data["x"]
                    norm_y = point_data["y"]
                    
                    # Denormalize to pixel coordinates
                    x, y = self.denormalize_point('right', (norm_x, norm_y))
                    self.right_pts.append((x, y))
        
        logger.info(f"Loaded {len(self.left_pts)} left points and {len(self.right_pts)} right points from normalized data")
        
    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Load calibration points from a dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary containing calibration data
        """
        if "points" not in data:
            raise ValueError("Invalid calibration data: 'points' key not found")
            
        points_data = data["points"]
        
        if "left" in points_data:
            self.left_pts = [tuple(point) for point in points_data["left"]]
        
        if "right" in points_data:
            self.right_pts = [tuple(point) for point in points_data["right"]]
            
        logger.info(f"Loaded {len(self.left_pts)} left points and {len(self.right_pts)} right points")
        
    def save_to_file(self, file_path: str) -> bool:
        """
        Save calibration points to a JSON file.
        
        Args:
            file_path (str): Path to the save file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            data = self.to_dict()
            
            # Ensure directory exists
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
                
            logger.info(f"Saved calibration data to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving calibration data: {e}")
            return False
            
    def load_from_file(self, file_path: str) -> bool:
        """
        Load calibration points from a JSON file.
        
        Args:
            file_path (str): Path to the save file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                logger.warning(f"Calibration file not found: {file_path}")
                return False
                
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            self.from_dict(data)
            return True
        except Exception as e:
            logger.error(f"Error loading calibration data: {e}")
            return False 