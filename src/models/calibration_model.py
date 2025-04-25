"""
CalibrationModel module.
This module defines the CalibrationModel class for storing and managing calibration points.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

class CalibrationModel:
    """
    Model for storing left and right calibration points.
    """

    def __init__(self):
        """
        Initialize empty lists for left and right points.
        """
        self.left_pts = []  # list of (x, y) tuples for left image
        self.right_pts = []  # list of (x, y) tuples for right image

    def add_point(self, side: str, point: tuple[float, float]) -> None:
        """
        Add a calibration point to the specified side.

        Args:
            side (str): 'left' or 'right'
            point (tuple[float, float]): (x, y) coordinates
        """
        if side == 'left':
            self.left_pts.append(point)
        elif side == 'right':
            self.right_pts.append(point)
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
            self.left_pts[index] = point
        elif side == 'right':
            self.right_pts[index] = point
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
            }
        }
        
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