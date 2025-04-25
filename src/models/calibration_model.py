#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calibration Model module.
This module contains the CalibrationModel class which stores and manages calibration point data.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

from src.utils.geometry import pixel_to_normalized, normalized_to_pixel

logger = logging.getLogger(__name__)

class CalibrationModel:
    """
    Model for calibration points data.
    Stores and manages calibration points for left and right camera views.
    """
    
    # Maximum number of calibration points
    MAX_POINTS = 14
    
    def __init__(self):
        """Initialize the CalibrationModel with empty points lists and default image dimensions."""
        # Dictionary to store points for each side
        # Each side has a list of (x, y) coordinates in pixel space
        self._points = {
            'left': [],
            'right': []
        }
        
        # Store image dimensions for normalizing/denormalizing coordinates
        # Each side has (width, height) in pixels
        self._image_dimensions = {
            'left': (1, 1),  # Default to 1x1 to avoid division by zero
            'right': (1, 1)
        }
    
    def add_point(self, side: str, point: Tuple[float, float]) -> int:
        """
        Add a calibration point for the specified side.
        
        Args:
            side (str): 'left' or 'right'
            point (Tuple[float, float]): (x, y) coordinates in pixel space
            
        Returns:
            int: Index of the added point
        """
        if side not in ['left', 'right']:
            logger.error(f"Invalid side specified: {side}")
            return -1
            
        if len(self._points[side]) >= self.MAX_POINTS:
            logger.warning(f"Cannot add more than {self.MAX_POINTS} points to {side} side")
            return -1
            
        # Add the point to the list for the specified side
        self._points[side].append(point)
        
        # Return the index of the added point
        return len(self._points[side]) - 1
    
    def update_point(self, side: str, index: int, point: Tuple[float, float]) -> bool:
        """
        Update an existing calibration point.
        
        Args:
            side (str): 'left' or 'right'
            index (int): Index of the point to update
            point (Tuple[float, float]): New (x, y) coordinates in pixel space
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        if side not in ['left', 'right']:
            logger.error(f"Invalid side specified: {side}")
            return False
            
        if index < 0 or index >= len(self._points[side]):
            logger.error(f"Invalid point index: {index}")
            return False
            
        # Update the point at the specified index
        self._points[side][index] = point
        return True
    
    def get_points(self, side: str) -> List[Tuple[float, float]]:
        """
        Get all calibration points for the specified side.
        
        Args:
            side (str): 'left' or 'right'
            
        Returns:
            List[Tuple[float, float]]: List of (x, y) coordinates in pixel space
        """
        if side not in ['left', 'right']:
            logger.error(f"Invalid side specified: {side}")
            return []
            
        return self._points[side]
    
    def get_point(self, side: str, index: int) -> Optional[Tuple[float, float]]:
        """
        Get a specific calibration point.
        
        Args:
            side (str): 'left' or 'right'
            index (int): Index of the point to get
            
        Returns:
            Optional[Tuple[float, float]]: (x, y) coordinates in pixel space, or None if not found
        """
        if side not in ['left', 'right']:
            logger.error(f"Invalid side specified: {side}")
            return None
            
        if index < 0 or index >= len(self._points[side]):
            logger.error(f"Invalid point index: {index}")
            return None
            
        return self._points[side][index]
    
    def clear_points(self, side: Optional[str] = None) -> None:
        """
        Clear all calibration points.
        
        Args:
            side (Optional[str]): 'left', 'right', or None to clear both sides
        """
        if side is None:
            # Clear both sides
            self._points['left'] = []
            self._points['right'] = []
            logger.info("Cleared all calibration points")
        elif side in ['left', 'right']:
            # Clear specified side
            self._points[side] = []
            logger.info(f"Cleared {side} calibration points")
        else:
            logger.error(f"Invalid side specified: {side}")
    
    def set_image_dimensions(self, side: str, width: float, height: float) -> None:
        """
        Set the image dimensions for normalizing coordinates.
        
        Args:
            side (str): 'left' or 'right'
            width (float): Image width in pixels
            height (float): Image height in pixels
        """
        if side not in ['left', 'right']:
            logger.error(f"Invalid side specified: {side}")
            return
            
        if width <= 0 or height <= 0:
            logger.error(f"Invalid dimensions: width={width}, height={height}")
            return
            
        self._image_dimensions[side] = (width, height)
        logger.debug(f"Set {side} image dimensions to ({width}, {height})")
    
    def get_image_dimensions(self, side: str) -> Tuple[float, float]:
        """
        Get the stored image dimensions.
        
        Args:
            side (str): 'left' or 'right'
            
        Returns:
            Tuple[float, float]: (width, height) in pixels
        """
        if side not in ['left', 'right']:
            logger.error(f"Invalid side specified: {side}")
            return (1, 1)
            
        return self._image_dimensions[side]
    
    def to_normalized_dict(self) -> Dict:
        """
        Convert calibration points to a dictionary with normalized coordinates (0-1 range).
        
        Returns:
            Dict: Dictionary with normalized coordinates
        """
        normalized_data = {
            'left': [],
            'right': []
        }
        
        for side in ['left', 'right']:
            width, height = self._image_dimensions[side]
            
            for point in self._points[side]:
                # Use geometry utility to normalize coordinates
                norm_x, norm_y = pixel_to_normalized(point, width, height)
                normalized_data[side].append((norm_x, norm_y))
        
        return normalized_data
    
    def from_normalized_dict(self, normalized_data: Dict) -> None:
        """
        Load calibration points from a dictionary with normalized coordinates (0-1 range).
        
        Args:
            normalized_data (Dict): Dictionary with normalized coordinates
        """
        # Clear existing points
        self.clear_points()
        
        for side in ['left', 'right']:
            if side not in normalized_data:
                logger.warning(f"No {side} data found in normalized data")
                continue
                
            width, height = self._image_dimensions[side]
            
            for norm_point in normalized_data[side]:
                if not isinstance(norm_point, (list, tuple)) or len(norm_point) != 2:
                    logger.warning(f"Invalid point format in normalized data: {norm_point}")
                    continue
                    
                # Use geometry utility to convert normalized coordinates to pixel space
                x, y = normalized_to_pixel(norm_point, width, height)
                self.add_point(side, (x, y))
        
        logger.info(f"Loaded {len(self._points['left'])} left points and {len(self._points['right'])} right points from normalized data")
    
    def to_dict(self) -> Dict:
        """
        Convert calibration points to a dictionary.
        
        Returns:
            Dict: Dictionary with pixel coordinates and image dimensions
        """
        return {
            'points': {
                'left': self._points['left'],
                'right': self._points['right']
            },
            'image_dimensions': {
                'left': self._image_dimensions['left'],
                'right': self._image_dimensions['right']
            }
        }
    
    def from_dict(self, data: Dict) -> None:
        """
        Load calibration points from a dictionary.
        
        Args:
            data (Dict): Dictionary with pixel coordinates and image dimensions
        """
        # Clear existing points
        self.clear_points()
        
        # Extract image dimensions if available
        if 'image_dimensions' in data:
            for side in ['left', 'right']:
                if side in data['image_dimensions']:
                    width, height = data['image_dimensions'][side]
                    self.set_image_dimensions(side, width, height)
        
        # Extract points
        if 'points' in data:
            for side in ['left', 'right']:
                if side in data['points']:
                    for point in data['points'][side]:
                        if isinstance(point, (list, tuple)) and len(point) == 2:
                            self.add_point(side, point)
        
        logger.info(f"Loaded {len(self._points['left'])} left points and {len(self._points['right'])} right points from dictionary")
    
    def save_to_file(self, file_path: Union[str, Path]) -> bool:
        """
        Save calibration points to a JSON file.
        
        Args:
            file_path (Union[str, Path]): Path to the file to save
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Create dictionary of calibration data
            calibration_data = self.to_dict()
            
            # Ensure path is a Path object
            file_path = Path(file_path)
            
            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert points to lists for JSON serialization
            json_data = {
                'points': {
                    'left': [list(p) for p in calibration_data['points']['left']],
                    'right': [list(p) for p in calibration_data['points']['right']]
                },
                'image_dimensions': {
                    'left': list(calibration_data['image_dimensions']['left']),
                    'right': list(calibration_data['image_dimensions']['right'])
                }
            }
            
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(json_data, f, indent=4)
                
            logger.info(f"Saved calibration data to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving calibration data to file: {e}")
            return False
    
    def load_from_file(self, file_path: Union[str, Path]) -> bool:
        """
        Load calibration points from a JSON file.
        
        Args:
            file_path (Union[str, Path]): Path to the file to load
            
        Returns:
            bool: True if load was successful, False otherwise
        """
        try:
            # Ensure path is a Path object
            file_path = Path(file_path)
            
            # Check if file exists
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return False
                
            # Read from file
            with open(file_path, 'r') as f:
                json_data = json.load(f)
                
            # Convert points to tuples
            data = {
                'points': {
                    'left': [tuple(p) for p in json_data['points']['left']],
                    'right': [tuple(p) for p in json_data['points']['right']]
                },
                'image_dimensions': {
                    'left': tuple(json_data['image_dimensions']['left']),
                    'right': tuple(json_data['image_dimensions']['right'])
                }
            }
            
            # Load data
            self.from_dict(data)
            
            logger.info(f"Loaded calibration data from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading calibration data from file: {e}")
            return False
    
    def get_point_count(self, side: str) -> int:
        """
        Get the number of calibration points for the specified side.
        
        Args:
            side (str): 'left' or 'right'
            
        Returns:
            int: Number of points
        """
        if side not in ['left', 'right']:
            logger.error(f"Invalid side specified: {side}")
            return 0
            
        return len(self._points[side]) 