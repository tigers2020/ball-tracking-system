#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calibration Model module.
This module contains the CalibrationModel class which stores calibration points data.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

from PySide6.QtCore import QObject, Signal, QPointF
from PySide6.QtGui import QImage
from src.utils.config_manager import ConfigManager

# Set up logger
logger = logging.getLogger(__name__)

class CalibrationModel(QObject):
    """
    Model for storing and managing calibration points data.
    """
    
    # Signals
    points_changed = Signal(str)  # side that changed
    point_updated = Signal(str, int, QPointF, bool)  # side, index, position, is_fine_tuned
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize calibration model.
        
        Args:
            config_manager (ConfigManager, optional): Configuration manager instance
        """
        super().__init__()
        self.left_points = []
        self.right_points = []
        self.config_manager = config_manager
        
        # Store image size and path
        self.left_image_size = (0, 0)  # (width, height)
        self.right_image_size = (0, 0)  # (width, height)
        self.left_image_path = ""
        self.right_image_path = ""
        
        # 하위 호환성을 위한 alias 속성 추가
        self.left_pts = self.left_points
        self.right_pts = self.right_points
        
        # Load initial calibration if config manager is provided
        if self.config_manager:
            self._load_from_config()
    
    def add_point(self, side: str, position: QPointF) -> None:
        """
        Add a calibration point.
        
        Args:
            side (str): 'left' or 'right'
            position (QPointF): Point position
        """
        point_data = {
            'position': position,
            'is_fine_tuned': False
        }
        
        if side == "left":
            self.left_points.append(point_data)
        else:
            self.right_points.append(point_data)
        
        # Save to config if available
        if self.config_manager:
            self._save_to_config()
        
        # Emit signal
        self.points_changed.emit(side)
    
    def update_point(self, side: str, index: int, position: QPointF, is_fine_tuned: bool = False) -> None:
        """
        Update a calibration point.
        
        Args:
            side (str): 'left' or 'right'
            index (int): Point index
            position (QPointF): New position
            is_fine_tuned (bool): Whether point is fine-tuned
        """
        points = self.left_points if side == "left" else self.right_points
        
        if 0 <= index < len(points):
            points[index]['position'] = position
            if is_fine_tuned:
                points[index]['is_fine_tuned'] = True
            
            # Save to config if available
            if self.config_manager:
                self._save_to_config()
            
            # Emit signals
            self.point_updated.emit(side, index, position, points[index]['is_fine_tuned'])
            self.points_changed.emit(side)
    
    def clear_points(self) -> None:
        """Clear all calibration points."""
        logger.info("Clearing calibration points")
        
        # Keep backup before clearing
        old_left_points = self.left_points.copy()
        old_right_points = self.right_points.copy()
        
        # Clear points
        self.left_points.clear()
        self.right_points.clear()
        
        # Save to config if available
        success = False
        if self.config_manager:
            success = self._save_to_config()
            if not success:
                logger.warning("Failed to save after clearing points, restoring backup")
                # Restore backup if save failed
                self.left_points = old_left_points
                self.right_points = old_right_points
        
        # Emit signals
        self.points_changed.emit("left")
        self.points_changed.emit("right")
        
        logger.info(f"Calibration points cleared, save success: {success}")
    
    def get_points(self, side: str) -> List[Dict[str, Union[QPointF, bool]]]:
        """
        Get calibration points.
        
        Args:
            side (str): 'left' or 'right'
            
        Returns:
            List[Dict]: List of point data dictionaries
        """
        return self.left_points if side == "left" else self.right_points
    
    def set_image_path(self, side: str, path: str):
        """
        Set image path for the specified side.
        
        Args:
            side (str): 'left' or 'right'
            path (str): Path to the image file
        """
        if side == 'left':
            self.left_image_path = path
            # Read image size if file exists
            if os.path.exists(path):
                img = QImage(path)
                if not img.isNull():
                    self.left_image_size = (img.width(), img.height())
                    logger.info(f"Left image size set to {self.left_image_size}")
        elif side == 'right':
            self.right_image_path = path
            # Read image size if file exists
            if os.path.exists(path):
                img = QImage(path)
                if not img.isNull():
                    self.right_image_size = (img.width(), img.height())
                    logger.info(f"Right image size set to {self.right_image_size}")
        else:
            logger.warning(f"Invalid side parameter: {side}")
    
    def set_image_paths(self, left_path: str, right_path: str) -> None:
        """
        Set image paths for both left and right sides.
        
        Args:
            left_path (str): Path to the left image file
            right_path (str): Path to the right image file
        """
        self.set_image_path('left', left_path)
        self.set_image_path('right', right_path)
    
    def set_image_sizes(self, left_size: Tuple[int, int], right_size: Tuple[int, int]) -> None:
        """
        Set image sizes for coordinate normalization.
        
        Args:
            left_size (tuple): Left image size as (width, height)
            right_size (tuple): Right image size as (width, height)
        """
        self.left_image_size = left_size
        self.right_image_size = right_size
        
        # Update config if available
        if self.config_manager:
            self._save_to_config()
            
    def _get_normalized_point(self, position: QPointF, image_size: Tuple[int, int]) -> Dict[str, float]:
        """
        Convert pixel position to normalized position (0-1 range).
        
        Args:
            position (QPointF): Point position in pixels
            image_size (tuple): Image size as (width, height)
            
        Returns:
            dict: Normalized point coordinates as {"x": x, "y": y}
        """
        width, height = image_size
        if width <= 0 or height <= 0:
            # Return raw pixel coordinates if image size is invalid
            logger.warning("Invalid image size for normalization: using raw pixel coordinates")
            return {"x": position.x(), "y": position.y()}
            
        # Normalize coordinates to 0-1 range by dividing by image dimensions
        return {
            "x": position.x() / width,
            "y": position.y() / height
        }
        
    def _get_pixel_point(self, normalized_point: Dict[str, float], image_size: Tuple[int, int]) -> QPointF:
        """
        Convert normalized position (0-1 range) to pixel position.
        
        Args:
            normalized_point (dict): Normalized point coordinates as {"x": x, "y": y}
            image_size (tuple): Image size as (width, height)
            
        Returns:
            QPointF: Point position in pixels
        """
        width, height = image_size
        
        # Check if coordinates are already in the 0-1 range
        is_normalized = True
        if "x" in normalized_point and "y" in normalized_point:
            # Consider as pixel coordinates if both x and y are greater than 1.0
            if normalized_point["x"] > 1.0 and normalized_point["y"] > 1.0:
                is_normalized = False
        
        if width <= 0 or height <= 0 or not is_normalized:
            # Return the original coordinates if image size is invalid or coordinates are already in pixel form
            if width <= 0 or height <= 0:
                logger.warning("Invalid image size for denormalization: using raw coordinates")
            return QPointF(normalized_point["x"], normalized_point["y"])
            
        # Convert normalized coordinates (0-1) to pixel coordinates based on image size
        return QPointF(
            normalized_point["x"] * width,
            normalized_point["y"] * height
        )
    
    def _list_to_dict(self, points, image_size):
        """
        Convert list of points to dictionary format for storage.
        
        Args:
            points (list): List of point data dictionaries
            image_size (tuple): Image size as (width, height)
            
        Returns:
            dict: Dictionary of normalized points with keys like "p00", "p01", etc.
        """
        result = {}
        for idx, point in enumerate(points):
            try:
                normalized = self._get_normalized_point(point['position'], image_size)
                result[f"p{idx:02d}"] = {
                    "x": normalized["x"],
                    "y": normalized["y"],
                    "is_fine_tuned": point.get('is_fine_tuned', False)
                }
            except Exception as e:
                logger.warning(f"Error converting point to dictionary: {e}")
                # Use raw pixel coordinates if conversion fails
                result[f"p{idx:02d}"] = {
                    "x": point['position'].x(),
                    "y": point['position'].y(),
                    "is_fine_tuned": point.get('is_fine_tuned', False)
                }
        return result
        
    def _dict_to_list(self, points_dict, image_size):
        """
        Convert dictionary of points to list format for usage.
        
        Args:
            points_dict (dict): Dictionary of normalized points
            image_size (tuple): Image size as (width, height)
            
        Returns:
            list: List of point data dictionaries
        """
        result = []
        # Ensure points are processed in order (p00, p01, p02, ...)
        for key in sorted(points_dict.keys()):
            try:
                point_data = points_dict[key]
                normalized_point = {"x": point_data["x"], "y": point_data["y"]}
                position = self._get_pixel_point(normalized_point, image_size)
                result.append({
                    'position': position,
                    'is_fine_tuned': point_data.get('is_fine_tuned', False)
                })
            except Exception as e:
                logger.warning(f"Error converting dictionary to point: {e}")
                # Skip invalid points
        return result

    def _save_to_config(self) -> bool:
        """
        Save calibration data to configuration.
        
        Returns:
            bool: Success or failure
        """
        if not self.config_manager:
            logger.warning("Cannot save calibration data: No config manager available")
            return False
        
        try:
            # Set default values for invalid image sizes
            left_size = self.left_image_size
            right_size = self.right_image_size
            
            if left_size[0] <= 0 or left_size[1] <= 0:
                left_size = (640, 480)  # Default size
                logger.info("Using default left image size (640x480) for normalization")
                
            if right_size[0] <= 0 or right_size[1] <= 0:
                right_size = (640, 480)  # Default size
                logger.info("Using default right image size (640x480) for normalization")
                
            # Debug points count before normalization
            logger.info(f"Saving {len(self.left_points)} left points and {len(self.right_points)} right points")
                
            # Convert points to dictionary format for storage
            left_dict = self._list_to_dict(self.left_points, left_size)
            right_dict = self._list_to_dict(self.right_points, right_size)
            
            # Debug points count after conversion
            logger.info(f"Converted points: {len(left_dict)} left and {len(right_dict)} right")
            
            calibration_data = {
                'left': left_dict,
                'right': right_dict,
                'left_image_size': {'width': self.left_image_size[0], 'height': self.left_image_size[1]},
                'right_image_size': {'width': self.right_image_size[0], 'height': self.right_image_size[1]},
                'left_image_path': self.left_image_path if self.left_image_path else None,
                'right_image_path': self.right_image_path if self.right_image_path else None,
                'calib_ver': 1.2  # Version update (using dictionary format)
            }
            
            # Save to config - force immediate save
            self.config_manager.set_calibration_points(calibration_data)
            logger.info("set_calibration_points called with calibration data")
            
            # Force an immediate save to ensure data is persisted
            self.config_manager.save_config(force=True)
            logger.info("Forced immediate save of configuration")
            
            return True
        except Exception as e:
            logger.error(f"Error saving calibration data to config: {str(e)}")
            # Print full exception details for debugging
            import traceback
            logger.error(f"Full exception details: {traceback.format_exc()}")
            return False
    
    def _load_from_config(self) -> bool:
        """
        Load calibration data from configuration.
        
        Returns:
            bool: Success or failure
        """
        if not self.config_manager:
            logger.warning("Cannot load calibration data: No config manager available")
            return False
        
        try:
            # Get calibration data from config
            calibration_data = self.config_manager.get_calibration_points()
            
            # Debug logging for troubleshooting
            logger.info(f"Loading calibration data, keys found: {list(calibration_data.keys()) if calibration_data else 'None'}")
            
            # Check if data is empty - Fixed operator precedence with explicit parentheses
            if (not calibration_data) or ((not calibration_data.get('left', {})) and (not calibration_data.get('right', {}))):
                logger.warning("No calibration data found in config")
                return False
                
            # Backup existing points before clearing
            old_left_points = self.left_points.copy()
            old_right_points = self.right_points.copy()
            self.left_points = []
            self.right_points = []
            
            # Load image sizes if available
            if 'left_image_size' in calibration_data:
                size_data = calibration_data['left_image_size']
                self.left_image_size = (size_data.get('width', 0), size_data.get('height', 0))
            
            if 'right_image_size' in calibration_data:
                size_data = calibration_data['right_image_size']
                self.right_image_size = (size_data.get('width', 0), size_data.get('height', 0))
            
            # Load image paths if available
            if 'left_image_path' in calibration_data:
                self.left_image_path = calibration_data['left_image_path']
            
            if 'right_image_path' in calibration_data:
                self.right_image_path = calibration_data['right_image_path']
            
            # Check calibration version
            calib_ver = calibration_data.get('calib_ver', 1.0)
            
            # Process left points based on version
            left_data = calibration_data.get('left', {})
            right_data = calibration_data.get('right', {})
            
            if calib_ver >= 1.2:
                # New dictionary format (version 1.2+)
                logger.info(f"Loading calibration data in dictionary format (version {calib_ver})")
                
                # Convert dictionary points to list format
                self.left_points = self._dict_to_list(left_data, self.left_image_size)
                self.right_points = self._dict_to_list(right_data, self.right_image_size)
            else:
                # Old list format (version 1.0 or 1.1)
                logger.info(f"Loading calibration data in list format (version {calib_ver})")
                
                # Process left points
                if isinstance(left_data, list):
                    for point_data in left_data:
                        try:
                            if calib_ver >= 1.1:
                                # Version 1.1 - normalized coordinates
                                pixel_point = self._get_pixel_point(point_data, self.left_image_size)
                                self.left_points.append({
                                    'position': pixel_point,
                                    'is_fine_tuned': False
                                })
                            else:
                                # Version 1.0 - pixel coordinates as [x, y] list
                                if isinstance(point_data, list) and len(point_data) == 2:
                                    self.left_points.append({
                                        'position': QPointF(point_data[0], point_data[1]),
                                        'is_fine_tuned': False
                                    })
                        except Exception as e:
                            logger.warning(f"Error loading left point: {e}")
                
                # Process right points
                if isinstance(right_data, list):
                    for point_data in right_data:
                        try:
                            if calib_ver >= 1.1:
                                # Version 1.1 - normalized coordinates
                                pixel_point = self._get_pixel_point(point_data, self.right_image_size)
                                self.right_points.append({
                                    'position': pixel_point,
                                    'is_fine_tuned': False
                                })
                            else:
                                # Version 1.0 - pixel coordinates as [x, y] list
                                if isinstance(point_data, list) and len(point_data) == 2:
                                    self.right_points.append({
                                        'position': QPointF(point_data[0], point_data[1]),
                                        'is_fine_tuned': False
                                    })
                        except Exception as e:
                            logger.warning(f"Error loading right point: {e}")
            
            # Debug points count after loading
            logger.info(f"Loaded {len(self.left_points)} left points and {len(self.right_points)} right points")
            
            # Emit signals for each side
            self.points_changed.emit("left")
            self.points_changed.emit("right")
            
            return True
        except Exception as e:
            logger.error(f"Error loading calibration data from config: {str(e)}")
            # Restore backup points if loading fails
            self.left_points = old_left_points
            self.right_points = old_right_points
            # Print full exception details for debugging
            import traceback
            logger.error(f"Full exception details: {traceback.format_exc()}")
            return False
    
    def load_from_config(self) -> bool:
        """
        Public method to load from config. 
        Useful if the config manager was set after initialization.
        
        Returns:
            bool: Success or failure
        """
        return self._load_from_config()
    
    def save_to_config(self) -> bool:
        """
        Public method to save to config.
        
        Returns:
            bool: Success or failure
        """
        return self._save_to_config()
    
    def save_to_json(self, file_path: str) -> bool:
        """
        Save calibration data to a separate JSON file.
        This method is kept for compatibility with existing code.
        
        Args:
            file_path (str): File path
            
        Returns:
            bool: Success or failure
        """
        try:
            # Set default values for invalid image sizes
            left_size = self.left_image_size
            right_size = self.right_image_size
            
            if left_size[0] <= 0 or left_size[1] <= 0:
                left_size = (640, 480)  # Default size
                logger.info("Using default left image size (640x480) for normalization")
                
            if right_size[0] <= 0 or right_size[1] <= 0:
                right_size = (640, 480)  # Default size
                logger.info("Using default right image size (640x480) for normalization")
            
            # Convert points to dictionary format
            left_dict = self._list_to_dict(self.left_points, left_size)
            right_dict = self._list_to_dict(self.right_points, right_size)
            
            # Create data dictionary
            data = {
                'calib_ver': 1.2,  # Using dictionary format
                'left': left_dict,
                'right': right_dict,
                'left_image_size': {'width': self.left_image_size[0], 'height': self.left_image_size[1]},
                'right_image_size': {'width': self.right_image_size[0], 'height': self.right_image_size[1]},
                'left_image_path': self.left_image_path if self.left_image_path else None,
                'right_image_path': self.right_image_path if self.right_image_path else None
            }
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
            
            logger.info(f"Calibration data saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving calibration data to file: {str(e)}")
            import traceback
            logger.error(f"Full exception details: {traceback.format_exc()}")
            return False
    
    def load_from_json(self, file_path: str) -> bool:
        """
        Load calibration data from a separate JSON file.
        This method is kept for compatibility with existing code.
        
        Args:
            file_path (str): File path
            
        Returns:
            bool: Success or failure
        """
        try:
            # Backup existing points
            old_left_points = self.left_points.copy()
            old_right_points = self.right_points.copy()
            
            # Clear existing points
            self.left_points = []
            self.right_points = []
            
            # Load data from file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Set image sizes
            if 'left_image_size' in data:
                size_data = data['left_image_size']
                self.left_image_size = (size_data.get('width', 0), size_data.get('height', 0))
            else:
                self.left_image_size = (640, 480)  # Default size
            
            if 'right_image_size' in data:
                size_data = data['right_image_size']
                self.right_image_size = (size_data.get('width', 0), size_data.get('height', 0))
            else:
                self.right_image_size = (640, 480)  # Default size
            
            # Set image paths
            self.left_image_path = data.get('left_image_path', data.get('left_image', None))
            self.right_image_path = data.get('right_image_path', data.get('right_image', None))
            
            # Check calibration version
            calib_ver = data.get('calib_ver', 1.0)
            logger.info(f"Loading calibration data from file, version {calib_ver}")
            
            # Process based on version
            if calib_ver >= 1.2:
                # New dictionary format (version 1.2+)
                left_data = data.get('left', {})
                right_data = data.get('right', {})
                
                # Convert to point list
                self.left_points = self._dict_to_list(left_data, self.left_image_size)
                self.right_points = self._dict_to_list(right_data, self.right_image_size)
            elif calib_ver >= 1.1:
                # Normalized list format (version 1.1)
                left_data = data.get('left', [])
                right_data = data.get('right', [])
                
                # Process left points
                for point_data in left_data:
                    try:
                        pixel_point = self._get_pixel_point(point_data, self.left_image_size)
                        self.left_points.append({
                            'position': pixel_point,
                            'is_fine_tuned': point_data.get('is_fine_tuned', False)
                        })
                    except Exception as e:
                        logger.warning(f"Error loading left point: {e}")
                
                # Process right points
                for point_data in right_data:
                    try:
                        pixel_point = self._get_pixel_point(point_data, self.right_image_size)
                        self.right_points.append({
                            'position': pixel_point,
                            'is_fine_tuned': point_data.get('is_fine_tuned', False)
                        })
                    except Exception as e:
                        logger.warning(f"Error loading right point: {e}")
            else:
                # Old v1.0 format or 'points' structure
                if 'points' in data:
                    points_data = data['points']
                    
                    # Load left points
                    for point_data in points_data.get('left', []):
                        if isinstance(point_data, dict) and 'x' in point_data and 'y' in point_data:
                            self.left_points.append({
                                'position': QPointF(point_data['x'], point_data['y']),
                                'is_fine_tuned': point_data.get('is_fine_tuned', False)
                            })
                    
                    # Load right points
                    for point_data in points_data.get('right', []):
                        if isinstance(point_data, dict) and 'x' in point_data and 'y' in point_data:
                            self.right_points.append({
                                'position': QPointF(point_data['x'], point_data['y']),
                                'is_fine_tuned': point_data.get('is_fine_tuned', False)
                            })
                elif 'raw_points' in data:
                    # Try to migrate old format
                    if not self._migrate_old_format(data):
                        # Restore backup on failure
                        self.left_points = old_left_points
                        self.right_points = old_right_points
                        return False
            
            # Debug points count
            logger.info(f"Loaded {len(self.left_points)} left points and {len(self.right_points)} right points from file")
            
            # Emit signals
            self.points_changed.emit("left")
            self.points_changed.emit("right")
            
            # Save to config if available
            if self.config_manager:
                self._save_to_config()
            
            return True
        except Exception as e:
            logger.error(f"Error loading calibration data from file: {str(e)}")
            # Print full exception details for debugging
            import traceback
            logger.error(f"Full exception details: {traceback.format_exc()}")
            return False
    
    def _migrate_old_format(self, data: Dict) -> bool:
        """
        Migrate data from old format.
        
        Args:
            data (Dict): Old format data
            
        Returns:
            bool: Success or failure
        """
        try:
            # Clear existing points
            self.clear_points()
            
            # Set image paths if available
            self.left_image_path = data.get('left_image')
            self.right_image_path = data.get('right_image')
            
            # Load points from raw_points
            if 'raw_points' in data:
                raw_points = data['raw_points']
                
                # Handle different possible formats
                if isinstance(raw_points, dict):
                    # Format: {'left': [...], 'right': [...]}
                    if 'left' in raw_points and 'right' in raw_points:
                        for point in raw_points['left']:
                            if isinstance(point, list) and len(point) >= 2:
                                self.left_points.append({
                                    'position': QPointF(point[0], point[1]),
                                    'is_fine_tuned': False
                                })
                            elif isinstance(point, dict) and 'x' in point and 'y' in point:
                                self.left_points.append({
                                    'position': QPointF(point['x'], point['y']),
                                    'is_fine_tuned': False
                                })
                        
                        for point in raw_points['right']:
                            if isinstance(point, list) and len(point) >= 2:
                                self.right_points.append({
                                    'position': QPointF(point[0], point[1]),
                                    'is_fine_tuned': False
                                })
                            elif isinstance(point, dict) and 'x' in point and 'y' in point:
                                self.right_points.append({
                                    'position': QPointF(point['x'], point['y']),
                                    'is_fine_tuned': False
                                })
                
                # Emit signals
                self.points_changed.emit("left")
                self.points_changed.emit("right")
                
                # Save to config if available
                if self.config_manager:
                    self._save_to_config()
                
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error migrating calibration data: {str(e)}")
            return False
    
    def set_config_manager(self, config_manager):
        """
        Set the config manager.
        
        Args:
            config_manager: ConfigManager instance
        """
        self.config_manager = config_manager
        # Load data from config if available
        if self.config_manager:
            self._load_from_config() 