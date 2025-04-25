#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration Manager module.
This module contains the ConfigManager class for managing application configuration.
"""

import json
import os
import logging
import time
from pathlib import Path

from src.utils.constants import HSV, ROI, HOUGH, KALMAN


class ConfigManager:
    """
    Configuration manager for the Stereo Image Player application.
    Manages loading and saving configuration to a JSON file.
    """
    
    def __init__(self, config_file="config.json"):
        """
        Initialize the configuration manager.
        
        Args:
            config_file (str): Path to the configuration file
        """
        # Default configuration
        self.default_config = {
            "last_image_folder": "",
            "hsv_settings": {
                "h_min": HSV.h_min,   # 0-179 in OpenCV
                "h_max": HSV.h_max,
                "s_min": HSV.s_min,   # 0-255 in OpenCV
                "s_max": HSV.s_max,
                "v_min": HSV.v_min,   # 0-255 in OpenCV
                "v_max": HSV.v_max,
                "blur_size": HSV.blur_size,
                "morph_iterations": HSV.morph_iterations,
                "dilation_iterations": HSV.dilation_iterations
            },
            "roi_settings": {
                "width": ROI.DEFAULT_WIDTH,  # ROI width in pixels
                "height": ROI.DEFAULT_HEIGHT, # ROI height in pixels
                "enabled": ROI.ENABLED, # Whether ROI is enabled
                "auto_center": ROI.AUTO_CENTER  # Whether to center ROI automatically on detected objects
            },
            "hough_circle_settings": {
                "dp": HOUGH.dp,               # Resolution ratio
                "min_dist": HOUGH.min_dist,        # Minimum distance between circles
                "param1": HOUGH.param1,         # Higher threshold for edge detection (Canny)
                "param2": HOUGH.param2,          # Threshold for center detection
                "min_radius": HOUGH.min_radius,      # Minimum radius
                "max_radius": HOUGH.max_radius,     # Maximum radius
                "adaptive": HOUGH.adaptive       # Whether to adapt parameters based on ROI size
            },
            "kalman_settings": {
                "process_noise": KALMAN.process_noise,
                "measurement_noise": KALMAN.measurement_noise,
                "max_lost_frames": KALMAN.max_lost_frames,
                "dynamic_process_noise": KALMAN.dynamic_process_noise,
                "adaptive_measurement_noise": KALMAN.adaptive_measurement_noise,
                "dt": 0.1,
                "reset_threshold": 100.0,
                "velocity_decay": 0.98,
                "position_memory": 0.7
            },
            "camera_settings": {
                "camera_location_x": 0.0,
                "camera_location_y": 29.089,
                "camera_location_z": 12.503,
                "camera_rotation_x": 65.0,
                "camera_rotation_y": 0.0,
                "camera_rotation_z": 180.0,
                "focal_length_mm": 50.0,
                "baseline_m": 1.0,
                "sensor_width": 36.0,
                "sensor_height": 24.0,
                "principal_point_x": 320.0,
                "principal_point_y": 240.0
            },
            "calibration_points": {
                "left": [],
                "right": [],
                "calib_ver": 1.0
            }
        }
        
        # Current configuration
        self.config = self.default_config.copy()
        
        # Configuration file path
        self.config_file = Path(config_file)
        
        # Add throttling for save operations - increase interval to reduce I/O
        self._last_save_time = 0
        self._save_debounce_interval = 5.0  # Increase from 1.0 to 5.0 seconds (5x less frequent)
        self._pending_save = False
        self._change_count = 0  # Track number of changes since last save
        self._max_changes_before_save = 10  # Force save after this many changes
        
        # Load configuration from file if it exists
        self.load_config()
    
    def load_config(self):
        """
        Load configuration from the configuration file.
        If the file doesn't exist or is invalid, use default configuration.
        """
        try:
            if self.config_file.exists():
                with open(self.config_file, "r") as f:
                    loaded_config = json.load(f)
                
                # Update configuration with loaded values
                self.config.update(loaded_config)
                logging.info(f"Configuration loaded from {self.config_file}")
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
    
    def save_config(self, force=False):
        """
        Save the current configuration to the configuration file.
        Throttles saves to avoid excessive disk I/O.
        
        Args:
            force (bool): If True, ignore throttling and save immediately
        """
        current_time = time.time()
        # Mark that we have a pending save
        self._pending_save = True
        
        # Increment change counter
        self._change_count += 1
        
        # Check if we should throttle the save
        if not force and (current_time - self._last_save_time) < self._save_debounce_interval:
            # Only force save if we've accumulated many changes
            if self._change_count < self._max_changes_before_save:
                logging.debug(f"Throttling config save, will save soon (change {self._change_count}/{self._max_changes_before_save})")
                return
            else:
                logging.debug(f"Forcing save after {self._change_count} accumulated changes")
                force = True
            
        try:
            # Only save if we have pending changes
            if self._pending_save:
                with open(self.config_file, "w") as f:
                    json.dump(self.config, f, indent=4)
                
                self._last_save_time = current_time
                self._pending_save = False
                self._change_count = 0  # Reset change counter
                logging.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")
    
    def get(self, key, default=None):
        """
        Get a configuration value.
        
        Args:
            key (str): Configuration key
            default: Default value if the key doesn't exist
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key, value):
        """
        Set a configuration value.
        
        Args:
            key (str): Configuration key
            value: Configuration value
        """
        self.config[key] = value
    
    def set_last_image_folder(self, folder_path):
        """
        Set the last image folder path.
        
        Args:
            folder_path (str): Folder path
        """
        self.set("last_image_folder", folder_path)
        self.save_config()
    
    def get_last_image_folder(self):
        """
        Get the last image folder path.
        
        Returns:
            str: Last image folder path
        """
        return self.get("last_image_folder", "")
    
    def get_hsv_settings(self):
        """
        Get the HSV settings for ball tracking.
        
        Returns:
            dict: HSV settings with standardized keys (h_min, h_max, etc.)
        """
        settings = self.get("hsv_settings", self.default_config["hsv_settings"]).copy()
        
        # Standardized key mapping
        key_mapping = {
            "lower_h": "h_min", "upper_h": "h_max",
            "lower_s": "s_min", "upper_s": "s_max",
            "lower_v": "v_min", "upper_v": "v_max"
        }
        
        # Convert old keys to new format and remove old keys
        for old_key, new_key in key_mapping.items():
            if old_key in settings:
                if new_key not in settings:
                    settings[new_key] = settings[old_key]
                # Always remove old key
                del settings[old_key]
        
        return settings
    
    def set_hsv_settings(self, hsv_settings):
        """
        Set the HSV settings for ball tracking.
        
        Args:
            hsv_settings (dict): HSV settings
        """
        current_settings = self.get_hsv_settings().copy()
        
        # Standardized key mapping
        key_mapping = {
            "lower_h": "h_min", "upper_h": "h_max",
            "lower_s": "s_min", "upper_s": "s_max",
            "lower_v": "v_min", "upper_v": "v_max"
        }
        
        # Convert old keys to new format
        for old_key, new_key in key_mapping.items():
            if old_key in hsv_settings:
                hsv_settings[new_key] = hsv_settings[old_key]
                # Remove old key
                if old_key in hsv_settings:
                    del hsv_settings[old_key]
            
            # Remove old format keys from current settings if present
            if old_key in current_settings:
                del current_settings[old_key]
        
        # Apply updated settings
        current_settings.update(hsv_settings)
        self.set("hsv_settings", current_settings)
        # Don't save immediately, allow bundling of changes
        self.save_config(force=False)
    
    def get_roi_settings(self):
        """
        Get the ROI settings for ball tracking.
        
        Returns:
            dict: ROI settings
        """
        return self.get("roi_settings", self.default_config["roi_settings"])
    
    def set_roi_settings(self, roi_settings):
        """
        Set the ROI settings for ball tracking.
        
        Args:
            roi_settings (dict): ROI settings
        """
        current_settings = self.get_roi_settings().copy()
        current_settings.update(roi_settings)
        self.set("roi_settings", current_settings)
        # Don't save immediately, allow throttling
        self.save_config(force=False)
    
    def get_hough_circle_settings(self):
        """
        Get the Hough Circle detection settings.
        
        Returns:
            dict: Hough Circle settings
        """
        # Return a copy to avoid mutating original configuration
        return self.get("hough_circle_settings", self.default_config["hough_circle_settings"]).copy()
    
    def set_hough_circle_settings(self, hough_circle_settings):
        """
        Set the Hough Circle detection settings.
        
        Args:
            hough_circle_settings (dict): Hough Circle settings
        """
        current_settings = self.get_hough_circle_settings().copy()
        current_settings.update(hough_circle_settings)
        self.set("hough_circle_settings", current_settings)
        # Don't save immediately, allow throttling
        self.save_config(force=False)
    
    def get_kalman_settings(self):
        """
        Get the Kalman filter settings.
        
        Returns:
            dict: Kalman filter settings
        """
        return self.get("kalman_settings", self.default_config["kalman_settings"])
    
    def set_kalman_settings(self, kalman_settings):
        """
        Set the Kalman filter settings.
        
        Args:
            kalman_settings (dict): Kalman filter settings
        """
        current_settings = self.get_kalman_settings().copy()
        current_settings.update(kalman_settings)
        self.set("kalman_settings", current_settings)
        # Don't save immediately, allow throttling
        self.save_config(force=False)
    
    def get_camera_settings(self):
        """
        Get the camera settings.
        
        Returns:
            dict: Camera settings
        """
        return self.get("camera_settings", self.default_config["camera_settings"])
    
    def set_camera_settings(self, camera_settings):
        """
        Set the camera settings.
        
        Args:
            camera_settings (dict): Camera settings
        """
        current_settings = self.get_camera_settings().copy()
        current_settings.update(camera_settings)
        self.set("camera_settings", current_settings)
        # Don't save immediately, allow throttling
        self.save_config(force=False)

    def validate_roi(self, roi_settings=None, image_width=None, image_height=None):
        """
        Validate ROI settings against image dimensions.
        Adjusts ROI dimensions if they exceed image boundaries.
        
        Args:
            roi_settings (dict, optional): ROI settings to validate.
                If None, current settings will be used.
            image_width (int, optional): Image width in pixels.
                If None, no validation against width will be performed.
            image_height (int, optional): Image height in pixels.
                If None, no validation against height will be performed.
                
        Returns:
            dict: Validated (and possibly adjusted) ROI settings
        """
        # Use provided ROI settings or get current ones
        settings = roi_settings.copy() if roi_settings is not None else self.get_roi_settings().copy()
        
        # Validate width if image width is provided
        if image_width is not None and settings.get("width") is not None:
            if settings["width"] > image_width:
                old_width = settings["width"]
                settings["width"] = image_width
                logging.warning(f"ROI width ({old_width}) exceeds image width ({image_width}). Adjusted to {settings['width']}.")
        
        # Validate height if image height is provided
        if image_height is not None and settings.get("height") is not None:
            if settings["height"] > image_height:
                old_height = settings["height"]
                settings["height"] = image_height
                logging.warning(f"ROI height ({old_height}) exceeds image height ({image_height}). Adjusted to {settings['height']}.")
        
        # Ensure ROI dimensions are positive
        if settings.get("width") is not None and settings["width"] <= 0:
            settings["width"] = 100  # Default reasonable width
            logging.warning(f"Invalid ROI width <= 0. Set to default: {settings['width']}.")
            
        if settings.get("height") is not None and settings["height"] <= 0:
            settings["height"] = 100  # Default reasonable height
            logging.warning(f"Invalid ROI height <= 0. Set to default: {settings['height']}.")
        
        return settings
    
    def set_roi_settings_with_validation(self, roi_settings, image_width=None, image_height=None):
        """
        Validate and set the ROI settings for ball tracking.
        
        Args:
            roi_settings (dict): ROI settings
            image_width (int, optional): Image width for validation
            image_height (int, optional): Image height for validation
        
        Returns:
            dict: The validated and applied ROI settings
        """
        # Validate ROI settings
        validated_settings = self.validate_roi(roi_settings, image_width, image_height)
        
        # Apply validated settings
        self.set_roi_settings(validated_settings)
        
        return validated_settings
    
    def set_camera_settings_with_validation(self, camera_settings):
        """
        Validate and set the camera settings.
        
        Args:
            camera_settings (dict): Camera settings
        
        Returns:
            dict: The validated and applied camera settings
        """
        # Apply validated settings
        self.set_camera_settings(camera_settings)
        
        return camera_settings
    
    def get_calibration_points(self):
        """
        Get the calibration points.
        
        Returns:
            dict: Calibration points data with 'left' and 'right' lists of vector coordinates
        """
        # Try to get from 'calibration_points' first (new structure)
        calibration_data = self.get("calibration_points", None)
        logging.debug(f"Initial calibration_data: {calibration_data is not None}")
        
        # If not found, try 'court_calibration.points' (old structure in config.json)
        if not calibration_data:
            court_calibration = self.get("court_calibration", None)
            if court_calibration and "points" in court_calibration:
                calibration_data = court_calibration["points"]
                # Include version if available
                if "calib_ver" in court_calibration:
                    calibration_data["calib_ver"] = court_calibration["calib_ver"]
                logging.debug("Using calibration data from court_calibration.points")
        
        # If still not found, try 'points' (another possible old structure)
        if not calibration_data:
            calibration_data = self.get("points", None)
            if calibration_data:
                logging.debug("Using calibration data from points")
        
        # If no calibration data found in any format, return default empty structure
        if not calibration_data:
            calibration_data = {
                "left": [],
                "right": [],
                "calib_ver": 1.0
            }
            logging.debug("No calibration data found, using default empty structure")
        
        # Log the state of the data
        if calibration_data:
            logging.debug(f"Calibration data has left points: {len(calibration_data.get('left', []))}, right points: {len(calibration_data.get('right', []))}")
            
        # Ensure we have both left and right arrays
        if "left" not in calibration_data:
            calibration_data["left"] = []
        if "right" not in calibration_data:
            calibration_data["right"] = []
        if "calib_ver" not in calibration_data:
            calibration_data["calib_ver"] = 1.0
        
        # Migrate from old array format [x, y] to object format {"x": x, "y": y}
        for side in ["left", "right"]:
            if side in calibration_data:
                for i, point in enumerate(calibration_data[side]):
                    if isinstance(point, list) and len(point) >= 2:
                        # Convert [x, y] to {"x": x, "y": y}
                        calibration_data[side][i] = {"x": point[0], "y": point[1]}
                        logging.debug(f"Converted point format for {side}[{i}]")
        
        # Additional check to handle the case where points exist but are incorrectly formatted
        # This could happen if the config file is manually edited
        if "left" in calibration_data and calibration_data["left"] is None:
            calibration_data["left"] = []
            logging.warning("Left calibration points were None, initialized to empty list")
        if "right" in calibration_data and calibration_data["right"] is None:
            calibration_data["right"] = []
            logging.warning("Right calibration points were None, initialized to empty list")
            
        # Clone the data to avoid reference issues
        result = {
            "left": list(calibration_data.get("left", [])),
            "right": list(calibration_data.get("right", [])),
            "calib_ver": calibration_data.get("calib_ver", 1.0)
        }
        
        # Include image size and path information if available
        if "left_image_size" in calibration_data:
            result["left_image_size"] = calibration_data["left_image_size"]
        if "right_image_size" in calibration_data:
            result["right_image_size"] = calibration_data["right_image_size"]
        if "left_image_path" in calibration_data:
            result["left_image_path"] = calibration_data["left_image_path"]
        if "right_image_path" in calibration_data:
            result["right_image_path"] = calibration_data["right_image_path"]
            
        return result
    
    def set_calibration_points(self, calibration_data):
        """
        Set the calibration points.
        
        Args:
            calibration_data (dict): Calibration points data with 'left' and 'right' lists
                of vector coordinates and optional 'calib_ver'
        """
        import copy
        
        # Detailed debug logging
        logging.info(f"Setting calibration points with data keys: {list(calibration_data.keys() if calibration_data else [])}")
        
        # Ensure we have the minimum required structure
        if not isinstance(calibration_data, dict):
            logging.error("Invalid calibration data format. Must be a dictionary.")
            return False
            
        # Check if left and right points exist and have content
        left_points = calibration_data.get("left", [])
        right_points = calibration_data.get("right", [])
        
        if not left_points and not right_points:
            logging.warning("Both left and right calibration points are empty or missing.")
            # Don't overwrite existing data if new data is empty
            existing_data = self.get("calibration_points")
            if existing_data and (existing_data.get("left") or existing_data.get("right")):
                logging.warning("Keeping existing calibration data instead of overwriting with empty data.")
                return False
        
        # Ensure 'left' and 'right' keys are present and are lists
        if "left" not in calibration_data or not isinstance(calibration_data["left"], list):
            calibration_data["left"] = []
        if "right" not in calibration_data or not isinstance(calibration_data["right"], list):
            calibration_data["right"] = []
            
        # Add version if not present
        if "calib_ver" not in calibration_data:
            calibration_data["calib_ver"] = 1.0
            
        # Create a deep copy to avoid reference issues
        data_to_store = copy.deepcopy(calibration_data)
        
        # Log point counts for debugging
        logging.info(f"Storing calibration data with {len(data_to_store.get('left', []))} left points and {len(data_to_store.get('right', []))} right points")
        
        # Store in the new format
        self.set("calibration_points", data_to_store)
        
        # Save config immediately to ensure data persistence
        self.save_config(force=True)
        logging.info(f"Saved calibration data to config file with force=True")
        
        return True
    
    def get_left_calibration_points(self):
        """
        Get the left calibration points.
        
        Returns:
            list: List of vector coordinates for left camera
        """
        calibration_data = self.get_calibration_points()
        return calibration_data.get("left", [])
        
    def get_right_calibration_points(self):
        """
        Get the right calibration points.
        
        Returns:
            list: List of vector coordinates for right camera
        """
        calibration_data = self.get_calibration_points()
        return calibration_data.get("right", []) 