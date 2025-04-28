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
                "sensor_width_mm": 36.0,
                "sensor_height_mm": 24.0,
                "principal_point_x": 320.0,
                "principal_point_y": 240.0
            },
            "calibration_points": {
                "left": {},
                "right": {},
                "left_image_size": {"width": 0, "height": 0},
                "right_image_size": {"width": 0, "height": 0},
                "left_image_path": None,
                "right_image_path": None,
                "calib_ver": 1.2
            },
            "coordinate_settings": {
                "rotation": {
                    "x": 0.0,  # Pitch in degrees
                    "y": 0.0,  # Roll in degrees
                    "z": 0.0   # Yaw in degrees
                },
                "scale": 0.01,  # Scale factor (meters per pixel)
                "camera_height": 3.0  # Camera height in meters
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
        
        # 이전 버전과의 호환성을 위한 필드 이름 업데이트
        self._update_legacy_field_names()
    
    def _update_legacy_field_names(self):
        """기존의 sensor_width, sensor_height 필드를 sensor_width_mm, sensor_height_mm로 업데이트"""
        camera_settings = self.get_camera_settings()
        
        # sensor_width가 있고 sensor_width_mm가 없는 경우
        if "sensor_width" in camera_settings and "sensor_width_mm" not in camera_settings:
            camera_settings["sensor_width_mm"] = camera_settings["sensor_width"]
            logging.info("Legacy field 'sensor_width' updated to 'sensor_width_mm'")
        
        # sensor_height가 있고 sensor_height_mm가 없는 경우
        if "sensor_height" in camera_settings and "sensor_height_mm" not in camera_settings:
            camera_settings["sensor_height_mm"] = camera_settings["sensor_height"]
            logging.info("Legacy field 'sensor_height' updated to 'sensor_height_mm'")
        
        # 업데이트된 설정 저장
        self.set_camera_settings(camera_settings)
    
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
    
    def get_section(self, section_name, default=None):
        """
        Get a configuration section.
        
        Args:
            section_name (str): Name of the configuration section
            default: Default value if the section doesn't exist
            
        Returns:
            Configuration section dict or default value
        """
        return self.config.get(section_name, default)
        
    def get_value(self, section, key, default=None):
        """
        Get a specific value from a configuration section.
        
        Args:
            section (str): Section name
            key (str): Key within the section
            default: Default value if the key or section doesn't exist
            
        Returns:
            Configuration value or default
        """
        section_data = self.config.get(section, {})
        return section_data.get(key, default)
    
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
        
        # 표준화된 키 매핑
        key_mapping = {
            "lower_h": "h_min", "upper_h": "h_max",
            "lower_s": "s_min", "upper_s": "s_max",
            "lower_v": "v_min", "upper_v": "v_max"
        }
        
        # 이전 키들을 새 형식으로 변환하고 이전 키는 제거
        for old_key, new_key in key_mapping.items():
            if old_key in settings:
                if new_key not in settings:
                    settings[new_key] = settings[old_key]
                # 이전 키는 항상 제거
                del settings[old_key]
        
        return settings
    
    def set_hsv_settings(self, hsv_settings):
        """
        Set the HSV settings for ball tracking.
        
        Args:
            hsv_settings (dict): HSV settings
        """
        current_settings = self.get_hsv_settings().copy()
        
        # 표준화된 키 매핑
        key_mapping = {
            "lower_h": "h_min", "upper_h": "h_max",
            "lower_s": "s_min", "upper_s": "s_max",
            "lower_v": "v_min", "upper_v": "v_max"
        }
        
        # 이전 키들을 새 형식으로 변환
        for old_key, new_key in key_mapping.items():
            if old_key in hsv_settings:
                hsv_settings[new_key] = hsv_settings[old_key]
                # 이전 키 제거
                if old_key in hsv_settings:
                    del hsv_settings[old_key]
            
            # 이전 형식의 키가 현재 설정에 있으면 제거
            if old_key in current_settings:
                del current_settings[old_key]
        
        # 업데이트된 설정 적용
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
        # Force immediate save to ensure settings are persisted
        self.save_config(force=True)
        logging.info("Camera settings updated and saved to configuration file")
        
    def get_calibration_points(self):
        """
        Get the calibration points settings.
        
        Returns:
            dict: Calibration points settings
        """
        return self.get("calibration_points", self.default_config["calibration_points"])
    
    def set_calibration_points(self, calibration_points):
        """
        Set the calibration points settings.
        
        Args:
            calibration_points (dict): Calibration points settings
        """
        current_settings = self.get_calibration_points().copy()
        current_settings.update(calibration_points)
        self.set("calibration_points", current_settings)
        # Don't save immediately, allow throttling
        self.save_config(force=False)
    
    def set_calibration_points_with_image_size(self, left_points=None, right_points=None, 
                                              left_image_size=None, right_image_size=None):
        """
        Set calibration points with image size information.
        
        Args:
            left_points (dict, optional): Dictionary of left calibration points (key: point_id, value: {x, y})
            right_points (dict, optional): Dictionary of right calibration points (key: point_id, value: {x, y})
            left_image_size (dict, optional): Dictionary with width and height of left image
            right_image_size (dict, optional): Dictionary with width and height of right image
        """
        calib_points = self.get_calibration_points()
        
        if left_points is not None:
            calib_points["left"] = left_points
        
        if right_points is not None:
            calib_points["right"] = right_points
            
        if left_image_size is not None:
            calib_points["left_image_size"] = left_image_size
            
        if right_image_size is not None:
            calib_points["right_image_size"] = right_image_size
            
        self.set_calibration_points(calib_points)
        # Force immediate save for calibration data to avoid potential loss
        self.save_config(force=True)

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

    def get_coordinate_settings(self):
        """
        Get the coordinate system settings.
        
        Returns:
            dict: Coordinate system settings
        """
        return self.get("coordinate_settings", self.default_config["coordinate_settings"]).copy()
    
    def set_coordinate_settings(self, coordinate_settings):
        """
        Set the coordinate system settings.
        
        Args:
            coordinate_settings (dict): Coordinate system settings
        """
        current_settings = self.get_coordinate_settings().copy()
        current_settings.update(coordinate_settings)
        self.set("coordinate_settings", current_settings)
        # Don't save immediately, allow throttling
        self.save_config(force=False)
    
    def get_coordinate_rotation(self):
        """
        Get the coordinate system rotation.
        
        Returns:
            dict: Rotation settings (x, y, z in degrees)
        """
        coordinate_settings = self.get_coordinate_settings()
        return coordinate_settings.get("rotation", self.default_config["coordinate_settings"]["rotation"]).copy()
    
    def set_coordinate_rotation(self, rotation):
        """
        Set the coordinate system rotation.
        
        Args:
            rotation (dict): Rotation settings (x, y, z in degrees)
        """
        coordinate_settings = self.get_coordinate_settings().copy()
        if "rotation" not in coordinate_settings:
            coordinate_settings["rotation"] = {}
        coordinate_settings["rotation"].update(rotation)
        self.set("coordinate_settings", coordinate_settings)
        # Don't save immediately, allow throttling
        self.save_config(force=False)
    
    def get_coordinate_scale(self):
        """
        Get the coordinate system scale.
        
        Returns:
            float: Scale factor (meters per pixel)
        """
        coordinate_settings = self.get_coordinate_settings()
        return coordinate_settings.get("scale", self.default_config["coordinate_settings"]["scale"])
    
    def set_coordinate_scale(self, scale):
        """
        Set the coordinate system scale.
        
        Args:
            scale (float): Scale factor (meters per pixel)
        """
        coordinate_settings = self.get_coordinate_settings().copy()
        coordinate_settings["scale"] = scale
        self.set("coordinate_settings", coordinate_settings)
        # Don't save immediately, allow throttling
        self.save_config(force=False)
    
    def get_camera_height(self):
        """
        Get the camera height setting in meters.
        
        Returns:
            float: Camera height in meters
        """
        return self.get_value("coordinate_settings", "camera_height", 
                             self.default_config["coordinate_settings"]["camera_height"])
    
    def set_camera_height(self, height):
        """
        Set the camera height setting in meters.
        
        Args:
            height (float): Camera height in meters
        """
        coord_settings = self.get_coordinate_settings().copy()
        coord_settings["camera_height"] = height
        self.set_coordinate_settings(coord_settings)
        
    def get_camera_baseline(self):
        """
        Get the camera baseline distance in meters.
        
        Returns:
            float: Camera baseline in meters
        """
        return self.get_value("camera_settings", "baseline_m", 
                             self.default_config["camera_settings"]["baseline_m"])
    
    def set_camera_baseline(self, baseline):
        """
        Set the camera baseline distance in meters.
        
        Args:
            baseline (float): Camera baseline in meters
        """
        camera_settings = self.get_camera_settings().copy()
        camera_settings["baseline_m"] = baseline
        self.set_camera_settings(camera_settings)
        
    def get_camera_focal_length(self):
        """
        Get the camera focal length setting in millimeters.
        
        Returns:
            float: Camera focal length in millimeters
        """
        return self.get_value("camera_settings", "focal_length_mm", 
                             self.default_config["camera_settings"]["focal_length_mm"])
    
    def set_camera_focal_length(self, focal_length):
        """
        Set the camera focal length setting in millimeters.
        
        Args:
            focal_length (float): Camera focal length in millimeters
        """
        camera_settings = self.get_camera_settings().copy()
        camera_settings["focal_length_mm"] = focal_length
        self.set_camera_settings(camera_settings)
        
    def get_camera_sensor_dimensions(self):
        """
        Get the camera sensor dimensions in millimeters.
        
        Returns:
            tuple: (width, height) in millimeters
        """
        camera_settings = self.get_camera_settings()
        width = camera_settings.get("sensor_width_mm", self.default_config["camera_settings"]["sensor_width_mm"])
        height = camera_settings.get("sensor_height_mm", self.default_config["camera_settings"]["sensor_height_mm"])
        return width, height
    
    def set_camera_sensor_dimensions(self, width, height):
        """
        Set the camera sensor dimensions in millimeters.
        
        Args:
            width (float): Sensor width in millimeters
            height (float): Sensor height in millimeters
        """
        camera_settings = self.get_camera_settings().copy()
        camera_settings["sensor_width_mm"] = width
        camera_settings["sensor_height_mm"] = height
        self.set_camera_settings(camera_settings)
        
    def set_triangulator(self, triangulator):
        """
        Set the triangulation service instance for shared use across controllers.
        
        Args:
            triangulator: Triangulation service instance
        """
        self._triangulator = triangulator
        logging.critical("Triangulator registered in ConfigManager for shared use")
        
    def get_triangulator(self):
        """
        Get the registered triangulation service instance.
        
        Returns:
            Triangulation service instance or None if not registered
        """
        if hasattr(self, '_triangulator'):
            return self._triangulator
        else:
            logging.critical("No triangulator registered in ConfigManager")
            return None 