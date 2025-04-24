#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration Manager module.
This module contains the ConfigManager class for managing application configuration.
"""

import json
import os
import logging
from pathlib import Path


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
                "h_min": 0,   # 0-179 in OpenCV
                "h_max": 10,
                "s_min": 100, # 0-255 in OpenCV
                "s_max": 255,
                "v_min": 100, # 0-255 in OpenCV
                "v_max": 255
            },
            "roi_settings": {
                "width": 100,  # ROI width in pixels
                "height": 100, # ROI height in pixels
                "enabled": True, # Whether ROI is enabled
                "auto_center": True  # Whether to center ROI automatically on detected objects
            },
            "hough_circle_settings": {
                "dp": 1,               # Resolution ratio
                "min_dist": 50,        # Minimum distance between circles
                "param1": 100,         # Higher threshold for edge detection (Canny)
                "param2": 30,          # Threshold for center detection
                "min_radius": 10,      # Minimum radius
                "max_radius": 100,     # Maximum radius
                "adaptive": False       # Whether to adapt parameters based on ROI size
            },
            "kalman_settings": {
                "process_noise": 0.02,
                "measurement_noise": 0.1,
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
            }
        }
        
        # Current configuration
        self.config = self.default_config.copy()
        
        # Configuration file path
        self.config_file = Path(config_file)
        
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
    
    def save_config(self):
        """
        Save the current configuration to the configuration file.
        """
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=4)
            
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
        self.save_config()
    
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
        self.save_config()
    
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
        self.save_config()
    
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
        self.save_config()
    
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
        self.save_config()

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