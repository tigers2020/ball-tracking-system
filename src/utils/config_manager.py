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
            dict: HSV settings
        """
        return self.get("hsv_settings", self.default_config["hsv_settings"])
    
    def set_hsv_settings(self, hsv_settings):
        """
        Set the HSV settings for ball tracking.
        
        Args:
            hsv_settings (dict): HSV settings
        """
        current_settings = self.get_hsv_settings().copy()
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