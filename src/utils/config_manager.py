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
            "last_image_folder": ""
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