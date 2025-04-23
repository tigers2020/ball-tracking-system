#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kalman Filter Factory module.
This module provides a factory class for creating and configuring Kalman filters.
"""

import logging
import os
import yaml
import numpy as np
import cv2
from typing import Dict, Any, Tuple


class KalmanFilterFactory:
    """
    Factory class for creating and configuring Kalman filters.
    Reads configuration from YAML file and creates properly configured filters.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the Kalman filter factory.
        
        Args:
            config_path (str, optional): Path to the Kalman filter configuration file.
                If None, default configuration is used.
        """
        self.config = self._load_config(config_path)
        logging.info(f"Kalman filter factory initialized with config: {self.config}")
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path (str, optional): Path to the configuration file.
            
        Returns:
            Dict containing the configuration values.
        """
        default_config = {
            'dt': 1.0,
            'process_noise': 0.03,
            'measurement_noise': 1.0,
            'initial_error_cov': 1.0,
            'init_velocity_x': 0.0,
            'init_velocity_y': 0.0
        }
        
        if not config_path or not os.path.exists(config_path):
            logging.warning(f"Configuration file {config_path} not found. Using default configuration.")
            return default_config
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Ensure all required keys are present
            for key in default_config.keys():
                if key not in config:
                    config[key] = default_config[key]
                    logging.warning(f"Missing configuration value for '{key}'. Using default: {default_config[key]}")
            
            return config
        except Exception as e:
            logging.error(f"Error loading Kalman filter configuration: {e}")
            return default_config
    
    def create_kalman_filter(self) -> Tuple[cv2.KalmanFilter, Dict[str, Any]]:
        """
        Create a new Kalman filter with configuration applied.
        
        Returns:
            Tuple containing:
            - The configured KalmanFilter object
            - Dictionary of configuration parameters used
        """
        try:
            # 4 state parameters (x, y, velocity_x, velocity_y), 2 measurement parameters (x, y)
            kalman = cv2.KalmanFilter(4, 2)
            
            # State transition matrix (how state evolves from t to t+1 without control or noise)
            # [ 1, 0, dt, 0  ]  x(t+1) = x(t) + vx(t)*dt
            # [ 0, 1, 0,  dt ]  y(t+1) = y(t) + vy(t)*dt
            # [ 0, 0, 1,  0  ]  vx(t+1) = vx(t)
            # [ 0, 0, 0,  1  ]  vy(t+1) = vy(t)
            dt = self.config['dt']
            transition_matrix = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], np.float32)
            
            kalman.transitionMatrix = transition_matrix
            
            # Measurement matrix (maps state vector to measurement vector)
            # [ 1, 0, 0, 0 ]  measure_x = x
            # [ 0, 1, 0, 0 ]  measure_y = y
            kalman.measurementMatrix = np.array([
                [1, 0, 0, 0], 
                [0, 1, 0, 0]
            ], np.float32)
            
            # Process noise covariance matrix (noise in the process model)
            # Larger values = faster adaptation to changes
            process_noise = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], np.float32) * self.config['process_noise']
            
            kalman.processNoiseCov = process_noise
            
            # Measurement noise covariance matrix (noise in measurements)
            # Larger values = measurements are treated as less reliable
            # Convert measurement_noise to float if it's a string
            measurement_noise_value = float(self.config['measurement_noise']) if isinstance(self.config['measurement_noise'], str) else self.config['measurement_noise']
            
            measurement_noise = np.array([
                [1, 0],
                [0, 1]
            ], np.float32) * measurement_noise_value
            
            kalman.measurementNoiseCov = measurement_noise
            
            # Error covariance matrix
            error_cov = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], np.float32) * self.config['initial_error_cov']
            
            kalman.errorCovPost = error_cov
            
            logging.info("Kalman filter created with dynamic configuration")
            
            return kalman, self.config
            
        except Exception as e:
            logging.error(f"Error creating Kalman filter: {e}")
            raise 