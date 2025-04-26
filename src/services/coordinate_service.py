#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Coordinate Service module.
This module provides unified coordinate transformation and validation services.
"""

import logging
import numpy as np
from typing import Tuple
from PySide6.QtCore import QObject, Signal

from src.utils.constants import ANALYSIS, COURT, STEREO


class CoordinateService(QObject):
    """
    Service for handling coordinate transformations and validations.
    Provides a unified coordinate system for the entire application.
    """
    
    # Signal for notifying position updates in court coordinate system
    court_position_updated = Signal(float, float, float)  # x, y, z in court frame
    
    def __init__(self, config=None):
        """
        Initialize the coordinate service.
        
        Args:
            config (dict, optional): Configuration dictionary with camera and coordinate settings
        """
        super(CoordinateService, self).__init__()
        
        # Default values
        self.pitch = 0.0  # rotation around X axis (in radians)
        self.roll = 0.0   # rotation around Y axis (in radians)
        self.yaw = 0.0    # rotation around Z axis (in radians)
        self.scale = STEREO.DEFAULT_SCALE  # scale factor (meters per pixel)
        self.camera_height = STEREO.DEFAULT_CAMERA_HEIGHT_M  # camera height in meters
        self.baseline = STEREO.DEFAULT_BASELINE_M  # stereo camera baseline in meters
        
        # Initialize transformation matrices
        self.R = np.eye(3, dtype=STEREO.DATA_TYPE)  # Rotation matrix (camera to world)
        self.T = np.array([0, 0, 0], dtype=STEREO.DATA_TYPE)  # Translation vector
        
        # Stereo calibration parameters
        self.R_stereo = np.eye(3, dtype=STEREO.DATA_TYPE)  # Rotation between left and right cameras
        self.T_stereo = np.array([self.baseline, 0, 0], dtype=STEREO.DATA_TYPE)  # Translation between cameras
        
        # Apply config if provided
        if config:
            self.update_config(config)
            
        logging.info("Coordinate service initialized")
    
    def update_config(self, config):
        """
        Update coordinate system parameters from config.
        
        Args:
            config (dict): Configuration dictionary
        """
        if "rotation" in config:
            rot = config["rotation"]
            if "x" in rot:
                self.pitch = np.deg2rad(rot["x"])
            if "y" in rot:
                self.roll = np.deg2rad(rot["y"])
            if "z" in rot:
                self.yaw = np.deg2rad(rot["z"])
        
        if "scale" in config:
            self.scale = config["scale"]
            
        if "camera_height" in config:
            self.camera_height = config["camera_height"]
        
        # Handle stereo calibration parameters if available
        if "stereo" in config:
            stereo_config = config["stereo"]
            
            if "baseline_m" in stereo_config:
                self.baseline = stereo_config["baseline_m"]
                
            if "camera_height_m" in stereo_config:
                self.camera_height = stereo_config["camera_height_m"]
                
            # Load stereo rotation matrix if available
            if "R_left_to_right" in stereo_config:
                self.R_stereo = np.array(stereo_config["R_left_to_right"], dtype=STEREO.DATA_TYPE)
                
            # Load stereo translation vector if available
            if "T_left_to_right" in stereo_config:
                self.T_stereo = np.array(stereo_config["T_left_to_right"], dtype=STEREO.DATA_TYPE)
            else:
                # Default translation is along X-axis (baseline)
                self.T_stereo = np.array([self.baseline, 0, 0], dtype=STEREO.DATA_TYPE)
            
        # Update rotation matrix
        self._update_rotation_matrix()
        
        logging.info(f"Coordinate service updated with pitch={np.rad2deg(self.pitch):.1f}°, "
                   f"roll={np.rad2deg(self.roll):.1f}°, yaw={np.rad2deg(self.yaw):.1f}°, "
                   f"scale={self.scale:.4f}, camera_height={self.camera_height:.2f}m")
        logging.info(f"Using stereo baseline: {self.baseline:.2f}m")
    
    def _update_rotation_matrix(self):
        """Calculate the 3x3 rotation matrix from Euler angles."""
        # Rotation matrices for each axis
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(self.pitch), -np.sin(self.pitch)],
            [0, np.sin(self.pitch), np.cos(self.pitch)]
        ], dtype=STEREO.DATA_TYPE)
        
        Ry = np.array([
            [np.cos(self.roll), 0, np.sin(self.roll)],
            [0, 1, 0],
            [-np.sin(self.roll), 0, np.cos(self.roll)]
        ], dtype=STEREO.DATA_TYPE)
        
        Rz = np.array([
            [np.cos(self.yaw), -np.sin(self.yaw), 0],
            [np.sin(self.yaw), np.cos(self.yaw), 0],
            [0, 0, 1]
        ], dtype=STEREO.DATA_TYPE)
        
        # Combined rotation matrix (order: Rz * Ry * Rx)
        self.R = Rz @ Ry @ Rx
    
    def world_to_court(self, position_3d: np.ndarray) -> Tuple[float, float, float]:
        """
        Transform world coordinates to court-centered coordinates.
        
        Args:
            position_3d: 3D position array [x, y, z] in world coordinates
            
        Returns:
            Tuple (x, y, z) in court coordinates
        """
        try:
            # Sometimes we might get NaN or Inf values
            if position_3d is None or not np.all(np.isfinite(position_3d)):
                # Return default valid values in this case
                logging.warning("Invalid position_3d values detected, using defaults")
                return 0.0, 0.0, 0.0
                
            # Convert to numpy array and ensure it's a vector
            p_cam = np.array(position_3d, dtype=STEREO.DATA_TYPE).flatten()
            
            # Scale from pixels to meters
            p_cam_m = p_cam * self.scale
            
            # Apply rotation to convert from camera to world coordinates
            p_world = self.R @ p_cam_m + self.T
            
            # Apply camera height offset
            p_world[2] -= self.camera_height
            
            # 축 맵핑 수정: 원래 좌표계 유지 (y/z 맵핑 오류 수정)
            # 기존: court_x = world_y, court_y = world_z, court_z = world_x (잘못된 매핑)
            # 수정: court_x = world_x, court_y = world_y, court_z = world_z (올바른 매핑)
            court_x = float(p_world[0])  # x축은 그대로
            court_y = float(p_world[1])  # y축은 그대로 (깊이)
            court_z = float(p_world[2])  # z축은 그대로 (높이)
            
            # Log the coordinate transformation
            logging.debug(f"Coordinate transformation: world ({position_3d[0]:.2f}, {position_3d[1]:.2f}, {position_3d[2]:.2f}) → "
                        f"court ({court_x:.2f}, {court_y:.2f}, {court_z:.2f})")
            
            # Return the court coordinates as a tuple
            return court_x, court_y, court_z
            
        except (TypeError, IndexError) as e:
            logging.error(f"Error in world_to_court transformation: {e}")
            return 0.0, 0.0, 0.0
    
    def validate_3d_position(self, position_3d: np.ndarray, confidence: float) -> Tuple[np.ndarray, float]:
        """
        Validate the 3D position and adjust confidence accordingly.
        
        Args:
            position_3d: 3D position array [x, y, z]
            confidence: Initial confidence score
            
        Returns:
            Tuple (adjusted_position, adjusted_confidence)
        """
        # Create a copy of the position
        adjusted_position = position_3d.copy()
        
        # Apply height constraints
        if adjusted_position[2] > ANALYSIS.MAX_VALID_HEIGHT:
            # If height is too high, reduce confidence
            height_factor = min((adjusted_position[2] - ANALYSIS.MAX_VALID_HEIGHT) / ANALYSIS.HEIGHT_CONFIDENCE_FACTOR, 1.0)
            height_confidence = max(ANALYSIS.MIN_HEIGHT_CONFIDENCE, 1.0 - height_factor)
            confidence = confidence * height_confidence
            
            if adjusted_position[2] > ANALYSIS.EXTREME_HEIGHT_THRESHOLD:
                logging.info(f"Extremely high position detected: {adjusted_position[2]:.2f}m")
                
            # Only log a warning and keep the original value
            logging.info(f"Height exceeds maximum valid value: {adjusted_position[2]:.2f}m")
            
        elif adjusted_position[2] < ANALYSIS.MIN_VALID_HEIGHT:
            # For negative height, clamp to minimum but don't severely reduce confidence
            logging.info(f"Negative height detected: {adjusted_position[2]:.2f}m, adjusting to 0")
            adjusted_position[2] = ANALYSIS.MIN_VALID_HEIGHT
            confidence = confidence * ANALYSIS.MIN_NEGATIVE_HEIGHT_CONFIDENCE
        
        # Check if position is within court boundaries
        COURT_WIDTH_HALF = COURT.WIDTH_HALF
        BOUNDARY_MARGIN = COURT.BOUNDARY_MARGIN
        
        if abs(adjusted_position[0]) > COURT_WIDTH_HALF + BOUNDARY_MARGIN:
            # If outside x boundary, reduce confidence
            x_factor = min((abs(adjusted_position[0]) - COURT_WIDTH_HALF - BOUNDARY_MARGIN) / BOUNDARY_MARGIN, 1.0)
            x_confidence = max(ANALYSIS.MIN_BOUNDARY_CONFIDENCE, 1.0 - x_factor)
            confidence = confidence * x_confidence
            
            # Log warning for extreme positions
            if abs(adjusted_position[0]) > COURT_WIDTH_HALF + BOUNDARY_MARGIN * ANALYSIS.EXTREME_BOUNDARY_FACTOR:
                logging.warning(f"X-axis boundary exceeded: {adjusted_position[0]:.2f}m")
        
        if abs(adjusted_position[1]) > COURT.LENGTH_HALF + BOUNDARY_MARGIN:
            # If outside y boundary, reduce confidence
            y_factor = min((abs(adjusted_position[1]) - COURT.LENGTH_HALF - BOUNDARY_MARGIN) / BOUNDARY_MARGIN, 1.0)
            y_confidence = max(ANALYSIS.MIN_BOUNDARY_CONFIDENCE, 1.0 - y_factor)
            confidence = confidence * y_confidence
            
            # Log warning for extreme positions
            if abs(adjusted_position[1]) > COURT.LENGTH_HALF + BOUNDARY_MARGIN * ANALYSIS.EXTREME_BOUNDARY_FACTOR:
                logging.warning(f"Y-axis boundary exceeded: {adjusted_position[1]:.2f}m")
        
        return adjusted_position, confidence 