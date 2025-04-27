#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Coordinate Service.
This module provides utilities for coordinate transformations between
different coordinate systems: image, camera, world, and court.
"""

import logging
import numpy as np
from typing import Tuple, List, Dict, Optional, Any, Union

from src.utils.constants import COURT


class CoordinateService:
    """
    Service for handling coordinate transformations between
    different coordinate systems.
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the coordinate service.
        
        Args:
            config_manager: Configuration manager for accessing settings
        """
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager
        
        # Default court to world transformation (identity)
        self.court_to_world_matrix = np.eye(4)
        self.world_to_court_matrix = np.eye(4)
        
        # Initialize from config if available
        if config_manager:
            self._init_from_config()
            
        self.logger.info("CoordinateService initialized")
        
    def _init_from_config(self):
        """Initialize coordinate transforms from configuration."""
        # Try to load court-to-world transformation matrix
        matrix = self.config_manager.get_value("coordinates", "court_to_world_matrix", None)
        if matrix is not None:
            try:
                self.court_to_world_matrix = np.array(matrix, dtype=np.float32).reshape(4, 4)
                self.world_to_court_matrix = np.linalg.inv(self.court_to_world_matrix)
                self.logger.info("Loaded court-to-world transformation from config")
            except Exception as e:
                self.logger.error(f"Error loading court-to-world matrix: {e}")
                # Fallback to identity matrix
                self.court_to_world_matrix = np.eye(4)
                self.world_to_court_matrix = np.eye(4)
        
    def world_to_court(self, world_point: np.ndarray) -> Tuple[float, float, float]:
        """
        Transform a 3D point from world coordinates to court coordinates.
        
        Args:
            world_point: Point in world coordinates (x, y, z)
            
        Returns:
            Point in court coordinates (x, y, z)
        """
        # Check input
        if world_point is None or world_point.size < 3:
            return (0, 0, 0)
            
        # Create homogeneous point
        world_point_homog = np.ones(4, dtype=np.float32)
        world_point_homog[:3] = world_point[:3]
        
        # Apply transformation
        court_point_homog = np.dot(self.world_to_court_matrix, world_point_homog)
        
        # Return 3D point
        return (
            float(court_point_homog[0]), 
            float(court_point_homog[1]),
            float(court_point_homog[2])
        )
        
    def court_to_world(self, court_point: Union[Tuple[float, float, float], np.ndarray]) -> np.ndarray:
        """
        Transform a 3D point from court coordinates to world coordinates.
        
        Args:
            court_point: Point in court coordinates (x, y, z)
            
        Returns:
            Point in world coordinates (x, y, z)
        """
        # Check input
        if court_point is None:
            return np.zeros(3, dtype=np.float32)
            
        # Create homogeneous point
        court_point_homog = np.ones(4, dtype=np.float32)
        if isinstance(court_point, np.ndarray):
            court_point_homog[:3] = court_point[:3]
        else:
            court_point_homog[:3] = court_point
        
        # Apply transformation
        world_point_homog = np.dot(self.court_to_world_matrix, court_point_homog)
        
        # Return 3D point
        return world_point_homog[:3]
        
    def set_court_to_world_matrix(self, matrix: np.ndarray) -> None:
        """
        Set the court-to-world transformation matrix.
        
        Args:
            matrix: 4x4 homogeneous transformation matrix
        """
        if matrix.shape != (4, 4):
            raise ValueError("Expected 4x4 transformation matrix")
            
        self.court_to_world_matrix = matrix.copy()
        try:
            self.world_to_court_matrix = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            self.logger.error("Singular matrix provided for court-to-world transform")
            # Fallback to identity
            self.court_to_world_matrix = np.eye(4)
            self.world_to_court_matrix = np.eye(4)
            
        # Save to config if available
        if self.config_manager:
            self.config_manager.set_value(
                "coordinates", "court_to_world_matrix", self.court_to_world_matrix.tolist())
            
    def validate_3d_position(self, position: np.ndarray, confidence: float = 1.0) -> Tuple[np.ndarray, float]:
        """
        Validate a 3D position and adjust confidence based on position validity.
        Apply court boundaries if configured to do so.
        
        Args:
            position: 3D position in world coordinates
            confidence: Initial confidence score (0-1)
            
        Returns:
            Tuple of (validated_position, adjusted_confidence)
        """
        if position is None or position.size < 3:
            return None, 0.0
            
        # Convert to court coordinates to check boundaries
        court_x, court_y, court_z = self.world_to_court(position)
        
        # Check if position is valid (within extended court boundaries)
        valid_x = -COURT.WIDTH_HALF - COURT.BOUNDARY_MARGIN <= court_x <= COURT.WIDTH_HALF + COURT.BOUNDARY_MARGIN
        valid_y = -COURT.BOUNDARY_MARGIN <= court_y <= COURT.LENGTH + COURT.BOUNDARY_MARGIN
        valid_z = COURT.MIN_VALID_HEIGHT <= court_z <= COURT.MAX_VALID_HEIGHT
        
        # Adjust confidence based on position validity
        adjusted_confidence = confidence
        if not (valid_x and valid_y and valid_z):
            self.logger.debug(f"Position outside court bounds: ({court_x:.2f}, {court_y:.2f}, {court_z:.2f})")
            # Reduce confidence for out-of-bounds position
            adjusted_confidence *= 0.5
            
        # Return original position with adjusted confidence
        # Note: we don't clamp the position, just adjust confidence
        return position, adjusted_confidence 