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

from src.utils.constants import ANALYSIS, COURT


class CoordinateService(QObject):
    """
    Service for handling coordinate transformations and validations.
    Provides a unified coordinate system for the entire application.
    """
    
    # Signal for notifying position updates in court coordinate system
    court_position_updated = Signal(float, float, float)  # x, y, z in court frame
    
    def __init__(self):
        """Initialize the coordinate service."""
        super(CoordinateService, self).__init__()
        logging.info("Coordinate service initialized")
    
    def world_to_court(self, position_3d: np.ndarray) -> Tuple[float, float, float]:
        """
        Transform world coordinates to court-centered coordinates.
        
        Args:
            position_3d: 3D position array [x, y, z] in world coordinates
            
        Returns:
            Tuple (x, y, z) in court coordinates
        """
        # For now, just ensure we're returning valid values that will be displayed properly
        # in the bounce overlay and other visualization components.
        # The coordinate system should match what is expected by the bounce overlay
        
        # The position_3d is already in the court-centered coordinate system
        # following the triangulation. We just need to make sure the values are valid
        # and have the correct sign conventions.
        
        # Assign x, y, z from position_3d (ensuring we have valid values)
        try:
            # Sometimes we might get NaN or Inf values
            if position_3d is None or not np.all(np.isfinite(position_3d)):
                # Return default valid values in this case
                logging.warning("Invalid position_3d values detected, using defaults")
                return 0.0, 0.0, 0.0
                
            x = float(position_3d[0])
            y = float(position_3d[1])
            z = float(position_3d[2])
            
            # Log the coordinate transformation
            logging.debug(f"Coordinate transformation: world ({x:.2f}, {y:.2f}, {z:.2f}) → court")
            
            # Apply any coordinate system conversions needed (none for now)
            # In the future, this could involve rotation, scaling, etc.
            
            # Return the court coordinates as a tuple
            return x, y, z
            
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
                logging.warning(f"Extremely high position detected: {adjusted_position[2]:.2f}m")
                
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