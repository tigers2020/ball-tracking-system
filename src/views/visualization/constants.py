#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization Constants Module.
This module centralizes all constants related to visualization across the application.
"""

from dataclasses import dataclass
from typing import Tuple

# =============================================
# Color Constants
# =============================================

@dataclass
class Color:
    """Color constants in BGR format for OpenCV and RGB for Qt."""
    # Primary colors
    RED: Tuple[int, int, int] = (0, 0, 255)        # BGR format
    GREEN: Tuple[int, int, int] = (0, 255, 0)      # BGR format
    BLUE: Tuple[int, int, int] = (255, 0, 0)       # BGR format
    
    # Secondary colors
    YELLOW: Tuple[int, int, int] = (0, 255, 255)   # BGR format
    CYAN: Tuple[int, int, int] = (255, 255, 0)     # BGR format
    MAGENTA: Tuple[int, int, int] = (255, 0, 255)  # BGR format
    
    # Additional colors
    WHITE: Tuple[int, int, int] = (255, 255, 255)  # BGR format
    BLACK: Tuple[int, int, int] = (0, 0, 0)        # BGR format
    GRAY: Tuple[int, int, int] = (128, 128, 128)   # BGR format
    ORANGE: Tuple[int, int, int] = (0, 165, 255)   # BGR format
    PURPLE: Tuple[int, int, int] = (128, 0, 128)   # BGR format

    # Qt color equivalents (RGB format)
    QT_RED: Tuple[int, int, int] = (255, 0, 0)     # RGB format
    QT_GREEN: Tuple[int, int, int] = (0, 255, 0)   # RGB format 
    QT_BLUE: Tuple[int, int, int] = (0, 0, 255)    # RGB format
    QT_YELLOW: Tuple[int, int, int] = (255, 255, 0)  # RGB format
    QT_CYAN: Tuple[int, int, int] = (0, 255, 255)    # RGB format
    QT_MAGENTA: Tuple[int, int, int] = (255, 0, 255) # RGB format


# =============================================
# ROI Constants
# =============================================

@dataclass
class ROI:
    """Constants for Region of Interest visualization."""
    # Default thickness for ROI rectangle
    THICKNESS: int = 2
    
    # Size of center marker
    CENTER_MARKER_SIZE: int = 5
    
    # Fill opacity
    FILL_ALPHA: float = 0.2
    
    # Qt pen width equivalent
    QT_PEN_WIDTH: int = 2


# =============================================
# Tracking and Visualization Constants
# =============================================

@dataclass
class Tracking:
    """Constants for tracking visualization elements."""
    # Circle constants
    CIRCLE_THICKNESS: int = 2
    MAIN_CIRCLE_COLOR: Tuple[int, int, int] = Color.YELLOW
    
    # ROI constants
    ROI_THICKNESS: int = 2
    ROI_COLOR: Tuple[int, int, int] = Color.GREEN
    
    # Trajectory constants
    TRAJECTORY_COLOR: Tuple[int, int, int] = Color.BLUE
    TRAJECTORY_THICKNESS: int = 2
    TRAJECTORY_MAX_POINTS: int = 100
    
    # Prediction visualization
    PREDICTION_ARROW_COLOR: Tuple[int, int, int] = Color.MAGENTA
    PREDICTION_THICKNESS: int = 2
    UNCERTAINTY_RADIUS: int = 25
    
    # Point/marker constants
    POINT_COLOR: Tuple[int, int, int] = Color.RED
    POINT_RADIUS: int = 5
    POINT_THICKNESS: int = -1  # -1 means filled
    
    # Grid constants
    GRID_COLOR: Tuple[int, int, int] = Color.GREEN
    GRID_THICKNESS: int = 1
    GRID_DASHED: bool = True


# =============================================
# Layout Constants
# =============================================

@dataclass
class Layout:
    """Constants for layout visualization."""
    # Qt widget-specific constants
    QT_MARGIN: int = 10
    QT_PADDING: int = 5
    
    # Font settings
    FONT_SCALE: float = 0.7
    FONT_THICKNESS: int = 2
    
    # Label offset
    LABEL_OFFSET_X: int = 10
    LABEL_OFFSET_Y: int = 10


# Create global instances for easy import
COLOR = Color()
ROI = ROI()
TRACKING = Tracking()
LAYOUT = Layout() 