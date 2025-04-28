#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
UI Constants for Tracking Overlay.
This module contains constants for the tracking overlay visualization.
"""

from dataclasses import dataclass


@dataclass
class TrackingColors:
    """Color constants for tracking visualization"""
    # Status colors (CSS format)
    SUCCESS = "#00b400"  # Green for successful tracking
    WARNING = "#f0a000"  # Yellow/Orange for warning states
    ERROR = "#f00000"    # Red for error states
    INFO = "#1f6feb"     # Blue for information/neutral states
    
    # Text colors
    TEXT_PRIMARY = "#ffffff"   # Primary text color
    TEXT_SECONDARY = "#c8c8c8" # Secondary text color
    
    # Background colors
    BACKGROUND = "#232323"        # Overlay background
    BACKGROUND_TRANSPARENT = "rgba(35, 35, 35, 200)"  # Semi-transparent background


@dataclass
class TrackingLayout:
    """Layout constants for tracking visualization"""
    # Overlay dimensions
    HEIGHT = 80           # Height of the overlay in pixels
    GROUP_SPACING = 10    # Spacing between groups in pixels
    MARGIN = 5            # Margin around the overlay in pixels
    
    # Label formatting
    LABEL_MIN_WIDTH = 60  # Minimum width for labels in pixels
    VALUE_MIN_WIDTH = 100 # Minimum width for value displays in pixels
    

@dataclass
class TrackingFormatting:
    """Formatting constants for tracking data display"""
    # Coordinate formatting
    COORDINATE_FORMAT = "{:.1f}"  # Format for 2D coordinates
    POSITION_FORMAT = "{:.2f} m"  # Format for 3D position with units
    TIME_FORMAT = "{:.1f} ms"     # Format for time values
    FPS_FORMAT = "{:.1f} FPS"     # Format for FPS values
    CONFIDENCE_FORMAT = "{:.1f}%" # Format for confidence values
    
    # Status display
    STATUS_PADDING = "2px 4px"    # Padding for status labels
    STATUS_BORDER_RADIUS = "2px"  # Border radius for status indicators 