#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Geometry utility functions.
This module contains utility functions for geometric calculations and coordinate transformations.
"""

def pixel_to_scene(x_px: float, y_px: float,
                  scale: float, offset_y: float = 0) -> tuple[float, float]:
    """
    Convert original pixel coordinates to QGraphicsScene coordinates.
    
    Args:
        x_px (float): X coordinate in original pixel space
        y_px (float): Y coordinate in original pixel space
        scale (float): Scale factor (displayed image = original * scale, e.g. 0.5)
        offset_y (float): Vertical offset in pixels if image is shifted down in scene
        
    Returns:
        tuple[float, float]: Transformed coordinates for scene display
    """
    return x_px * scale, y_px * scale + offset_y


def scene_to_pixel(x_scene: float, y_scene: float,
                  scale: float, offset_y: float = 0) -> tuple[float, float]:
    """
    Convert QGraphicsScene coordinates to original pixel coordinates.
    
    Args:
        x_scene (float): X coordinate in scene space
        y_scene (float): Y coordinate in scene space
        scale (float): Scale factor (displayed image = original * scale, e.g. 0.5)
        offset_y (float): Vertical offset in pixels if image is shifted down in scene
        
    Returns:
        tuple[float, float]: Transformed coordinates in original pixel space
    """
    return x_scene / scale, (y_scene - offset_y) / scale 