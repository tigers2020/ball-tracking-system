#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Geometry utility functions for coordinate transformations.
Contains functions for converting between pixel, normalized, and scene coordinates.
"""

import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def get_scale_factors(scene_width: float, scene_height: float, 
                      image_width: float, image_height: float) -> Tuple[float, float]:
    """
    Calculate scale factors between scene and image dimensions.
    
    Args:
        scene_width (float): Width of the scene
        scene_height (float): Height of the scene
        image_width (float): Width of the image
        image_height (float): Height of the image
        
    Returns:
        Tuple[float, float]: (width_scale, height_scale) - scale factors for width and height
    """
    # Ensure non-zero dimensions to prevent division by zero
    if image_width <= 0 or image_height <= 0 or scene_width <= 0 or scene_height <= 0:
        logger.warning(f"Invalid dimensions for scale calculation: "
                      f"scene({scene_width}x{scene_height}), image({image_width}x{image_height})")
        return 1.0, 1.0
    
    # Calculate the scale factors
    width_scale = scene_width / image_width
    height_scale = scene_height / image_height
    
    return width_scale, height_scale

def pixel_to_scene(pixel_coords: Tuple[float, float], 
                   width_scale: float, height_scale: float) -> Tuple[float, float]:
    """
    Convert pixel coordinates to scene coordinates.
    
    Args:
        pixel_coords (Tuple[float, float]): (x, y) in pixel coordinates
        width_scale (float): Width scale factor
        height_scale (float): Height scale factor
        
    Returns:
        Tuple[float, float]: (x, y) in scene coordinates
    """
    pixel_x, pixel_y = pixel_coords
    scene_x = pixel_x * width_scale
    scene_y = pixel_y * height_scale
    return scene_x, scene_y

def scene_to_pixel(scene_coords: Tuple[float, float], 
                   width_scale: float, height_scale: float) -> Tuple[float, float]:
    """
    Convert scene coordinates to pixel coordinates.
    
    Args:
        scene_coords (Tuple[float, float]): (x, y) in scene coordinates
        width_scale (float): Width scale factor
        height_scale (float): Height scale factor
        
    Returns:
        Tuple[float, float]: (x, y) in pixel coordinates
    """
    # Prevent division by zero
    if width_scale <= 0 or height_scale <= 0:
        logger.warning(f"Invalid scale factors for scene_to_pixel: ({width_scale}, {height_scale})")
        return scene_coords
    
    scene_x, scene_y = scene_coords
    pixel_x = scene_x / width_scale
    pixel_y = scene_y / height_scale
    return pixel_x, pixel_y

def pixel_to_normalized(pixel_coords: Tuple[float, float], 
                        image_width: float, image_height: float) -> Tuple[float, float]:
    """
    Convert pixel coordinates to normalized coordinates (0-1).
    
    Args:
        pixel_coords (Tuple[float, float]): (x, y) in pixel coordinates
        image_width (float): Width of the image
        image_height (float): Height of the image
        
    Returns:
        Tuple[float, float]: (x, y) in normalized coordinates (0-1)
    """
    # Ensure non-zero dimensions to prevent division by zero
    if image_width <= 0 or image_height <= 0:
        logger.warning(f"Invalid image dimensions for normalization: "
                      f"({image_width}x{image_height})")
        return 0.0, 0.0
    
    pixel_x, pixel_y = pixel_coords
    norm_x = pixel_x / image_width
    norm_y = pixel_y / image_height
    return norm_x, norm_y

def normalized_to_pixel(norm_coords: Tuple[float, float], 
                        image_width: float, image_height: float) -> Tuple[float, float]:
    """
    Convert normalized coordinates (0-1) to pixel coordinates.
    
    Args:
        norm_coords (Tuple[float, float]): (x, y) in normalized coordinates (0-1)
        image_width (float): Width of the image
        image_height (float): Height of the image
        
    Returns:
        Tuple[float, float]: (x, y) in pixel coordinates
    """
    norm_x, norm_y = norm_coords
    pixel_x = norm_x * image_width
    pixel_y = norm_y * image_height
    return pixel_x, pixel_y

def scene_to_normalized(scene_coords: Tuple[float, float], 
                        width_scale: float, height_scale: float,
                        image_width: float, image_height: float) -> Tuple[float, float]:
    """
    Convert scene coordinates to normalized coordinates (0-1).
    
    Args:
        scene_coords (Tuple[float, float]): (x, y) in scene coordinates
        width_scale (float): Width scale factor
        height_scale (float): Height scale factor
        image_width (float): Width of the image
        image_height (float): Height of the image
        
    Returns:
        Tuple[float, float]: (x, y) in normalized coordinates (0-1)
    """
    # First convert scene to pixel
    pixel_coords = scene_to_pixel(scene_coords, width_scale, height_scale)
    
    # Then convert pixel to normalized
    return pixel_to_normalized(pixel_coords, image_width, image_height)

def normalized_to_scene(norm_coords: Tuple[float, float], 
                       width_scale: float, height_scale: float,
                       image_width: float, image_height: float) -> Tuple[float, float]:
    """
    Convert normalized coordinates (0-1) to scene coordinates.
    
    Args:
        norm_coords (Tuple[float, float]): (x, y) in normalized coordinates (0-1)
        width_scale (float): Width scale factor
        height_scale (float): Height scale factor
        image_width (float): Width of the image
        image_height (float): Height of the image
        
    Returns:
        Tuple[float, float]: (x, y) in scene coordinates
    """
    # First convert normalized to pixel
    pixel_coords = normalized_to_pixel(norm_coords, image_width, image_height)
    
    # Then convert pixel to scene
    return pixel_to_scene(pixel_coords, width_scale, height_scale) 