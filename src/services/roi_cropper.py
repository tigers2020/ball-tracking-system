#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ROI Cropper service.
This module contains functions for cropping regions of interest from images.
"""

import logging
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def crop_roi(image: np.ndarray, 
            center: Tuple[float, float], 
            radius: float,
            min_size: int = 40) -> Optional[np.ndarray]:
    """
    Crop a Region of Interest (ROI) from an image around a center point.
    
    Args:
        image (np.ndarray): Input image
        center (Tuple[float, float]): Center coordinates (x, y) of the ROI
        radius (float): Radius around the center to crop
        min_size (int, optional): Minimum ROI size in pixels (default: 40)
        
    Returns:
        Optional[np.ndarray]: Cropped region of interest or None if crop fails
    """
    try:
        # Check if image is valid
        if image is None or image.size == 0:
            logger.error("Invalid image: None or empty")
            return None
            
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Input validation
        if not isinstance(center, tuple) or len(center) != 2:
            logger.error(f"Invalid center point: {center}")
            return None
            
        if radius <= 0:
            logger.error(f"Invalid radius: {radius}")
            return None
            
        # Ensure center is within image bounds or clamp to boundary
        x = min(max(0, int(round(center[0]))), width - 1)
        y = min(max(0, int(round(center[1]))), height - 1)
        
        # Calculate ROI size (use max of radius*2.5 or min_size)
        roi_size = max(int(radius * 2.5), min_size)
        
        # Calculate ROI boundaries with improved boundary handling
        half_size = roi_size // 2
        left = max(0, x - half_size)
        top = max(0, y - half_size)
        right = min(width, x + half_size + (roi_size % 2))  # Add 1 for odd sizes
        bottom = min(height, y + half_size + (roi_size % 2))
        
        # Ensure ROI has consistent dimensions
        actual_width = right - left
        actual_height = bottom - top
        
        # If ROI is stretched near boundaries, adjust to maintain aspect ratio
        if abs(actual_width - actual_height) > 2:
            target_size = min(actual_width, actual_height)
            if actual_width > actual_height:
                # Adjust width
                excess = actual_width - target_size
                left += excess // 2
                right = left + target_size
            else:
                # Adjust height
                excess = actual_height - target_size
                top += excess // 2
                bottom = top + target_size
        
        # Verify all values are valid
        if left < 0 or top < 0 or right > width or bottom > height or left >= right or top >= bottom:
            logger.warning(f"Invalid ROI coordinates: ({left}, {top}, {right}, {bottom})")
            return None
        
        # Crop the ROI
        roi = image[top:bottom, left:right]
        
        # Check if ROI is valid (has at least some size)
        if roi.size == 0 or roi.shape[0] < 2 or roi.shape[1] < 2:
            logger.warning(f"Empty or too small ROI at ({x}, {y}) with radius {radius}")
            return None
            
        logger.debug(f"Cropped ROI at ({x}, {y}) with size {roi.shape[1]}x{roi.shape[0]}")
        return roi
        
    except Exception as e:
        logger.error(f"Error cropping ROI: {e}")
        return None
        
def crop_roi_with_padding(image: np.ndarray, 
                         center: Tuple[float, float], 
                         radius: float,
                         min_size: int = 40) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
    """
    Crop a Region of Interest (ROI) with padding if ROI extends beyond image boundaries.
    
    Args:
        image (np.ndarray): Input image
        center (Tuple[float, float]): Center coordinates (x, y) of the ROI
        radius (float): Radius around the center to crop
        min_size (int, optional): Minimum ROI size in pixels (default: 40)
        
    Returns:
        Tuple[Optional[np.ndarray], Tuple[int, int]]: 
            - Cropped ROI or None if crop fails
            - Offset of the top-left corner of the ROI relative to the original image (x, y)
    """
    try:
        # Check if image is valid
        if image is None or image.size == 0:
            logger.error("Invalid image: None or empty")
            return None, (0, 0)
            
        # Input validation
        if not isinstance(center, tuple) or len(center) != 2:
            logger.error(f"Invalid center point: {center}")
            return None, (0, 0)
            
        if radius <= 0:
            logger.error(f"Invalid radius: {radius}")
            return None, (0, 0)
            
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Convert center coordinates to integers
        x, y = int(round(center[0])), int(round(center[1]))
        
        # Calculate ROI size (use max of radius*2.5 or min_size)
        roi_size = max(int(radius * 2.5), min_size)
        
        # Make sure ROI size is even for better center alignment
        if roi_size % 2 != 0:
            roi_size += 1
            
        # Calculate ROI boundaries
        half_size = roi_size // 2
        left = x - half_size
        top = y - half_size
        right = x + half_size
        bottom = y + half_size
        
        # Calculate padding if ROI extends beyond image boundaries
        pad_left = abs(min(0, left))
        pad_top = abs(min(0, top))
        pad_right = max(0, right - width)
        pad_bottom = max(0, bottom - height)
        
        # Adjust ROI boundaries to be within image
        left = max(0, left)
        top = max(0, top)
        right = min(width, right)
        bottom = min(height, bottom)
        
        # Verify valid crop dimensions
        if left >= right or top >= bottom:
            logger.warning(f"Invalid crop dimensions: [{left}:{right}, {top}:{bottom}]")
            # Create a black ROI with the desired dimensions
            roi_shape = [roi_size, roi_size]
            if len(image.shape) > 2:
                roi_shape.append(image.shape[2])  # Add channel dimension
            return np.zeros(roi_shape, dtype=image.dtype), (left - pad_left, top - pad_top)
        
        # Crop the ROI
        roi = image[top:bottom, left:right].copy()  # Make a copy to avoid reference issues
        
        # Apply padding if needed
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            # Determine padding for each dimension
            pad_width = [(pad_top, pad_bottom), (pad_left, pad_right)]
            
            # Add channel dimension padding if needed
            if len(image.shape) > 2:
                pad_width.append((0, 0))
                
            # Pad the ROI with zeros
            roi = np.pad(roi, pad_width, mode='constant', constant_values=0)
            
            logger.debug(f"Applied padding to ROI: left={pad_left}, top={pad_top}, right={pad_right}, bottom={pad_bottom}")
        
        # Return the padded ROI and the offset of its top-left corner
        return roi, (left - pad_left, top - pad_top)
        
    except Exception as e:
        logger.error(f"Error cropping ROI with padding: {e}")
        return None, (0, 0)


def adjust_point_to_roi(point: Tuple[float, float], roi_offset: Tuple[int, int]) -> Tuple[float, float]:
    """
    Adjust a point's coordinates to be relative to a ROI.
    
    Args:
        point (Tuple[float, float]): The point coordinates in the original image (x, y)
        roi_offset (Tuple[int, int]): The ROI top-left offset (left, top)
            
    Returns:
        Tuple[float, float]: The adjusted point coordinates relative to the ROI
    """
    return (point[0] - roi_offset[0], point[1] - roi_offset[1])


def adjust_point_from_roi(point: Tuple[float, float], roi_offset: Tuple[int, int]) -> Tuple[float, float]:
    """
    Convert a point's coordinates from ROI-relative back to the original image.
    
    Args:
        point (Tuple[float, float]): The point coordinates in the ROI (x, y)
        roi_offset (Tuple[int, int]): The ROI top-left offset (left, top)
            
    Returns:
        Tuple[float, float]: The adjusted point coordinates in the original image
    """
    return (point[0] + roi_offset[0], point[1] + roi_offset[1]) 