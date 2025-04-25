#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ROI Cropper service.
This module contains functions for cropping regions of interest around calibration points.
"""

import logging
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)


def crop_roi(image: np.ndarray, center: Tuple[int, int], radius: float = 20.0) -> np.ndarray:
    """
    Crop a region of interest around a center point.
    
    Args:
        image (np.ndarray): The input image
        center (Tuple[int, int]): The center point (x, y)
        radius (float, optional): The initial radius around the center
            
    Returns:
        np.ndarray: The cropped ROI
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid input image")
    
    # Calculate ROI size based on the radius with a minimum size
    roi_size = max(40, int(radius * 2.5))
    
    # Make size even for better centering
    if roi_size % 2 != 0:
        roi_size += 1
    
    half_size = roi_size // 2
    
    # Get center coordinates (note the y, x order for numpy arrays)
    cx, cy = int(center[0]), int(center[1])
    
    # Calculate ROI boundaries
    left = max(0, cx - half_size)
    top = max(0, cy - half_size)
    right = min(image.shape[1], cx + half_size)
    bottom = min(image.shape[0], cy + half_size)
    
    # Check if ROI is too small
    if (right - left) < 10 or (bottom - top) < 10:
        raise ValueError(f"ROI is too small: {right-left}x{bottom-top}")
    
    # Crop the ROI
    roi = image[top:bottom, left:right].copy()
    
    logger.debug(f"Cropped ROI with dimensions: {roi.shape}")
    
    return roi


def crop_roi_with_offset(image: np.ndarray, center: Tuple[int, int], 
                         radius: float = 20.0) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Crop a region of interest around a center point and return the offset.
    
    Args:
        image (np.ndarray): The input image
        center (Tuple[int, int]): The center point (x, y)
        radius (float, optional): The initial radius around the center
            
    Returns:
        Tuple[np.ndarray, Tuple[int, int]]: The cropped ROI and the top-left offset (left, top)
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid input image")
    
    # Calculate ROI size based on the radius with a minimum size
    roi_size = max(40, int(radius * 2.5))
    
    # Make size even for better centering
    if roi_size % 2 != 0:
        roi_size += 1
    
    half_size = roi_size // 2
    
    # Get center coordinates (note the y, x order for numpy arrays)
    cx, cy = int(center[0]), int(center[1])
    
    # Calculate ROI boundaries
    left = max(0, cx - half_size)
    top = max(0, cy - half_size)
    right = min(image.shape[1], cx + half_size)
    bottom = min(image.shape[0], cy + half_size)
    
    # Check if ROI is too small
    if (right - left) < 10 or (bottom - top) < 10:
        raise ValueError(f"ROI is too small: {right-left}x{bottom-top}")
    
    # Crop the ROI
    roi = image[top:bottom, left:right].copy()
    
    logger.debug(f"Cropped ROI with dimensions: {roi.shape}, offset: ({left}, {top})")
    
    return roi, (left, top)


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