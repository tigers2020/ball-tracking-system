#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Skeletonizer service.
This module contains functions for converting images to their skeletal representation.
"""

import logging
import numpy as np
import cv2
from typing import Optional, Tuple
from src.utils.error_handling import error_handler, ErrorAction

logger = logging.getLogger(__name__)

def skeletonize_image(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to its skeletal representation using thinning algorithm.
    
    Args:
        image (np.ndarray): Input image (grayscale)
        
    Returns:
        np.ndarray: Skeletonized binary image
    """
    with error_handler(
        message="Skeletonization failed: {error}",
        action=ErrorAction.RETURN_DEFAULT,
        default_return=np.zeros((1, 1), dtype=np.uint8) if image is None else image
    ) as handler:
        # Ensure the image is valid
        if image is None or image.size == 0:
            logger.error("Invalid image for skeletonization: None or empty")
            return np.zeros((1, 1), dtype=np.uint8)
            
        # Ensure the image is binary (0 or 255)
        if image.dtype != np.uint8:
            logger.warning("Converting image to uint8 for skeletonization")
            image = image.astype(np.uint8)
            
        # If multichannel, convert to grayscale
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # Ensure the image is properly thresholded for thinning
        _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
        
        # Try OpenCV's thinning algorithm (Zhang-Suen)
        try:
            # Use OpenCV's built-in thinning
            try:
                skeleton = cv2.ximgproc.thinning(binary)
                logger.debug("Used OpenCV ximgproc.thinning for skeletonization")
                return skeleton
            except (cv2.error, AttributeError):
                # Fallback to implementation from ximgproc_contrib
                skeleton = cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
                logger.debug("Used OpenCV ximgproc.thinning with THINNING_ZHANGSUEN")
                return skeleton
                
        except (cv2.error, AttributeError) as e:
            logger.warning(f"OpenCV thinning failed: {e}. Falling back to scikit-image.")
            
            # Fallback to scikit-image if OpenCV fails
            try:
                from skimage.morphology import skeletonize
                binary_normalized = binary > 0  # Convert to boolean for skimage
                skeleton = skeletonize(binary_normalized).astype(np.uint8) * 255
                logger.debug("Used scikit-image for skeletonization")
                return skeleton
            except ImportError as e:
                logger.error(f"scikit-image import failed: {e}")
                
                # Fallback to custom implementation if all else fails
                skeleton = _custom_skeletonize(binary)
                logger.debug("Used custom skeletonization implementation")
                return skeleton
    
    return handler.result
        
def skeletonize_roi(image: np.ndarray, threshold: int = 128) -> np.ndarray:
    """
    Preprocess a Region of Interest (ROI) and skeletonize it.
    
    Args:
        image (np.ndarray): Input ROI image (grayscale)
        threshold (int): Threshold value for binarization (default: 128)
        
    Returns:
        np.ndarray: Skeletonized binary image
    """
    with error_handler(
        message="ROI skeletonization failed: {error}",
        action=ErrorAction.RETURN_DEFAULT,
        default_return=np.zeros((1, 1), dtype=np.uint8) if image is None else image
    ) as handler:
        # Ensure image is valid
        if image is None or image.size == 0:
            logger.error("Invalid ROI for skeletonization: None or empty")
            return np.zeros((1, 1), dtype=np.uint8)
            
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
        
        # Use adaptive thresholding for better results with varying lighting
        binary = cv2.adaptiveThreshold(
            blurred, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 2
        )
        
        # Apply morphological operations to clean up the binary image
        kernel = np.ones((3, 3), np.uint8)
        
        # Opening to remove small noise
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Closing to fill small holes
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Skeletonize the cleaned image
        skeleton = skeletonize_image(cleaned)
        
        # Remove isolated pixels (noise) from skeleton
        filtered_skeleton = _filter_skeleton(skeleton)
        
        return filtered_skeleton
    
    return handler.result
        
def _custom_skeletonize(binary_image: np.ndarray) -> np.ndarray:
    """
    Custom implementation of skeletonization using distance transform.
    Used as a last resort if other methods fail.
    
    Args:
        binary_image (np.ndarray): Binary image
        
    Returns:
        np.ndarray: Skeletonized binary image
    """
    with error_handler(
        message="Custom skeletonization failed: {error}",
        action=ErrorAction.RETURN_DEFAULT,
        default_return=binary_image
    ) as handler:
        # Ensure binary image
        _, binary = cv2.threshold(binary_image, 128, 255, cv2.THRESH_BINARY)
        
        # Distance transform
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        
        # Normalize the distance transform
        cv2.normalize(dist, dist, 0, 255, cv2.NORM_MINMAX)
        dist = dist.astype(np.uint8)
        
        # Threshold to find ridges
        _, dist_binary = cv2.threshold(dist, 50, 255, cv2.THRESH_BINARY)
        
        # Use a morphological operation to thin the ridges
        kernel = np.ones((3, 3), np.uint8)
        skeleton = cv2.morphologyEx(dist_binary, cv2.MORPH_OPEN, kernel)
        
        return skeleton
    
    return handler.result
        
def _filter_skeleton(skeleton: np.ndarray) -> np.ndarray:
    """
    Filter the skeleton to remove isolated pixels and short branches.
    
    Args:
        skeleton (np.ndarray): Skeletonized image
        
    Returns:
        np.ndarray: Filtered skeleton
    """
    with error_handler(
        message="Skeleton filtering failed: {error}",
        action=ErrorAction.RETURN_DEFAULT,
        default_return=skeleton
    ) as handler:
        # Create a copy to avoid modifying the original
        filtered = skeleton.copy()
        
        # Find all non-zero pixels
        y_coords, x_coords = np.where(filtered > 0)
        
        if len(y_coords) == 0:
            return filtered
            
        # Remove isolated pixels (pixels with no neighbors)
        kernel = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ], dtype=np.uint8)
        
        # Count neighbors for each non-zero pixel
        neighbor_count = cv2.filter2D(filtered // 255, -1, kernel)
        
        # Zero out pixels with no neighbors
        filtered[neighbor_count < 1] = 0
        
        return filtered
    
    return handler.result
        
def enhance_intersections(skeleton: np.ndarray) -> np.ndarray:
    """
    Enhance intersection points in the skeleton for better detection.
    
    Args:
        skeleton (np.ndarray): Skeletonized image
        
    Returns:
        np.ndarray: Skeleton with enhanced intersections
    """
    with error_handler(
        message="Intersection enhancement failed: {error}",
        action=ErrorAction.RETURN_DEFAULT,
        default_return=skeleton
    ) as handler:
        # Create a copy of the skeleton
        enhanced = skeleton.copy()
        
        # Create kernel for detecting intersections
        kernel = np.array([
            [1, 1, 1],
            [1, 10, 1],
            [1, 1, 1]
        ], dtype=np.uint8)
        
        # Convolve to find potential intersections
        convolved = cv2.filter2D(enhanced // 255, -1, kernel)
        
        # Find intersection points (value > 12 means at least 3 neighbors)
        intersections = np.where(convolved >= 13)
        
        # Enhance intersections by creating a slightly larger dot
        for y, x in zip(intersections[0], intersections[1]):
            cv2.circle(enhanced, (x, y), 2, 255, -1)
            
        return enhanced
    
    return handler.result 