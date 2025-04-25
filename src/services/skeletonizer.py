#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Skeletonizer service.
This module contains functions for converting images to their skeletal representation.
"""

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

def skeletonize_image(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to its skeletal representation using thinning algorithm.
    
    Args:
        image (np.ndarray): Input image (grayscale)
        
    Returns:
        np.ndarray: Skeletonized binary image
    """
    try:
        # Ensure the image is binary (0 or 255)
        if image.dtype != np.uint8:
            logger.warning("Converting image to uint8 for skeletonization")
            image = image.astype(np.uint8)
            
        # Ensure the image is properly thresholded for thinning
        _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
        
        # Try OpenCV's thinning algorithm
        try:
            skeleton = cv2.ximgproc.thinning(binary)
            logger.debug("Used OpenCV ximgproc.thinning for skeletonization")
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
                raise
    except Exception as e:
        logger.error(f"Skeletonization failed: {e}")
        # Return original image as fallback
        return image
        
def skeletonize_roi(image: np.ndarray, threshold: int = 128) -> np.ndarray:
    """
    Preprocess a Region of Interest (ROI) and skeletonize it.
    
    Args:
        image (np.ndarray): Input ROI image (grayscale)
        threshold (int): Threshold value for binarization (default: 128)
        
    Returns:
        np.ndarray: Skeletonized binary image
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply some preprocessing to improve results
        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding for better results with varying lighting
        binary = cv2.adaptiveThreshold(
            blurred, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 
            2
        )
        
        # Morphological operations to clean up the binary image
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Skeletonize the cleaned image
        skeleton = skeletonize_image(cleaned)
        
        return skeleton
    except Exception as e:
        logger.error(f"ROI skeletonization failed: {e}")
        # Return original image as fallback
        return image 