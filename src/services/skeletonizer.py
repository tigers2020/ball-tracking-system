#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Skeletonizer Service.
This service provides functionality for thinning and skeletonizing binary images.
"""

import logging
import numpy as np
import cv2

try:
    from skimage.morphology import skeletonize as sk_skeletonize
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    logging.warning("scikit-image not available, falling back to OpenCV thinning")


class Skeletonizer:
    """
    Service for skeletonizing binary images to extract thin line structures.
    Uses OpenCV thinning or scikit-image skeletonize if available.
    """
    
    @staticmethod
    def run(img: np.ndarray, method: str = 'opencv') -> np.ndarray:
        """
        Run skeletonization on a binary image.
        
        Args:
            img (np.ndarray): Binary image (0/255)
            method (str, optional): Method to use - 'opencv' or 'skimage'
            
        Returns:
            np.ndarray: Skeletonized image
        """
        if img is None or img.size == 0:
            logging.error("Invalid image provided for skeletonization")
            return np.array([])
            
        # Convert to binary if needed
        if len(img.shape) == 3:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Apply threshold
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        else:
            binary = img.copy()
            # Ensure binary values
            if np.max(binary) > 1 and np.max(binary) <= 255:
                _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
        
        # Apply methods
        if method == 'skimage' and SKIMAGE_AVAILABLE:
            # Convert to skimage format (True/False)
            bin_bool = (binary > 0)
            # Apply skeletonization
            skeleton_bool = sk_skeletonize(bin_bool)
            # Convert back to uint8 (0/255)
            skeleton = np.where(skeleton_bool, 255, 0).astype(np.uint8)
            logging.debug("Applied scikit-image skeletonization")
        else:
            # Apply OpenCV thinning (Zhang-Suen algorithm)
            skeleton = cv2.ximgproc.thinning(binary)
            logging.debug("Applied OpenCV thinning")
            
        return skeleton
    
    @staticmethod
    def preprocess_for_skeletonization(img: np.ndarray) -> np.ndarray:
        """
        Preprocess an image for better skeletonization results.
        Applies morphological operations to clean the image.
        
        Args:
            img (np.ndarray): Input image
            
        Returns:
            np.ndarray: Preprocessed binary image
        """
        if img is None or img.size == 0:
            return np.array([])
            
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            
        # Apply adaptive threshold for better line detection
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2)
            
        # Invert if needed (lines should be white on black)
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
            
        # Remove noise with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return cleaned 