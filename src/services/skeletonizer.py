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
            method (str, optional): Method to use - 'opencv', 'skimage', or 'morph'
            
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
        
        # Auto-select method if ximgproc is not available
        has_ximgproc = hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning')
        if method == 'opencv' and not has_ximgproc:
            method = 'skimage' if SKIMAGE_AVAILABLE else 'morph'
            logging.warning(f"OpenCV ximgproc thinning not available, using {method} method instead")
        
        # Apply methods
        if method == 'skimage' and SKIMAGE_AVAILABLE:
            # Convert to skimage format (True/False)
            bin_bool = (binary > 0)
            # Apply skeletonization
            skeleton_bool = sk_skeletonize(bin_bool)
            # Convert back to uint8 (0/255)
            skeleton = np.where(skeleton_bool, 255, 0).astype(np.uint8)
            logging.debug("Applied scikit-image skeletonization")
        elif method == 'morph':
            # Apply morphological thinning as fallback
            skeleton = Skeletonizer._morphological_thinning(binary)
            logging.debug("Applied morphological thinning")
        else:
            # Apply OpenCV thinning (Zhang-Suen algorithm) if available
            try:
                skeleton = cv2.ximgproc.thinning(binary)
                logging.debug("Applied OpenCV thinning")
            except (AttributeError, cv2.error) as e:
                logging.warning(f"OpenCV thinning failed: {str(e)}")
                # Fallback to morphological thinning
                skeleton = Skeletonizer._morphological_thinning(binary)
                logging.debug("Falling back to morphological thinning")
            
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
    
    @staticmethod
    def _morphological_thinning(img: np.ndarray, max_iterations: int = 100) -> np.ndarray:
        """
        Apply iterative morphological thinning to binary image.
        This is a fallback method when OpenCV ximgproc or scikit-image is not available.
        
        Args:
            img (np.ndarray): Binary image (0/255)
            max_iterations (int): Maximum number of iterations to prevent infinite loops
            
        Returns:
            np.ndarray: Thinned skeleton image
        """
        if img is None or img.size == 0:
            return np.array([])
        
        # Ensure binary image (white objects on black background)
        if np.mean(img) > 127:
            img = cv2.bitwise_not(img)
        
        # Create kernels for thinning
        kernel1 = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=np.uint8)
        kernel2 = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0]], dtype=np.uint8)
        
        # Additional rotation kernels
        kernel3 = np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=np.uint8)
        kernel4 = np.array([[0, 0, 1], [0, 1, 1], [0, 0, 1]], dtype=np.uint8)
        kernel5 = np.array([[0, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=np.uint8)
        kernel6 = np.array([[0, 1, 0], [1, 1, 0], [0, 1, 0]], dtype=np.uint8)
        
        # Define all kernels for morphological operations
        kernels = [kernel1, kernel2, kernel3, kernel4, kernel5, kernel6]
        
        # Make a copy to work on
        thinned = img.copy()
        
        # Iteratively apply thinning
        iteration = 0
        while iteration < max_iterations:
            previous = thinned.copy()
            
            # Apply each kernel
            for kernel in kernels:
                # Hit-or-miss transform
                erosion = cv2.erode(thinned, kernel, iterations=1)
                temp = cv2.dilate(erosion, kernel, iterations=1)
                # Remove pixels that match the pattern
                thinned = cv2.subtract(thinned, temp)
            
            # Check if image changed
            if np.array_equal(previous, thinned):
                break
                
            iteration += 1
        
        logging.debug(f"Morphological thinning completed in {iteration} iterations")
        return thinned 