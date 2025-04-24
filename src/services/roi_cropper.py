#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ROI Cropper Service.
This service provides functionality for cropping regions of interest (ROIs) around points.
"""

import logging
import numpy as np
import cv2


class RoiCropper:
    """
    Service for cropping regions of interest from images.
    """
    
    @staticmethod
    def crop(img: np.ndarray, pt: tuple[int, int], size: int = 20) -> np.ndarray:
        """
        Crop a region of interest around a point.
        
        Args:
            img (np.ndarray): Source image
            pt (tuple[int, int]): Center point (x, y)
            size (int, optional): Half-size of the ROI (resulting ROI is 2*size x 2*size)
            
        Returns:
            np.ndarray: Cropped region or empty array if crop is out of bounds
        """
        if img is None or not isinstance(img, np.ndarray):
            logging.error("Invalid image provided for cropping")
            return np.array([])
            
        x, y = pt
        h, w = img.shape[:2]
        
        # Ensure crop region is within image bounds
        x1 = max(0, x - size)
        y1 = max(0, y - size)
        x2 = min(w, x + size)
        y2 = min(h, y + size)
        
        # Check if resulting crop has sufficient size
        if x2 - x1 < size or y2 - y1 < size:
            logging.warning(f"Crop at point {pt} is too close to image boundary")
        
        # Return the cropped region
        return img[y1:y2, x1:x2].copy()
    
    @staticmethod
    def crop_multiple(img: np.ndarray, pts: list[tuple[int, int]], size: int = 20) -> list[np.ndarray]:
        """
        Crop multiple regions of interest around points.
        
        Args:
            img (np.ndarray): Source image
            pts (list[tuple[int, int]]): List of center points (x, y)
            size (int, optional): Half-size of the ROI
            
        Returns:
            list[np.ndarray]: List of cropped regions
        """
        return [RoiCropper.crop(img, pt, size) for pt in pts]
    
    @staticmethod
    def preprocess_roi(roi: np.ndarray, threshold: int = 200) -> np.ndarray:
        """
        Preprocess a ROI for line detection.
        Converts to grayscale and applies binary thresholding.
        
        Args:
            roi (np.ndarray): ROI image
            threshold (int, optional): Binary threshold value
            
        Returns:
            np.ndarray: Preprocessed binary image
        """
        if roi.size == 0:
            return np.array([])
            
        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
            
        # Apply binary thresholding
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Apply morphological opening to remove noise
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return cleaned 