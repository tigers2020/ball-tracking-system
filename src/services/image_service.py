#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image Service module.
This module contains the ImageService class which manages image operations and frame access.
"""

import os
import logging
from typing import Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ImageService:
    """
    Service for managing images and providing access to current frames.
    """
    
    def __init__(self, base_dir: str = None):
        """
        Initialize the image service.
        
        Args:
            base_dir (str, optional): Base directory for storing images
        """
        self.base_dir = base_dir or os.path.join(os.path.expanduser("~"), "cpet_tennis")
        
        # Ensure base directory exists
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Current frame paths
        self._left_frame_path = None
        self._right_frame_path = None
        
        logger.debug(f"ImageService initialized with base directory: {self.base_dir}")
    
    def set_current_frame_paths(self, left_path: str, right_path: str) -> bool:
        """
        Set the current frame image paths.
        
        Args:
            left_path (str): Path to the left camera image
            right_path (str): Path to the right camera image
            
        Returns:
            bool: True if paths were successfully set
        """
        # Validate paths
        if not os.path.exists(left_path) or not os.path.exists(right_path):
            logger.error("One or both frame paths do not exist")
            return False
        
        self._left_frame_path = left_path
        self._right_frame_path = right_path
        logger.debug(f"Current frame paths set: {left_path}, {right_path}")
        return True
    
    def get_current_frame_paths(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the current frame image paths.
        
        Returns:
            Tuple[Optional[str], Optional[str]]: Paths to the left and right camera images
        """
        return self._left_frame_path, self._right_frame_path
    
    def save_frame_images(self, left_image, right_image, prefix: str = "frame") -> Tuple[str, str]:
        """
        Save frame images to disk.
        
        Args:
            left_image: Left camera image (numpy array or QImage)
            right_image: Right camera image (numpy array or QImage)
            prefix (str): Prefix for the filenames
            
        Returns:
            Tuple[str, str]: Paths to the saved images
        """
        import cv2
        import numpy as np
        from datetime import datetime
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory
        frames_dir = os.path.join(self.base_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Generate filenames
        left_path = os.path.join(frames_dir, f"{prefix}_left_{timestamp}.png")
        right_path = os.path.join(frames_dir, f"{prefix}_right_{timestamp}.png")
        
        # Check image type and save
        try:
            # For numpy arrays (OpenCV images)
            if isinstance(left_image, np.ndarray):
                cv2.imwrite(left_path, left_image)
            # For QImage
            else:
                left_image.save(left_path)
                
            if isinstance(right_image, np.ndarray):
                cv2.imwrite(right_path, right_image)
            else:
                right_image.save(right_path)
            
            # Update current frame paths
            self._left_frame_path = left_path
            self._right_frame_path = right_path
            
            logger.debug(f"Frame images saved: {left_path}, {right_path}")
            return left_path, right_path
        
        except Exception as e:
            logger.error(f"Error saving frame images: {e}")
            return None, None 