#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Court Calibration Model.
This module contains the CourtCalibrationModel class for managing court calibration data.
"""

import logging
from dataclasses import dataclass, field
import numpy as np


@dataclass
class CourtCalibrationModel:
    """
    Model for court calibration data.
    Stores the original images, raw points selected by user, and fine-tuned points.
    """
    left_img: np.ndarray = None
    right_img: np.ndarray = None
    left_raw_pts: list[tuple[int, int]] = field(default_factory=list)
    right_raw_pts: list[tuple[int, int]] = field(default_factory=list)
    left_fine_pts: list[tuple[float, float]] = field(default_factory=list)
    right_fine_pts: list[tuple[float, float]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize logger and validate attributes."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Court Calibration Model")
    
    def set_images(self, left_img: np.ndarray, right_img: np.ndarray) -> bool:
        """
        Set the left and right calibration images.
        
        Args:
            left_img (np.ndarray): Left image
            right_img (np.ndarray): Right image
            
        Returns:
            bool: True if images were set successfully
        """
        if left_img is None and right_img is None:
            self.logger.error("Invalid images provided for calibration")
            return False
        
        # Set left image if provided
        if left_img is not None:
            if len(left_img.shape) != 3:
                self.logger.error("Left image must be RGB/BGR (3 channels)")
                return False
            self.left_img = left_img.copy()
            
        # Set right image if provided
        if right_img is not None:
            if len(right_img.shape) != 3:
                self.logger.error("Right image must be RGB/BGR (3 channels)")
                return False
            self.right_img = right_img.copy()
            
        self.logger.info("Calibration images set successfully")
        return True
    
    def add_raw_point(self, point: tuple[int, int], side: str = "left") -> None:
        """
        Add a raw calibration point for either left or right image.
        
        Args:
            point (tuple[int, int]): (x, y) coordinates of the point
            side (str): "left" or "right" indicating which image the point belongs to
        """
        if side == "left":
            if point not in self.left_raw_pts:
                self.left_raw_pts.append(point)
                self.logger.info(f"Added left raw calibration point {len(self.left_raw_pts)}: {point}")
        else:  # right
            if point not in self.right_raw_pts:
                self.right_raw_pts.append(point)
                self.logger.info(f"Added right raw calibration point {len(self.right_raw_pts)}: {point}")
    
    def clear_raw_points(self) -> None:
        """Clear all raw calibration points."""
        self.left_raw_pts = []
        self.right_raw_pts = []
        self.left_fine_pts = []
        self.right_fine_pts = []
        self.logger.info("Cleared all calibration points")
    
    def update_fine_points(self, left_fine_pts: list[tuple[float, float]] = None, 
                          right_fine_pts: list[tuple[float, float]] = None) -> None:
        """
        Update fine-tuned calibration points.
        
        Args:
            left_fine_pts (list[tuple[float, float]], optional): List of fine-tuned (x, y) points for left image
            right_fine_pts (list[tuple[float, float]], optional): List of fine-tuned (x, y) points for right image
        """
        if left_fine_pts is not None:
            self.left_fine_pts = left_fine_pts
            self.logger.info(f"Updated {len(left_fine_pts)} fine-tuned calibration points for left image")
            
        if right_fine_pts is not None:
            self.right_fine_pts = right_fine_pts
            self.logger.info(f"Updated {len(right_fine_pts)} fine-tuned calibration points for right image")
    
    def get_active_points(self, side: str = "left") -> list[tuple]:
        """
        Get the currently active set of points for display or computation.
        
        Args:
            side (str): "left" or "right" indicating which image's points to return
            
        Returns:
            list[tuple]: Fine-tuned points if available, otherwise raw points for specified side
        """
        if side == "left":
            return self.left_fine_pts if self.left_fine_pts else self.left_raw_pts
        else:  # right
            return self.right_fine_pts if self.right_fine_pts else self.right_raw_pts
    
    def is_ready_for_tuning(self) -> bool:
        """
        Check if model has enough data for tuning.
        
        Returns:
            bool: True if model has images and sufficient raw points (at least 4 points per side)
        """
        has_left_image = self.left_img is not None
        has_right_image = self.right_img is not None
        enough_left_points = len(self.left_raw_pts) >= 4
        enough_right_points = len(self.right_raw_pts) >= 4
        
        # Need at least one image with 4 points
        return (has_left_image and enough_left_points) or (has_right_image and enough_right_points)
        
    def load_from_config(self, config_data: dict) -> bool:
        """
        Load calibration points from configuration data.
        
        Args:
            config_data (dict): Configuration data containing court calibration points
            
        Returns:
            bool: True if data was loaded successfully
        """
        try:
            if "left_points" in config_data:
                self.left_raw_pts = [tuple(pt) for pt in config_data["left_points"]]
                
            if "right_points" in config_data:
                self.right_raw_pts = [tuple(pt) for pt in config_data["right_points"]]
                
            if "left_fine_points" in config_data:
                self.left_fine_pts = [tuple(pt) for pt in config_data["left_fine_points"]]
                
            if "right_fine_points" in config_data:
                self.right_fine_pts = [tuple(pt) for pt in config_data["right_fine_points"]]
                
            self.logger.info("Calibration points loaded from configuration")
            return True
        except Exception as e:
            self.logger.error(f"Error loading calibration points from configuration: {e}")
            return False 