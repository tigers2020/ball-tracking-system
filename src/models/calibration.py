#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Court Calibration Model.
This module contains the CourtCalibrationModel class for managing court calibration data.
"""

import logging
from dataclasses import dataclass, field
import numpy as np
from PySide6.QtCore import QObject, Signal


class CourtCalibrationModel(QObject):
    """
    Model for court calibration data.
    Stores the original images and calibration points.
    """
    # 시그널 정의
    calibration_updated = Signal()  # 보정 데이터가 변경될 때 발생하는 시그널
    
    def __init__(self):
        """Initialize model and logger."""
        super(CourtCalibrationModel, self).__init__()
        self.left_img = None
        self.right_img = None
        self.left_pts = []   # 통합된 하나의 포인트 배열
        self.right_pts = []  # 통합된 하나의 포인트 배열
        
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
    
    def add_point(self, point: tuple[int, int], side: str = "left") -> None:
        """
        Add a calibration point for either left or right image.
        
        Args:
            point (tuple[int, int]): (x, y) coordinates of the point
            side (str): "left" or "right" indicating which image the point belongs to
        """
        if side == "left":
            if point not in self.left_pts:
                self.left_pts.append(point)
                self.logger.info(f"Added left calibration point {len(self.left_pts)}: {point}")
        else:  # right
            if point not in self.right_pts:
                self.right_pts.append(point)
                self.logger.info(f"Added right calibration point {len(self.right_pts)}: {point}")
        
        # 포인트 추가 시 신호 발생
        self.calibration_updated.emit()
    
    def update_point(self, side: str, idx: int, new_pos: tuple[int, int]) -> None:
        """
        Update a point at specific index with new position.
        
        Args:
            side (str): "left" or "right" indicating which side to update
            idx (int): Index of the point to update
            new_pos (tuple[int, int]): New (x, y) coordinates
        """
        pts = self.left_pts if side == "left" else self.right_pts
        
        if 0 <= idx < len(pts):
            pts[idx] = new_pos
            self.logger.debug(f"Updated {side} point {idx} to {new_pos}")
            self.calibration_updated.emit()
        else:
            self.logger.warning(f"Invalid point index {idx} for {side} side")
    
    def update_points(self, side: str, new_points: list[tuple[int, int]]) -> None:
        """
        Update all points for specified side with fine-tuned new positions.
        
        Args:
            side (str): "left" or "right" indicating which side to update
            new_points (list[tuple[int, int]]): New positions for all points
        """
        if not new_points:
            self.logger.warning(f"Empty point list provided for {side} side")
            return
            
        if side == "left":
            # 기존 포인트 수와 다르면 기존 배열 유지
            if len(new_points) == len(self.left_pts):
                self.left_pts = new_points.copy()
                self.logger.info(f"Updated {len(new_points)} points for left image")
            else:
                self.logger.warning(f"Point count mismatch for left side: {len(self.left_pts)} vs {len(new_points)}")
        else:  # right
            if len(new_points) == len(self.right_pts):
                self.right_pts = new_points.copy()
                self.logger.info(f"Updated {len(new_points)} points for right image")
            else:
                self.logger.warning(f"Point count mismatch for right side: {len(self.right_pts)} vs {len(new_points)}")
        
        # 포인트 업데이트 시 신호 발생
        self.calibration_updated.emit()
    
    def clear_points(self) -> None:
        """Clear all calibration points."""
        self.left_pts = []
        self.right_pts = []
        self.logger.info("Cleared all calibration points")
        self.calibration_updated.emit()
    
    def get_points(self, side: str = "left") -> list[tuple]:
        """
        Get the calibration points for display or computation.
        
        Args:
            side (str): "left" or "right" indicating which image's points to return
            
        Returns:
            list[tuple]: Calibration points for specified side
        """
        return self.left_pts if side == "left" else self.right_pts
    
    def is_ready_for_tuning(self) -> bool:
        """
        Check if model has enough data for tuning.
        
        Returns:
            bool: True if model has images and sufficient points (at least 4 points per side)
        """
        has_left_image = self.left_img is not None
        has_right_image = self.right_img is not None
        enough_left_points = len(self.left_pts) >= 4
        enough_right_points = len(self.right_pts) >= 4
        
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
            # 이전 형식 지원 (마이그레이션)
            if "left_points" in config_data or "right_points" in config_data:
                self.logger.info("Migrating legacy calibration format")
                
                # 레거시 형식 불러오기
                if "left_points" in config_data:
                    self.left_pts = [tuple(pt) for pt in config_data["left_points"]]
                    # 기존 fine points가 있으면 우선 적용
                    if "left_fine_points" in config_data and config_data["left_fine_points"]:
                        self.left_pts = [tuple(pt) for pt in config_data["left_fine_points"]]
                
                if "right_points" in config_data:
                    self.right_pts = [tuple(pt) for pt in config_data["right_points"]]
                    # 기존 fine points가 있으면 우선 적용
                    if "right_fine_points" in config_data and config_data["right_fine_points"]:
                        self.right_pts = [tuple(pt) for pt in config_data["right_fine_points"]]
            
            # 새 형식 지원
            elif "points" in config_data:
                if "left" in config_data["points"]:
                    self.left_pts = [tuple(pt) for pt in config_data["points"]["left"]]
                
                if "right" in config_data["points"]:
                    self.right_pts = [tuple(pt) for pt in config_data["points"]["right"]]
            
            self.logger.info(f"Calibration points loaded: left={len(self.left_pts)}, right={len(self.right_pts)}")
            self.calibration_updated.emit()
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading calibration points from configuration: {e}")
            return False 