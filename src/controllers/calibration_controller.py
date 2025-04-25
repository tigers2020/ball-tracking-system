#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calibration Controller module.
This module contains the CalibrationController class which serves as the controller for the Court Calibration tab.
"""

import os
import logging
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from PySide6.QtCore import QObject, Slot, Signal
from PySide6.QtGui import QImage, QPixmap

from src.models.calibration_model import CalibrationModel
from src.views.calibration_tab import CalibrationTab
from src.utils.ui_constants import Messages, Calibration
from src.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class CalibrationController(QObject):
    """
    Controller class for Court Calibration tab.
    Manages communication between the CalibrationModel and CalibrationTab.
    """
    
    # Signal emitted when a status message should be displayed
    status_updated = Signal(str)
    
    def __init__(self, model: CalibrationModel, view: CalibrationTab, config_manager: ConfigManager = None):
        """
        Initialize the calibration controller.
        
        Args:
            model (CalibrationModel): The calibration model
            view (CalibrationTab): The calibration tab view
            config_manager (ConfigManager, optional): Configuration manager
        """
        super().__init__()
        
        self.model = model
        self.view = view
        
        # 뷰에 모델 설정
        self.view.model = model
        
        # Configuration manager
        self.config_manager = config_manager or ConfigManager()
        
        # Current image paths
        self.left_image_path = ""
        self.right_image_path = ""
        
        # Current image service (will be set by the main controller)
        self.image_service = None
        
        # Connect signals from view
        self._connect_signals()
        
        # Initialize config directory if it doesn't exist
        self._init_config_dir()
        
        # Timestamp for throttling saves
        self._last_save_time = 0
    
    def _connect_signals(self):
        """Connect signals from view to controller methods."""
        # Point management signals
        self.view.point_added.connect(self.add_point)
        self.view.point_updated.connect(self.update_point)
        self.view.points_cleared.connect(self.clear_points)
        
        # Button action signals
        self.view.fine_tune_requested.connect(self.fine_tune)
        self.view.save_requested.connect(self.save_calibration)
        self.view.load_requested.connect(self.load_calibration)
        self.view.load_current_frame_requested.connect(self.load_current_frame)
    
    def _init_config_dir(self):
        """Initialize the configuration directory if it doesn't exist."""
        config_dir = os.path.dirname(Calibration.CONFIG_FILE)
        os.makedirs(config_dir, exist_ok=True)
        logger.debug(f"Initialized config directory: {config_dir}")
    
    def _get_current_image_size(self, side: str) -> Optional[Tuple[int, int]]:
        """
        Get the size of the currently loaded image.
        
        Args:
            side (str): 'left' or 'right' side
            
        Returns:
            Optional[Tuple[int, int]]: (width, height) or None if no image
        """
        # Get the image path for the specified side
        image_path = self.left_image_path if side == "left" else self.right_image_path
        
        if not image_path or not os.path.exists(image_path):
            return None
            
        # Get image size using OpenCV
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        height, width = img.shape[:2]
        return (width, height)
    
    @Slot(str, tuple)
    def add_point(self, side: str, position: Tuple[float, float]):
        """
        Add a calibration point to the specified side.
        
        Args:
            side (str): 'left' or 'right' side
            position (Tuple[float, float]): (x, y) coordinates
        """
        logger.debug(f"Adding point at {position} to {side} side")
        
        # Get current image size
        image_size = self._get_current_image_size(side)
        if not image_size:
            logger.error(f"Cannot get image size for {side} side")
            return
            
        width, height = image_size
        
        # Normalize screen coordinates to 1080p standard
        normalized_position = self.model.normalize_points_to_1080p([position], width, height)[0]
        logger.debug(f"Normalized position: {normalized_position} (original: {position})")
        
        # Add point to model using normalized coordinates
        if self.model.add_point(side, normalized_position):
            # Get points from model
            points = self.model.left_pts if side == "left" else self.model.right_pts
            
            # Denormalize 1080p coordinates to current screen coordinates for display
            screen_points = self.model.denormalize_points_from_1080p(points, width, height)
            
            # Update display
            self.view.update_points(side, screen_points)
            
            # Show status message
            self.status_updated.emit(Messages.CALIBRATION_POINTS_ADDED)
            
            # Auto-save with throttling
            self._throttled_save()
    
    @Slot(str, int, tuple)
    def update_point(self, side: str, index: int, position: Tuple[float, float]):
        """
        Update a calibration point.
        
        Args:
            side (str): 'left' or 'right' side
            index (int): Index of point to update
            position (Tuple[float, float]): New (x, y) coordinates
        """
        logger.debug(f"Updating point {index} to {position} on {side} side")
        
        # Get current image size
        image_size = self._get_current_image_size(side)
        if not image_size:
            logger.error(f"Cannot get image size for {side} side")
            return
            
        width, height = image_size
        
        # Normalize screen coordinates to 1080p standard
        normalized_position = self.model.normalize_points_to_1080p([position], width, height)[0]
        logger.debug(f"Normalized position: {normalized_position} (original: {position})")
        
        # Update model with normalized coordinates
        if self.model.update_point(side, index, normalized_position):
            # Get points from model
            points = self.model.left_pts if side == "left" else self.model.right_pts
            
            # Denormalize 1080p coordinates to current screen coordinates for display
            screen_points = self.model.denormalize_points_from_1080p(points, width, height)
            
            # Update display
            self.view.update_points(side, screen_points)
            
            # Show status message
            self.status_updated.emit(Messages.CALIBRATION_POINTS_UPDATED)
            
            # Auto-save with throttling
            self._throttled_save()
    
    @Slot(str)
    def clear_points(self, side=None):
        """
        Clear calibration points for the specified side or both sides.
        
        Args:
            side (str, optional): 'left', 'right', or None for both sides
        """
        logger.debug(f"Clearing points for {side if side else 'both'} side(s)")
        
        # Clear points in model
        self.model.clear_points(side)
        
        # Update view
        if side is None or side == "left":
            self.view.update_points("left", self.model.left_pts)
        
        if side is None or side == "right":
            self.view.update_points("right", self.model.right_pts)
        
        # Show status message
        self.status_updated.emit(Messages.CALIBRATION_POINTS_CLEARED)
    
    @Slot()
    def fine_tune(self):
        """
        Fine-tune calibration points.
        This is a placeholder that will be implemented in Week 3.
        """
        logger.debug("Fine-tune requested (placeholder)")
        self.status_updated.emit(Messages.CALIBRATION_FINE_TUNE_START)
        
        # Placeholder for Week 3 implementation
        self.view.show_info("Fine-tuning will be implemented in Week 3")
        
        self.status_updated.emit(Messages.CALIBRATION_FINE_TUNE_COMPLETE)
    
    @Slot()
    def save_calibration(self):
        """Save calibration data to config.json using ConfigManager."""
        logger.debug("Saving calibration data using ConfigManager")
        
        try:
            # Convert model to dictionary
            data = self.model.to_dict()
            
            # Save to config.json using ConfigManager
            self.config_manager.set("court_calibration", data)
            self.config_manager.save_config(force=True)
            
            # Show status message
            self.status_updated.emit(Messages.CALIBRATION_SAVED)
            self._last_save_time = time.time()
        except Exception as e:
            logger.error(f"Error saving calibration data: {e}")
    
    @Slot()
    def load_calibration(self):
        """Load calibration data from config.json using ConfigManager."""
        logger.debug("Loading calibration data using ConfigManager")
        
        try:
            # Load calibration data from ConfigManager
            calibration_data = self.config_manager.get("court_calibration", None)
            
            if calibration_data:
                # Update model
                if self.model.from_dict(calibration_data):
                    # Get current image sizes for both left and right
                    left_size = self._get_current_image_size("left")
                    right_size = self._get_current_image_size("right")
                    
                    # Update left points
                    if left_size:
                        left_width, left_height = left_size
                        # Denormalize 1080p coordinates to current screen resolution
                        screen_left_pts = self.model.denormalize_points_from_1080p(
                            self.model.left_pts, left_width, left_height)
                        self.view.update_points("left", screen_left_pts)
                    else:
                        # Use original coordinates if image size is unknown
                        self.view.update_points("left", self.model.left_pts)
                    
                    # Update right points
                    if right_size:
                        right_width, right_height = right_size
                        # Denormalize 1080p coordinates to current screen resolution
                        screen_right_pts = self.model.denormalize_points_from_1080p(
                            self.model.right_pts, right_width, right_height)
                        self.view.update_points("right", screen_right_pts)
                    else:
                        # Use original coordinates if image size is unknown
                        self.view.update_points("right", self.model.right_pts)
                    
                    # Show status message
                    self.status_updated.emit(Messages.CALIBRATION_LOADED)
                else:
                    logger.warning("Invalid calibration data format")
            else:
                logger.warning("No calibration data found in config")
        except Exception as e:
            logger.error(f"Error loading calibration data: {e}")
    
    @Slot()
    def load_current_frame(self):
        """Load the current frame images for calibration."""
        logger.debug("Loading current frame images for calibration")
        
        # Check if image service is available
        if self.image_service is None:
            logger.warning("Image service is not available")
            return
        
        try:
            # Get current frame images
            left_image_path, right_image_path = self.image_service.get_current_frame_paths()
            logger.debug(f"Retrieved paths from image service: left={left_image_path}, right={right_image_path}")
            
            if left_image_path and right_image_path:
                # Set images in view
                self.set_images(left_image_path, right_image_path)
                
                # Show status message
                self.status_updated.emit("Current frame images loaded")
            else:
                logger.warning("No current frame images available")
        except Exception as e:
            logger.error(f"Error loading current frame images: {e}")
            # Print stack trace for debugging
            import traceback
            logger.error(traceback.format_exc())
    
    def _throttled_save(self):
        """Save calibration data with throttling to avoid too frequent I/O."""
        current_time = time.time()
        
        # Only save if enough time has passed since the last save
        if current_time - self._last_save_time > Calibration.CONFIG_SAVE_COOLDOWN:
            self.save_calibration()
    
    def set_images(self, left_image_path: str, right_image_path: str):
        """
        Set the images for calibration.
        
        Args:
            left_image_path (str): Path to the left image
            right_image_path (str): Path to the right image
        """
        self.left_image_path = left_image_path
        self.right_image_path = right_image_path
        
        # Set images in view
        self.view.set_images(left_image_path, right_image_path)
        
        # Update model points in view (to match current resolution)
        if self.model and (self.model.left_pts or self.model.right_pts):
            self.load_calibration()
        
        logger.debug(f"Set calibration images: {left_image_path}, {right_image_path}")
    
    def set_image_service(self, image_service):
        """
        Set the image service for accessing current frames.
        
        Args:
            image_service: The image service
        """
        self.image_service = image_service
        logger.info("Image service set for calibration controller")
        
        # Verify image service is working
        if self.image_service is not None:
            paths = self.image_service.get_current_frame_paths()
            logger.info(f"Image service paths: {paths}")
        else:
            logger.warning("Image service is None")
    
    def initialize(self):
        """Initialize the controller, typically after all components are set up."""
        # Try to load existing calibration data
        self.load_calibration()
            
        logger.debug("Calibration controller initialized") 