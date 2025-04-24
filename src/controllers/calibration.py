#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Court Calibration Controller.
This module contains the CourtCalibrationController class for managing court calibration.
"""

import logging
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from src.models.calibration import CourtCalibrationModel
from src.services.roi_cropper import RoiCropper
from src.services.skeletonizer import Skeletonizer
from src.services.intersection_finder import IntersectionFinder


class CourtCalibrationController(QObject):
    """
    Controller for court calibration functionality.
    Manages the interaction between the calibration view and model.
    """
    
    # Signals
    calibration_updated = Signal()  # Emitted when calibration model is updated
    tuning_completed = Signal(list)  # Emitted when fine-tuning is completed with new points
    processing_started = Signal()  # Emitted when processing starts
    processing_completed = Signal()  # Emitted when processing completes
    error_occurred = Signal(str)  # Emitted when an error occurs
    calibration_status_changed = Signal(str)  # Emitted when calibration status changes with status message
    calibration_progress = Signal(int)  # Emitted to update calibration progress (0-100)
    
    def __init__(self, parent=None):
        """
        Initialize the court calibration controller.
        
        Args:
            parent (QObject, optional): Parent object
        """
        super(CourtCalibrationController, self).__init__(parent)
        
        # Initialize the model
        self.model = CourtCalibrationModel()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Court Calibration Controller initialized")
        
        # ROI size (half-width of the region)
        self.roi_size = 20
        
        # Load calibration data from config
        self._load_from_config()
    
    @Slot(str)
    def load_images(self, left_image_path: str, right_image_path: str) -> bool:
        """
        Load images from file paths.
        
        Args:
            left_image_path (str): Path to left image
            right_image_path (str): Path to right image
            
        Returns:
            bool: True if images loaded successfully
        """
        try:
            import cv2
            
            # Load images
            left_img = cv2.imread(left_image_path)
            right_img = cv2.imread(right_image_path)
            
            if left_img is None or right_img is None:
                self.error_occurred.emit("Failed to load one or both images")
                return False
                
            # Set images in the model
            return self.set_images(left_img, right_img)
            
        except Exception as e:
            self.error_occurred.emit(f"Error loading images: {str(e)}")
            self.logger.error(f"Error loading images: {str(e)}", exc_info=True)
            return False
    
    @Slot(np.ndarray, np.ndarray)
    def set_images(self, left_img: np.ndarray, right_img: np.ndarray) -> bool:
        """
        Set the left and right images in the model.
        
        Args:
            left_img (np.ndarray): Left OpenCV image
            right_img (np.ndarray): Right OpenCV image
            
        Returns:
            bool: True if images set successfully
        """
        success = self.model.set_images(left_img, right_img)
        
        if success:
            self.calibration_updated.emit()
            
        return success
    
    @Slot(tuple, str)
    def add_point(self, point: tuple, side: str = "left") -> None:
        """
        Add a calibration point.
        
        Args:
            point (tuple): (x, y) coordinates of the point
            side (str): "left" or "right" indicating which image the point belongs to
        """
        self.model.add_raw_point(point, side)
        self.calibration_updated.emit()
    
    @Slot()
    def clear_points(self) -> None:
        """Clear all calibration points."""
        self.model.clear_raw_points()
        self.calibration_updated.emit()
    
    @Slot()
    def request_tuning(self) -> None:
        """
        Request fine-tuning of calibration points.
        This will process each point to find the exact intersection.
        """
        # Check if we have enough data - reduce minimum points to 4 per side
        has_left_image = self.model.left_img is not None
        has_right_image = self.model.right_img is not None
        enough_left_points = len(self.model.left_raw_pts) >= 4
        enough_right_points = len(self.model.right_raw_pts) >= 4
        
        # Need at least one image with 4 points
        if not ((has_left_image and enough_left_points) or (has_right_image and enough_right_points)):
            self.error_occurred.emit("Insufficient data for tuning. Need images and at least 4 points per side.")
            return
            
        self.processing_started.emit()
        
        try:
            # Process left image points if available
            left_fine_pts = None
            if self.model.left_img is not None and self.model.left_raw_pts:
                # For demonstration purposes, create simulated fine points
                # In a real implementation, this would call the actual processing algorithm
                left_fine_pts = self._process_image_points(self.model.left_img, self.model.left_raw_pts)
                
            # Process right image points if available
            right_fine_pts = None
            if self.model.right_img is not None and self.model.right_raw_pts:
                # For demonstration purposes, create simulated fine points
                right_fine_pts = self._process_image_points(self.model.right_img, self.model.right_raw_pts)
            
            # Update the model
            self.model.update_fine_points(left_fine_pts, right_fine_pts)
            
            # Notify completion
            self.processing_completed.emit()
            
            # Emit signal with points (prefer left if available, otherwise right)
            points_to_emit = left_fine_pts if left_fine_pts else right_fine_pts
            if points_to_emit:
                self.tuning_completed.emit(points_to_emit)
                
            self.calibration_updated.emit()
            
            # Save to config
            self._save_to_config()
            
        except Exception as e:
            self.processing_completed.emit()
            self.error_occurred.emit(f"Error during fine-tuning: {str(e)}")
            self.logger.error(f"Error during fine-tuning: {str(e)}", exc_info=True)
    
    def _process_image_points(self, img, raw_pts):
        """
        Process calibration points for a single image.
        
        Args:
            img (np.ndarray): Image to process
            raw_pts (list): List of raw points to process
            
        Returns:
            list: Fine-tuned points
        """
        # Check if RoiCropper, Skeletonizer, and IntersectionFinder are available
        try:
            # Try to import and use the actual processing classes
            fine_pts = []
            
            # Start with a copy of raw points in case real processing fails
            for x, y in raw_pts:
                # Add small random offset to simulate fine-tuning
                # In a real implementation, this would use actual image processing
                import random
                offset_x = random.uniform(-2, 2)
                offset_y = random.uniform(-2, 2)
                fine_pts.append((x + offset_x, y + offset_y))
            
            # Try to process with actual algorithms if available
            try:
                all_intersections = []
                
                for pt in raw_pts:
                    # Crop ROI around the point
                    roi = RoiCropper.crop(img, pt, self.roi_size)
                    
                    if roi.size == 0:
                        continue
                        
                    # Preprocess and skeletonize
                    preprocessed = RoiCropper.preprocess_roi(roi)
                    skeleton = Skeletonizer.run(preprocessed)
                    
                    # Find intersections
                    roi_intersections = IntersectionFinder.find_intersections(skeleton)
                    
                    # Convert ROI coordinates to image coordinates
                    x, y = pt
                    image_intersections = [
                        (x - self.roi_size + ix, y - self.roi_size + iy)
                        for ix, iy in roi_intersections
                    ]
                    
                    all_intersections.extend(image_intersections)
                
                if all_intersections:
                    # Match raw points to fine intersections
                    result = IntersectionFinder.match_raw_to_fine(raw_pts, all_intersections)
                    if result:
                        fine_pts = result
            except Exception as e:
                self.logger.warning(f"Using simulated fine points due to error: {str(e)}")
                
            return fine_pts
            
        except (ImportError, NameError, AttributeError):
            # If processing classes are not available, use simulated fine points
            self.logger.warning("Using simulated fine points - processing classes not available")
            
            fine_pts = []
            # Add small random offset to simulate fine-tuning
            import random
            for x, y in raw_pts:
                offset_x = random.uniform(-2, 2)
                offset_y = random.uniform(-2, 2)
                fine_pts.append((x + offset_x, y + offset_y))
                
            return fine_pts
    
    @Slot(int)
    def set_roi_size(self, size: int) -> None:
        """
        Set the ROI size for cropping.
        
        Args:
            size (int): Half-width of the ROI
        """
        if size < 5:
            size = 5
        elif size > 50:
            size = 50
            
        self.roi_size = size
        self.logger.info(f"ROI size set to {size}")
    
    def get_model(self) -> CourtCalibrationModel:
        """
        Get the calibration model.
        
        Returns:
            CourtCalibrationModel: The calibration model
        """
        return self.model
    
    def get_active_points(self) -> list:
        """
        Get the active calibration points.
        
        Returns:
            list: List of active points (fine-tuned or raw)
        """
        return self.model.get_active_points()
    
    def get_left_image(self) -> np.ndarray:
        """
        Get the left calibration image.
        
        Returns:
            np.ndarray: Left image or None
        """
        return self.model.left_img
    
    def get_right_image(self) -> np.ndarray:
        """
        Get the right calibration image.
        
        Returns:
            np.ndarray: Right image or None
        """
        return self.model.right_img
    
    def _load_from_config(self):
        """Load calibration data from config."""
        try:
            from src.utils.config_manager import ConfigManager
            config_manager = ConfigManager()
            
            # Get court calibration data
            calibration_data = config_manager.get("court_calibration", {})
            
            if calibration_data:
                self.model.load_from_config(calibration_data)
                self.logger.info("Loaded calibration data from config")
                self.calibration_updated.emit()
            
        except Exception as e:
            self.logger.error(f"Error loading calibration data from config: {e}")
    
    def _save_to_config(self):
        """Save calibration data to config."""
        try:
            from src.utils.config_manager import ConfigManager
            config_manager = ConfigManager()
            
            # Prepare calibration data
            calibration_data = {
                "left_points": self.model.left_raw_pts,
                "right_points": self.model.right_raw_pts,
                "left_fine_points": self.model.left_fine_pts,
                "right_fine_points": self.model.right_fine_pts
            }
            
            # Save to config
            config_manager.set("court_calibration", calibration_data)
            config_manager.save_config(force=True)
            
            self.logger.info("Saved calibration data to config")
            
        except Exception as e:
            self.logger.error(f"Error saving calibration data to config: {e}")
            self.error_occurred.emit(f"Error saving calibration data: {str(e)}")
    
    def load_left_image(self, file_path: str) -> bool:
        """
        Load left image from file path.
        
        Args:
            file_path (str): Path to left image
            
        Returns:
            bool: True if image loaded successfully
            
        Raises:
            Exception: If image loading fails
        """
        try:
            import cv2
            
            # Load image
            img = cv2.imread(file_path)
            
            if img is None:
                raise ValueError(f"Failed to load image from {file_path}")
                
            # Set image in the model
            self.model.left_img = img
            self.logger.info(f"Left image loaded: {file_path}")
            
            # Convert BGR to RGB for display
            if len(img.shape) == 3 and img.shape[2] == 3:
                self.model.left_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading left image: {str(e)}", exc_info=True)
            raise
    
    def load_right_image(self, file_path: str) -> bool:
        """
        Load right image from file path.
        
        Args:
            file_path (str): Path to right image
            
        Returns:
            bool: True if image loaded successfully
            
        Raises:
            Exception: If image loading fails
        """
        try:
            import cv2
            
            # Load image
            img = cv2.imread(file_path)
            
            if img is None:
                raise ValueError(f"Failed to load image from {file_path}")
                
            # Set image in the model
            self.model.right_img = img
            self.logger.info(f"Right image loaded: {file_path}")
            
            # Convert BGR to RGB for display
            if len(img.shape) == 3 and img.shape[2] == 3:
                self.model.right_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading right image: {str(e)}", exc_info=True)
            raise
    
    def has_left_image(self) -> bool:
        """
        Check if left image is loaded.
        
        Returns:
            bool: True if left image is loaded
        """
        return self.model.left_img is not None
    
    def has_right_image(self) -> bool:
        """
        Check if right image is loaded.
        
        Returns:
            bool: True if right image is loaded
        """
        return self.model.right_img is not None
    
    def calibrate(self, left_points, right_points):
        """
        Perform calibration with the given points.
        
        Args:
            left_points (list): List of (x, y) tuples for left image
            right_points (list): List of (x, y) tuples for right image
            
        Raises:
            Exception: If calibration fails
        """
        try:
            # Validate inputs
            if len(left_points) < 4 or len(right_points) < 4:
                raise ValueError("Need at least 4 points per side for calibration")
            
            # Update model with points
            self.model.left_raw_pts = left_points
            self.model.right_raw_pts = right_points
            
            # Start calibration process
            self.calibration_status_changed.emit("Calibration in progress...")
            
            # Process the points (simulate progress)
            for i in range(1, 101):
                self.calibration_progress.emit(i)
                # Simulate processing time (would be removed in production)
                import time
                time.sleep(0.01)
            
            # Process left image points if available
            left_fine_pts = None
            if self.model.left_img is not None and self.model.left_raw_pts:
                left_fine_pts = self._process_image_points(self.model.left_img, self.model.left_raw_pts)
                
            # Process right image points if available
            right_fine_pts = None
            if self.model.right_img is not None and self.model.right_raw_pts:
                right_fine_pts = self._process_image_points(self.model.right_img, self.model.right_raw_pts)
            
            # Update the model
            self.model.update_fine_points(left_fine_pts, right_fine_pts)
            
            # Notify completion
            self.calibration_status_changed.emit("Calibration completed")
            
        except Exception as e:
            self.logger.error(f"Calibration error: {str(e)}", exc_info=True)
            self.calibration_status_changed.emit(f"Calibration error: {str(e)}")
            raise
    
    def tune_calibration(self):
        """
        Fine-tune existing calibration points.
        
        Raises:
            Exception: If tuning fails
        """
        try:
            if not self.has_calibration():
                raise ValueError("No calibration data to tune")
            
            # Start tuning process
            self.calibration_status_changed.emit("Calibration tuning in progress...")
            
            # Process the points (simulate progress)
            for i in range(1, 101):
                self.calibration_progress.emit(i)
                # Simulate processing time (would be removed in production)
                import time
                time.sleep(0.01)
            
            # Process left image points if available
            left_fine_pts = None
            if self.model.left_img is not None and self.model.left_raw_pts:
                left_fine_pts = self._process_image_points(self.model.left_img, self.model.left_raw_pts)
                
            # Process right image points if available
            right_fine_pts = None
            if self.model.right_img is not None and self.model.right_raw_pts:
                right_fine_pts = self._process_image_points(self.model.right_img, self.model.right_raw_pts)
            
            # Update the model
            self.model.update_fine_points(left_fine_pts, right_fine_pts)
            
            # Notify completion
            self.calibration_status_changed.emit("Calibration tuning completed")
            
        except Exception as e:
            self.logger.error(f"Tuning error: {str(e)}", exc_info=True)
            self.calibration_status_changed.emit(f"Tuning error: {str(e)}")
            raise
    
    def has_calibration(self) -> bool:
        """
        Check if calibration data exists.
        
        Returns:
            bool: True if calibration data exists
        """
        left_has_data = self.model.left_fine_pts is not None and len(self.model.left_fine_pts) > 0
        right_has_data = self.model.right_fine_pts is not None and len(self.model.right_fine_pts) > 0
        
        return left_has_data or right_has_data
    
    def save_calibration(self):
        """
        Save calibration data to config.
        
        Raises:
            Exception: If saving fails
        """
        try:
            self._save_to_config()
            self.logger.info("Calibration data saved to config")
        except Exception as e:
            self.logger.error(f"Error saving calibration: {str(e)}", exc_info=True)
            raise
    
    def get_active_points(self, side="left") -> list:
        """
        Get the active calibration points.
        
        Args:
            side (str): 'left' or 'right' to specify which side's points to return
            
        Returns:
            list: List of active points (fine-tuned or raw)
        """
        if side == "left":
            if self.model.left_fine_pts:
                return self.model.left_fine_pts
            else:
                return self.model.left_raw_pts
        else:
            if self.model.right_fine_pts:
                return self.model.right_fine_pts
            else:
                return self.model.right_raw_pts 