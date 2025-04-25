#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calibration Controller module.
This module contains the CalibrationController class which connects the calibration model and view.
"""

import logging
import os
from typing import List, Tuple, Dict
from pathlib import Path

from PySide6.QtCore import QObject, Slot
from PySide6.QtWidgets import QFileDialog, QMessageBox

from src.models.calibration_model import CalibrationModel
from src.views.calibration_tab import CalibrationTab
from src.utils.config_manager import ConfigManager
import cv2
from PySide6.QtGui import QImage, QPixmap

# Import new services
from src.services.roi_cropper import crop_roi, crop_roi_with_padding
from src.services.skeletonizer import skeletonize_roi
from src.services.intersection_finder import find_and_sort_intersections

import numpy as np

logger = logging.getLogger(__name__)


class CalibrationController(QObject):
    """
    Controller for the calibration functionality.
    Connects the calibration model and view.
    """
    
    def __init__(self, model: CalibrationModel, view: CalibrationTab, config_manager: ConfigManager = None):
        """
        Initialize the calibration controller.
        
        Args:
            model (CalibrationModel): The calibration model
            view (CalibrationTab): The calibration view
            config_manager (ConfigManager, optional): The configuration manager
        """
        super().__init__()
        
        self.model = model
        self.view = view
        self.config_manager = config_manager
        
        # Set up point adding/moving connections
        self.view.point_added.connect(self.on_add_point)
        self.view.point_moved.connect(self.on_move_point)
        
        # Set up button connections
        self.view.clear_button.clicked.connect(self.on_clear_points)
        self.view.fine_tune_button.clicked.connect(self.on_fine_tune)
        self.view.save_button.clicked.connect(self.on_save_to_config)
        self.view.load_button.clicked.connect(self.on_load_from_config)
        self.view.load_current_frame_button.clicked.connect(self.on_load_current_frame)
        
        # Default calibration file directory
        self.default_save_dir = Path.home() / "Court_Calibration"
        
        # Reference to the stereo image model (will be set from app_controller)
        self.stereo_image_model = None
        
        # Try to load calibration points from config if available
        if self.config_manager:
            self._load_points_from_config()
    
    @Slot(str, float, float)
    def on_add_point(self, side: str, x: float, y: float):
        """
        Handle adding a point.
        
        Args:
            side (str): 'left' or 'right'
            x (float): X-coordinate
            y (float): Y-coordinate
        """
        try:
            # Add point to model
            self.model.add_point(side, (x, y))
            
            # Get index of the newly added point
            points = self.model.get_points(side)
            index = len(points) - 1
            
            # Add point item to view
            self.view.add_point_item(side, x, y, index)
            
            # Update grid lines if we have enough points
            self._update_grid_lines(side)
            
            logger.info(f"Added point at ({x}, {y}) to {side} view")
        except Exception as e:
            logger.error(f"Error adding point: {e}")
    
    @Slot(str, int, float, float)
    def on_move_point(self, side: str, index: int, x: float, y: float):
        """
        Handle moving a point.
        
        Args:
            side (str): 'left' or 'right'
            index (int): Point index
            x (float): New X-coordinate
            y (float): New Y-coordinate
        """
        try:
            # Update point in model
            self.model.update_point(side, index, (x, y))
            
            # Update grid lines
            self._update_grid_lines(side)
            
            logger.info(f"Moved point {index} to ({x}, {y}) in {side} view")
        except Exception as e:
            logger.error(f"Error moving point: {e}")
    
    @Slot()
    def on_clear_points(self):
        """Handle clearing all points."""
        try:
            # Clear points in model
            self.model.clear_points()
            
            # Clear points in view
            self.view.clear_points()
            
            logger.info("Cleared all calibration points")
        except Exception as e:
            logger.error(f"Error clearing points: {e}")
    
    @Slot()
    def on_fine_tune(self):
        """
        Handle fine-tuning points using computer vision techniques.
        Extracts a Region of Interest (ROI) around each calibration point,
        skeletonizes the ROI, finds intersections, and updates the point
        to the nearest intersection.
        """
        try:
            # Get stereo image model to access images
            if not self.stereo_image_model:
                logger.warning("No stereo image model available, cannot fine-tune points")
                QMessageBox.warning(
                    self.view,
                    "Fine-Tune Failed",
                    "No stereo image model available. Load images first."
                )
                return
                
            # Get the current frame from stereo image model
            current_frame = self.stereo_image_model.get_current_frame()
            if not current_frame:
                logger.warning("No current frame available, cannot fine-tune points")
                QMessageBox.warning(
                    self.view,
                    "Fine-Tune Failed",
                    "No current frame available. Load images first."
                )
                return
                
            # Get left and right images from the frame
            left_img = current_frame.get_left_image()
            right_img = current_frame.get_right_image()
            
            if left_img is None or right_img is None:
                logger.warning("Failed to get images from current frame, cannot fine-tune points")
                QMessageBox.warning(
                    self.view,
                    "Fine-Tune Failed",
                    "Failed to get images from current frame."
                )
                return
                
            # Fine-tune left points
            self._fine_tune_points('left', left_img)
            
            # Fine-tune right points
            self._fine_tune_points('right', right_img)
            
            # Update grid lines
            self._update_grid_lines('left')
            self._update_grid_lines('right')
            
            # Show success message
            QMessageBox.information(
                self.view,
                "Fine-Tune Successful",
                "Calibration points have been fine-tuned.\n"
                "Points may have been adjusted to nearby intersections."
            )
            
            logger.info("Fine-tuned calibration points successfully")
            
        except Exception as e:
            logger.error(f"Error fine-tuning points: {e}")
            QMessageBox.critical(
                self.view,
                "Fine-Tune Error",
                f"An error occurred while fine-tuning points: {str(e)}"
            )
            
    def _fine_tune_points(self, side: str, image: np.ndarray):
        """
        Fine-tune points for a specific side using computer vision.
        
        Args:
            side (str): 'left' or 'right'
            image (np.ndarray): Image for the specified side
        """
        # Get points for this side
        points = self.model.get_points(side)
        
        # Loop through each point
        for index, point in enumerate(points):
            try:
                # Show ROI overlay to indicate processing
                self.view.show_roi(side, point, 25.0)
                
                # Extract ROI around the point
                roi = crop_roi(image, point, radius=25.0)
                
                if roi is None:
                    logger.warning(f"Failed to crop ROI for {side} point {index}")
                    continue
                
                # Skeletonize ROI
                skeleton = skeletonize_roi(roi)
                
                # Find intersections in skeletonized ROI
                # We'll use the ROI center (half of width/height) as the origin reference 
                # for sorting intersections by proximity
                roi_height, roi_width = roi.shape[:2]
                roi_center = (roi_width // 2, roi_height // 2)
                
                intersections = find_and_sort_intersections(skeleton, roi_center, max_points=3)
                
                # If intersections found, use the closest one
                if intersections:
                    # Get the closest intersection (the first in the sorted list)
                    best_x, best_y = intersections[0]
                    
                    # Calculate the offset to convert ROI coordinates to image coordinates
                    roi_with_padding, (offset_x, offset_y) = crop_roi_with_padding(image, point, radius=25.0)
                    
                    # Adjust intersection coordinates to image coordinates
                    adjusted_x = best_x + offset_x
                    adjusted_y = best_y + offset_y
                    
                    # Update the point in the model
                    self.model.update_point(side, index, (adjusted_x, adjusted_y))
                    
                    # Update the point in the view
                    self.view.update_point_item(side, index, adjusted_x, adjusted_y)
                    
                    logger.info(f"Fine-tuned {side} point {index} from {point} to ({adjusted_x}, {adjusted_y})")
                else:
                    logger.warning(f"No intersections found for {side} point {index}")
                
                # Hide ROI overlay
                self.view.hide_roi(side)
                
            except Exception as e:
                logger.error(f"Error fine-tuning {side} point {index}: {e}")
                
                # Hide ROI overlay on error
                self.view.hide_roi(side)
                
                # Continue to next point rather than aborting
                continue
                
        logger.info(f"Completed fine-tuning {len(points)} {side} points")
    
    @Slot()
    def on_save_to_config(self):
        """
        Save calibration points to the configuration file using ConfigManager.
        Normalizes coordinates to 0-1 range before saving.
        """
        try:
            if not self.config_manager:
                logger.error("No ConfigManager available, cannot save to config")
                QMessageBox.warning(
                    self.view,
                    "Save Failed",
                    "Configuration manager not available"
                )
                return
            
            # Get the image dimensions from view's scenes
            left_scene = self.view.left_scene
            right_scene = self.view.right_scene
            
            left_width = left_scene.width()
            left_height = left_scene.height()
            right_width = right_scene.width()
            right_height = right_scene.height()
            
            # Update model with image dimensions
            self.model.set_image_dimensions('left', left_width, left_height)
            self.model.set_image_dimensions('right', right_width, right_height)
            
            # Get normalized calibration data
            normalized_data = self.model.to_normalized_dict()
            
            # Check if we have all 14 points for both sides
            left_point_count = len(self.model.get_points('left'))
            right_point_count = len(self.model.get_points('right'))
            
            if left_point_count < self.model.MAX_POINTS or right_point_count < self.model.MAX_POINTS:
                warning_msg = (f"Warning: Incomplete calibration points.\n"
                              f"Left: {left_point_count}/{self.model.MAX_POINTS}\n"
                              f"Right: {right_point_count}/{self.model.MAX_POINTS}\n\n"
                              f"Continue anyway?")
                
                reply = QMessageBox.warning(
                    self.view, 
                    "Incomplete Calibration",
                    warning_msg, 
                    QMessageBox.Yes | QMessageBox.No
                )
                
                if reply == QMessageBox.No:
                    return
            
            # Save to config
            self.config_manager.set_calibration_points(normalized_data)
            
            # Show success message
            QMessageBox.information(
                self.view,
                "Save Successful",
                "Calibration points saved to configuration"
            )
            
            logger.info(f"Saved {left_point_count} left points and {right_point_count} right points to config")
        except Exception as e:
            logger.error(f"Error saving calibration points to config: {e}")
            QMessageBox.critical(
                self.view,
                "Save Error",
                f"An error occurred while saving to config: {str(e)}"
            )
    
    @Slot()
    def on_load_from_config(self):
        """
        Load calibration points from the configuration file using ConfigManager.
        Converts normalized coordinates (0-1) to pixel coordinates.
        """
        try:
            if not self.config_manager:
                logger.error("No ConfigManager available, cannot load from config")
                QMessageBox.warning(
                    self.view,
                    "Load Failed",
                    "Configuration manager not available"
                )
                return
            
            # Get calibration data from config
            calibration_data = self.config_manager.get_calibration_points()
            
            if not calibration_data or not calibration_data.get("left") or not calibration_data.get("right"):
                logger.warning("No calibration points found in configuration")
                QMessageBox.warning(
                    self.view,
                    "Load Failed",
                    "No calibration points found in configuration"
                )
                return
            
            # Get the image dimensions from view's scenes
            left_scene = self.view.left_scene
            right_scene = self.view.right_scene
            
            left_width = left_scene.width()
            left_height = left_scene.height()
            right_width = right_scene.width()
            right_height = right_scene.height()
            
            # Update model with current image dimensions
            self.model.set_image_dimensions('left', left_width, left_height)
            self.model.set_image_dimensions('right', right_width, right_height)
            
            # Clear existing points
            self.model.clear_points()
            self.view.clear_points()
            
            # Load normalized data into model
            self.model.from_normalized_dict(calibration_data)
            
            # Update view with loaded points
            self._render_loaded_points()
            
            # Show success message
            left_point_count = len(self.model.get_points('left'))
            right_point_count = len(self.model.get_points('right'))
            
            QMessageBox.information(
                self.view,
                "Load Successful",
                f"Loaded {left_point_count} left points and {right_point_count} right points from configuration"
            )
            
            logger.info(f"Loaded {left_point_count} left points and {right_point_count} right points from config")
        except Exception as e:
            logger.error(f"Error loading calibration points from config: {e}")
            QMessageBox.critical(
                self.view,
                "Load Error",
                f"An error occurred while loading from config: {str(e)}"
            )
    
    @Slot()
    def on_save(self):
        """
        Handle saving calibration points to a JSON file.
        Opens a file dialog for the user to choose the save location.
        """
        try:
            # Ensure default save directory exists
            self.default_save_dir.mkdir(parents=True, exist_ok=True)
            
            # Open file dialog
            file_path, _ = QFileDialog.getSaveFileName(
                self.view,
                "Save Calibration Configuration",
                str(self.default_save_dir / "calibration.json"),
                "JSON Files (*.json)"
            )
            
            if not file_path:
                logger.info("Save operation canceled by user")
                return
                
            # Save calibration data
            success = self.model.save_to_file(file_path)
            
            if success:
                # Show success message
                QMessageBox.information(
                    self.view,
                    "Save Successful",
                    f"Calibration data saved to:\n{file_path}"
                )
            else:
                # Show error message
                QMessageBox.warning(
                    self.view,
                    "Save Failed",
                    "Failed to save calibration data. See log for details."
                )
                
        except Exception as e:
            logger.error(f"Error saving calibration points: {e}")
            QMessageBox.critical(
                self.view,
                "Save Error",
                f"An error occurred while saving: {str(e)}"
            )
    
    @Slot()
    def on_load(self):
        """
        Handle loading calibration points from a file.
        Opens a file dialog for the user to choose the file to load.
        """
        try:
            # Open file dialog
            file_path, _ = QFileDialog.getOpenFileName(
                self.view,
                "Load Calibration Configuration",
                str(self.default_save_dir),
                "JSON Files (*.json)"
            )
            
            if not file_path:
                logger.info("Load operation canceled by user")
                return
                
            # Clear existing points first
            self.model.clear_points()
            self.view.clear_points()
            
            # Load calibration data
            success = self.model.load_from_file(file_path)
            
            if success:
                # Update view with loaded points
                self._render_loaded_points()
                
                # Show success message
                QMessageBox.information(
                    self.view,
                    "Load Successful",
                    f"Calibration data loaded from:\n{file_path}"
                )
            else:
                # Show error message
                QMessageBox.warning(
                    self.view,
                    "Load Failed",
                    "Failed to load calibration data. See log for details."
                )
                
        except Exception as e:
            logger.error(f"Error loading calibration points: {e}")
            QMessageBox.critical(
                self.view,
                "Load Error",
                f"An error occurred while loading: {str(e)}"
            )
            
    def _load_points_from_config(self):
        """
        Load calibration points from the configuration manager.
        Called during initialization if config_manager is available.
        """
        try:
            if not self.config_manager:
                return
                
            # Get calibration data from config
            calibration_data = self.config_manager.get_calibration_points()
            
            if not calibration_data or not calibration_data.get("left") or not calibration_data.get("right"):
                logger.debug("No calibration points found in configuration during initialization")
                return
                
            # We'll defer loading the points until we have a valid scene with dimensions
            # This will be triggered when the view is shown and images are loaded
            logger.info("Calibration data found in config, will load when view is ready")
            
        except Exception as e:
            logger.error(f"Error loading calibration points from config during initialization: {e}")
    
    def _render_loaded_points(self):
        """
        Render points loaded from file in the view.
        Called after loading points from a file.
        """
        # Render left points
        left_points = self.model.get_points('left')
        for index, (x, y) in enumerate(left_points):
            self.view.add_point_item('left', x, y, index)
            
        # Render right points
        right_points = self.model.get_points('right')
        for index, (x, y) in enumerate(right_points):
            self.view.add_point_item('right', x, y, index)
            
        # Update grid lines
        if left_points:
            self._update_grid_lines('left')
        if right_points:
            self._update_grid_lines('right')
            
        logger.info(f"Rendered {len(left_points)} left points and {len(right_points)} right points")
    
    def _update_grid_lines(self, side: str):
        """
        Update grid lines for the specified side.
        
        Args:
            side (str): 'left' or 'right'
        """
        points = self.model.get_points(side)
        
        # Only draw grid lines if we have enough points
        if len(points) < 4:
            return
        
        # Determine grid dimensions (assume square grid for now)
        # We'll refine this in later weeks
        grid_size = int(len(points) ** 0.5)
        rows = grid_size
        cols = grid_size
        
        # Draw grid lines
        self.view.draw_grid_lines(side, points, rows, cols)
    
    def set_images(self, left_image, right_image):
        """
        Set the images for the calibration view.
        
        Args:
            left_image: QPixmap or QImage for the left view
            right_image: QPixmap or QImage for the right view
        """
        self.view.set_images(left_image, right_image)
        
        # After setting images, try to load points from config if available
        if self.config_manager:
            # Get the image dimensions from view's scenes
            left_scene = self.view.left_scene
            right_scene = self.view.right_scene
            
            left_width = left_scene.width()
            left_height = left_scene.height()
            right_width = right_scene.width()
            right_height = right_scene.height()
            
            # Update model with current image dimensions
            self.model.set_image_dimensions('left', left_width, left_height)
            self.model.set_image_dimensions('right', right_width, right_height)
            
            # Load calibration data from config
            calibration_data = self.config_manager.get_calibration_points()
            
            if calibration_data and (calibration_data.get("left") or calibration_data.get("right")):
                # Load normalized data into model
                self.model.from_normalized_dict(calibration_data)
                
                # Update view with loaded points
                self._render_loaded_points()
                
                logger.info("Loaded calibration points from config after setting images")
        
    def set_stereo_image_model(self, stereo_image_model):
        """
        Set the stereo image model reference.
        
        Args:
            stereo_image_model: StereoImageModel instance
        """
        self.stereo_image_model = stereo_image_model
    
    def set_config_manager(self, config_manager: ConfigManager):
        """
        Set the configuration manager reference.
        
        Args:
            config_manager (ConfigManager): ConfigManager instance
        """
        self.config_manager = config_manager
    
    @Slot()
    def on_load_current_frame(self):
        """
        Handle loading the current frame from the stereo image model.
        This loads the current left and right images from the player into the calibration view.
        """
        try:
            if not self.stereo_image_model:
                logger.warning("No stereo image model available")
                QMessageBox.warning(
                    self.view,
                    "Load Frame Failed",
                    "No stereo image model available."
                )
                return
                
            # Get current frame from stereo image model
            current_frame = self.stereo_image_model.get_current_frame()
            if not current_frame:
                logger.warning("No current frame available")
                QMessageBox.warning(
                    self.view,
                    "Load Frame Failed",
                    "No current frame available."
                )
                return
                
            # Get left and right images from the frame
            left_img = current_frame.get_left_image()
            right_img = current_frame.get_right_image()
            
            if left_img is None or right_img is None:
                logger.warning("Failed to get images from current frame")
                QMessageBox.warning(
                    self.view,
                    "Load Frame Failed",
                    "Failed to get images from current frame."
                )
                return
                
            # Convert OpenCV images (BGR) to RGB for Qt
            left_img_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            right_img_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
            
            # Convert to QImage
            height, width, channel = left_img_rgb.shape
            bytes_per_line = 3 * width
            
            left_qimg = QImage(left_img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            right_qimg = QImage(right_img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # Convert to QPixmap
            left_pixmap = QPixmap.fromImage(left_qimg)
            right_pixmap = QPixmap.fromImage(right_qimg)
            
            # Set images in the view
            self.view.set_images(left_pixmap, right_pixmap)
            
            logger.info("Loaded current frame into calibration view")
            
        except Exception as e:
            logger.error(f"Error loading current frame: {e}")
            QMessageBox.critical(
                self.view,
                "Load Frame Error",
                f"An error occurred while loading the current frame: {str(e)}"
            ) 