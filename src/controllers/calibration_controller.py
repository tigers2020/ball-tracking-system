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
from src.utils.geometry import pixel_to_scene, scene_to_pixel
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
        
        # Display scale for coordinate transformation
        self.display_scale = 1.0
        if self.config_manager:
            camera_settings = self.config_manager.get_camera_settings()
            if camera_settings and 'resizing_scale' in camera_settings:
                self.display_scale = camera_settings['resizing_scale']
        
        # Offset values for coordinate transformations (will be updated if needed)
        self.left_offset_y = 0
        self.right_offset_y = 0
        
        # Try to load calibration points from config if available
        if self.config_manager:
            self._load_points_from_config()
    
    @Slot(str, float, float)
    def on_add_point(self, side, scene_x, scene_y):
        """
        Add a point to the scene and update the model
        
        Args:
            side (str): 'left' or 'right' to indicate which view
            scene_x (float): x coordinate in scene coordinates
            scene_y (float): y coordinate in scene coordinates
        """
        if side not in ['left', 'right']:
            logger.error(f"Invalid side specified: {side}")
            return
            
        width_scale, height_scale = self._get_view_scale(side)
        if width_scale == 0 or height_scale == 0:
            logger.error(f"Invalid scale calculated for {side} view: ({width_scale}, {height_scale})")
            return

        # Convert scene coordinates to model coordinates (pixels)
        pixel_x = scene_x / width_scale
        pixel_y = scene_y / height_scale
        
        logger.info(f"Adding point to {side} at scene({scene_x:.1f}, {scene_y:.1f}), "
                   f"pixel({pixel_x:.1f}, {pixel_y:.1f})")
        
        # Add to model
        point_id = self.model.add_point(side, (pixel_x, pixel_y))
        
        # Add to view - view works with scene coordinates
        self.view.add_point(side, point_id, scene_x, scene_y)
        
        # Update grid lines after adding a point
        self._update_grid_lines(side)
    
    @Slot(str, int, float, float)
    def on_move_point(self, side, point_id, scene_x, scene_y):
        """
        Handle moving a point
        
        Args:
            side (str): 'left' or 'right' to indicate which view
            point_id (int): The ID of the point to move
            scene_x (float): New x coordinate in scene coordinates
            scene_y (float): New y coordinate in scene coordinates
        """
        if side not in ['left', 'right']:
            logger.error(f"Invalid side specified: {side}")
            return
            
        width_scale, height_scale = self._get_view_scale(side)
        if width_scale == 0 or height_scale == 0:
            logger.error(f"Invalid scale calculated for {side} view: ({width_scale}, {height_scale})")
            return
        
        # Convert scene coordinates to model coordinates (pixels)
        pixel_x = scene_x / width_scale
        pixel_y = scene_y / height_scale
        
        logger.info(f"Moving point {point_id} in {side} to scene({scene_x:.1f}, {scene_y:.1f}), "
                   f"pixel({pixel_x:.1f}, {pixel_y:.1f})")
        
        # Update in model
        self.model.update_point(side, point_id, (pixel_x, pixel_y))
        
        # Update in view (view uses scene coordinates)
        self.view.update_point(side, point_id, scene_x, scene_y)
        
        # Update grid lines after moving a point
        self._update_grid_lines(side)
    
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
        
        # Get image dimensions
        img_height, img_width = image.shape[:2]
        
        # Get scale factors for this view
        w_scale, h_scale = self._get_view_scale(side)
        
        # Calculate dynamic ROI radius based on image size (approximately 2.5% of image width)
        roi_radius = max(int(img_width * 0.025), 15)  # Min 15 pixels
        logger.info(f"Using dynamic ROI radius of {roi_radius} pixels for {side} image ({img_width}x{img_height})")
        
        # Loop through each point
        for index, point in enumerate(points):
            try:
                # Show ROI overlay to indicate processing
                # Convert model point to scene coordinates for display
                scene_x = point[0] * w_scale
                scene_y = point[1] * h_scale
                self.view.show_roi(side, (scene_x, scene_y), roi_radius * w_scale)  # Scale ROI radius too
                
                # Extract ROI around the point (using model coordinates - original pixels)
                roi = crop_roi(image, point, radius=roi_radius)
                
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
                    roi_with_padding, (offset_x, offset_y) = crop_roi_with_padding(image, point, radius=roi_radius)
                    
                    # Adjust intersection coordinates to image coordinates (original pixel space)
                    adjusted_x = best_x + offset_x
                    adjusted_y = best_y + offset_y
                    
                    # Calculate adjustment distance for logging
                    adjustment_distance = ((adjusted_x - point[0])**2 + (adjusted_y - point[1])**2)**0.5
                    logger.info(f"Fine-tuned {side} point {index} from {point} to ({adjusted_x:.1f}, {adjusted_y:.1f}), "
                               f"adjustment distance: {adjustment_distance:.2f} pixels")
                    
                    # Update the point in the model (store original pixel coordinates)
                    self.model.update_point(side, index, (adjusted_x, adjusted_y))
                    
                    # Convert pixel coordinates to scene coordinates for display
                    # Get the latest scale factors to ensure accuracy
                    # Important: we need to recalculate scale after each point update
                    # to ensure we're using the most accurate values
                    current_w_scale, current_h_scale = self._get_view_scale(side)
                    scene_x = adjusted_x * current_w_scale
                    scene_y = adjusted_y * current_h_scale
                    
                    logger.debug(f"Converted pixel ({adjusted_x:.1f}, {adjusted_y:.1f}) to scene ({scene_x:.1f}, {scene_y:.1f}) "
                                f"using scale factors ({current_w_scale:.3f}, {current_h_scale:.3f})")
                    
                    # Update the point position in the view (using scene coordinates)
                    self.view.update_point(side, index, scene_x, scene_y)
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
        # Get image dimensions
        left_w_scale, left_h_scale = self._get_view_scale('left')
        right_w_scale, right_h_scale = self._get_view_scale('right')
        
        # Render left points
        left_points = self.model.get_points('left')
        for index, (x, y) in enumerate(left_points):
            # Scale original pixel coordinates to scene coordinates using the correct scale
            scene_x = x * left_w_scale
            scene_y = y * left_h_scale
            
            self.view.add_point('left', index, scene_x, scene_y)
            
        # Render right points
        right_points = self.model.get_points('right')
        for index, (x, y) in enumerate(right_points):
            # Scale original pixel coordinates to scene coordinates using the correct scale
            scene_x = x * right_w_scale
            scene_y = y * right_h_scale
            
            self.view.add_point('right', index, scene_x, scene_y)
            
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
            # Get the image dimensions from the actual pixmaps, not the scenes
            left_width = self.view.left_pixmap.width()
            left_height = self.view.left_pixmap.height()
            right_width = self.view.right_pixmap.width()
            right_height = self.view.right_pixmap.height()
            
            logger.info(f"Setting model image dimensions to Left: {left_width}x{left_height}, Right: {right_width}x{right_height}")
            
            # Update model with actual image dimensions
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
        
        # Update display scale from config
        if self.config_manager:
            camera_settings = self.config_manager.get_camera_settings()
            if camera_settings and 'resizing_scale' in camera_settings:
                self.display_scale = camera_settings['resizing_scale']
                logger.info(f"Updated display scale to {self.display_scale} from config")
    
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
    
    def _get_view_scale(self, side):
        """
        Calculate the scaling factor between model coordinates (pixels) and view coordinates (scene).
        
        Args:
            side (str): 'left' or 'right' to indicate which view
        
        Returns:
            tuple: (width_scale, height_scale) - scale factors for width and height
        """
        if side == 'left':
            scene = self.view.left_scene
            pixmap = self.view.left_pixmap
        else:
            scene = self.view.right_scene
            pixmap = self.view.right_pixmap
        
        if pixmap is None or pixmap.width() == 0 or pixmap.height() == 0:
            logger.warning(f"Cannot calculate scale for {side} view: pixmap is not available or has zero dimensions")
            return 1.0, 1.0
        
        # Calculate scale factors
        width_scale = scene.width() / pixmap.width()
        height_scale = scene.height() / pixmap.height()
        
        logger.debug(f"{side} view scale: pixmap({pixmap.width()}x{pixmap.height()}), "
                    f"scene({scene.width():.1f}x{scene.height():.1f}), " 
                    f"scale({width_scale:.3f}, {height_scale:.3f})")
        return width_scale, height_scale 