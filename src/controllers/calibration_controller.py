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
import cv2
from PySide6.QtGui import QImage, QPixmap

logger = logging.getLogger(__name__)


class CalibrationController(QObject):
    """
    Controller for the calibration functionality.
    Connects the calibration model and view.
    """
    
    def __init__(self, model: CalibrationModel, view: CalibrationTab, config_manager=None):
        """
        Initialize the calibration controller.
        
        Args:
            model (CalibrationModel): The calibration model
            view (CalibrationTab): The calibration view
            config_manager (ConfigManager, optional): Configuration manager instance
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
        self.view.save_button.clicked.connect(self.on_save)
        self.view.load_button.clicked.connect(self.on_load)
        self.view.load_current_frame_button.clicked.connect(self.on_load_current_frame)
        
        # Default calibration file directory
        self.default_save_dir = Path.home() / "Court_Calibration"
        
        # Reference to the stereo image model (will be set from app_controller)
        self.stereo_image_model = None
        
        # Fixed number of calibration points
        self.num_points = 14
        
        # Point IDs (p00, p01, etc.)
        self.point_ids = [f"p{i:02d}" for i in range(self.num_points)]
    
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
            # Get image dimensions for normalization
            scene_rect = self.view.left_scene.sceneRect() if side == 'left' else self.view.right_scene.sceneRect()
            width = scene_rect.width()
            height = scene_rect.height()
            
            # Normalize coordinates (0-1 range)
            norm_x = x / width if width > 0 else 0
            norm_y = y / height if height > 0 else 0
            
            # Determine which point to update (based on how many we have)
            points = self.model.get_points(side)
            index = min(len(points), self.num_points - 1)
            point_id = self.point_ids[index]
            
            # If we already have max points, replace the last one
            if len(points) >= self.num_points:
                # Remove the last point from view
                if side == 'left' and index in self.view.left_points:
                    self.view.left_scene.removeItem(self.view.left_points[index])
                    del self.view.left_points[index]
                elif side == 'right' and index in self.view.right_points:
                    self.view.right_scene.removeItem(self.view.right_points[index])
                    del self.view.right_points[index]
                
                # Update the point in the model with normalized coordinates
                self.model.update_point(side, index, (norm_x, norm_y))
            else:
                # Add new point to model with normalized coordinates
                self.model.add_point(side, (norm_x, norm_y))
            
            # Add point item to view (with scene coordinates)
            point_item = self.view.add_point_item(side, x, y, index)
            if point_item:
                point_item.point_id = point_id
            
            # Update grid lines if we have enough points
            self._update_grid_lines(side)
            
            logger.info(f"Added/updated point {point_id} at ({norm_x:.4f}, {norm_y:.4f}) to {side} view")
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
            # Get image dimensions for normalization
            scene_rect = self.view.left_scene.sceneRect() if side == 'left' else self.view.right_scene.sceneRect()
            width = scene_rect.width()
            height = scene_rect.height()
            
            # Normalize coordinates (0-1 range)
            norm_x = x / width if width > 0 else 0
            norm_y = y / height if height > 0 else 0
            
            # Update point in model with normalized coordinates
            self.model.update_point(side, index, (norm_x, norm_y))
            
            # Update grid lines
            self._update_grid_lines(side)
            
            logger.info(f"Moved point {index} to ({norm_x:.4f}, {norm_y:.4f}) in {side} view")
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
        Handle fine-tuning points.
        This is a placeholder for Week 3 implementation.
        """
        logger.info("Fine-tune functionality will be implemented in Week 3")
        # To be implemented in Week 3
    
    @Slot()
    def on_save(self):
        """
        Handle saving calibration points to config.json.
        Uses the ConfigManager if available, otherwise falls back to file dialog.
        """
        try:
            # Get normalized points from model
            left_points = self.model.get_points('left')
            right_points = self.model.get_points('right')
            
            # Get image sizes
            left_size = {
                "width": int(self.view.left_scene.sceneRect().width()),
                "height": int(self.view.left_scene.sceneRect().height())
            }
            right_size = {
                "width": int(self.view.right_scene.sceneRect().width()),
                "height": int(self.view.right_scene.sceneRect().height())
            }
            
            # Prepare calibration data in config.json format
            calibration_data = {
                "left": {},
                "right": {},
                "left_image_size": left_size,
                "right_image_size": right_size,
                "left_image_path": None,  # These could be set if needed
                "right_image_path": None,
                "calib_ver": 1.2
            }
            
            # Convert points to the format used in config.json
            for i, (x, y) in enumerate(left_points):
                if i < self.num_points:
                    point_id = self.point_ids[i]
                    calibration_data["left"][point_id] = {
                        "x": x,
                        "y": y,
                        "is_fine_tuned": False
                    }
            
            for i, (x, y) in enumerate(right_points):
                if i < self.num_points:
                    point_id = self.point_ids[i]
                    calibration_data["right"][point_id] = {
                        "x": x,
                        "y": y,
                        "is_fine_tuned": False
                    }
            
            # If we have a config manager, use it to save the data
            if self.config_manager:
                # Update the configuration
                self.config_manager.set("calibration_points", calibration_data)
                self.config_manager.save_config(force=True)
                
                # Show success message
                QMessageBox.information(
                    self.view,
                    "Save Successful",
                    "Calibration data saved to configuration."
                )
                logger.info("Calibration data saved to configuration")
                return
                
            # Fallback to file dialog if no config_manager
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
        Handle loading calibration points from config.json.
        Uses the ConfigManager if available, otherwise falls back to file dialog.
        """
        try:
            # Clear existing points first
            self.model.clear_points()
            self.view.clear_points()
            
            # If we have a config manager, use it to load the data
            if self.config_manager:
                # Get the calibration data
                calib_data = self.config_manager.get("calibration_points")
                
                if not calib_data:
                    logger.warning("No calibration data found in configuration")
                    QMessageBox.warning(
                        self.view,
                        "Load Failed",
                        "No calibration data found in configuration."
                    )
                    return
                
                # Load points from config format
                self._load_points_from_config(calib_data)
                
                # Update view with loaded points
                self._render_loaded_points()
                
                # Show success message
                QMessageBox.information(
                    self.view,
                    "Load Successful",
                    "Calibration data loaded from configuration."
                )
                logger.info("Calibration data loaded from configuration")
                return
                
            # Fallback to file dialog if no config_manager
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
    
    def _load_points_from_config(self, calib_data):
        """
        Load calibration points from config data format.
        
        Args:
            calib_data (dict): Calibration data from config
        """
        # Load left points
        if "left" in calib_data:
            left_points = []
            # Sort by point IDs to maintain order
            for point_id in sorted(calib_data["left"].keys()):
                point = calib_data["left"][point_id]
                left_points.append((point["x"], point["y"]))
            
            # Add points to model
            for x, y in left_points:
                self.model.add_point("left", (x, y))
                
        # Load right points
        if "right" in calib_data:
            right_points = []
            # Sort by point IDs to maintain order
            for point_id in sorted(calib_data["right"].keys()):
                point = calib_data["right"][point_id]
                right_points.append((point["x"], point["y"]))
            
            # Add points to model
            for x, y in right_points:
                self.model.add_point("right", (x, y))
    
    def _render_loaded_points(self):
        """
        Render points loaded from file in the view.
        Called after loading points from a file.
        Converts normalized coordinates back to scene coordinates.
        """
        # Get scene dimensions
        left_width = self.view.left_scene.sceneRect().width()
        left_height = self.view.left_scene.sceneRect().height()
        right_width = self.view.right_scene.sceneRect().width()
        right_height = self.view.right_scene.sceneRect().height()
        
        # Render left points
        left_points = self.model.get_points('left')
        for index, (norm_x, norm_y) in enumerate(left_points):
            # Convert normalized coordinates to scene coordinates
            scene_x = norm_x * left_width
            scene_y = norm_y * left_height
            
            # Add point to view
            point_id = self.point_ids[index] if index < len(self.point_ids) else f"p{index:02d}"
            point_item = self.view.add_point_item('left', scene_x, scene_y, index)
            if point_item:
                point_item.point_id = point_id
            
        # Render right points
        right_points = self.model.get_points('right')
        for index, (norm_x, norm_y) in enumerate(right_points):
            # Convert normalized coordinates to scene coordinates
            scene_x = norm_x * right_width
            scene_y = norm_y * right_height
            
            # Add point to view
            point_id = self.point_ids[index] if index < len(self.point_ids) else f"p{index:02d}"
            point_item = self.view.add_point_item('right', scene_x, scene_y, index)
            if point_item:
                point_item.point_id = point_id
            
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
        
        # Get scene dimensions for denormalization
        if side == 'left':
            width = self.view.left_scene.sceneRect().width()
            height = self.view.left_scene.sceneRect().height()
        else:
            width = self.view.right_scene.sceneRect().width()
            height = self.view.right_scene.sceneRect().height()
        
        # Denormalize coordinates for display
        scene_points = [(x * width, y * height) for x, y in points]
        
        # Determine grid dimensions (assume square grid for now)
        # We'll refine this in later weeks
        grid_size = int(len(points) ** 0.5)
        rows = grid_size
        cols = grid_size
        
        # Draw grid lines
        self.view.draw_grid_lines(side, scene_points, rows, cols)
    
    def set_images(self, left_image, right_image):
        """
        Set the images for the calibration view.
        
        Args:
            left_image: QPixmap or QImage for the left view
            right_image: QPixmap or QImage for the right view
        """
        self.view.set_images(left_image, right_image)
        
        # Re-render points to update positions
        if (self.model.get_points('left') or self.model.get_points('right')):
            self._render_loaded_points()
        
    def set_stereo_image_model(self, stereo_image_model):
        """
        Set the stereo image model reference.
        
        Args:
            stereo_image_model: StereoImageModel instance
        """
        self.stereo_image_model = stereo_image_model
    
    def set_config_manager(self, config_manager):
        """
        Set the configuration manager reference.
        
        Args:
            config_manager: ConfigManager instance
        """
        self.config_manager = config_manager
        
        # If we have config_manager, load calibration points if available
        if self.config_manager:
            calib_data = self.config_manager.get("calibration_points")
            if calib_data:
                self.model.clear_points()
                self.view.clear_points()
                self._load_points_from_config(calib_data)
                self._render_loaded_points()
    
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
            self.set_images(left_pixmap, right_pixmap)
            
            logger.info("Loaded current frame into calibration view")
            
        except Exception as e:
            logger.error(f"Error loading current frame: {e}")
            QMessageBox.critical(
                self.view,
                "Load Frame Error",
                f"An error occurred while loading the current frame: {str(e)}"
            ) 