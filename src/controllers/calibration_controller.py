#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calibration Controller module.
This module contains the CalibrationController class which connects the calibration model and view.
"""

import logging
import os
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path

from PySide6.QtCore import QObject, Slot, QThread
from PySide6.QtWidgets import QFileDialog, QMessageBox, QApplication
from PySide6.QtGui import QPen, QColor

from src.models.calibration_model import CalibrationModel
from src.models.stereo_image_model import StereoImageModel
from src.models.config_model import CalibrationConfig
from src.views.calibration_view import CalibrationView
from src.utils.config_manager import ConfigManager
from src.utils.geometry import pixel_to_scene, scene_to_pixel, get_scale_factors
import cv2
from PySide6.QtGui import QImage, QPixmap

# Import new services
from src.services.roi_cropper import crop_roi, crop_roi_with_padding
from src.services.skeletonizer import skeletonize_roi
from src.services.intersection_finder import find_and_sort_intersections
from src.services.calibration_fine_tuning_service import CalibrationFineTuningService

import numpy as np

logger = logging.getLogger(__name__)


class CalibrationController(QObject):
    """
    Controller for the calibration functionality.
    Connects the calibration model and view.
    """
    
    def __init__(self, 
                model: StereoImageModel, 
                view: CalibrationView, 
                config_manager: Optional[ConfigManager] = None,
                fine_tuning_service: Optional[CalibrationFineTuningService] = None):
        """
        Initialize the calibration controller.
        
        Args:
            model: The stereo image model
            view: The calibration view
            config_manager: Optional configuration manager for loading/saving calibration
            fine_tuning_service: Optional service for fine-tuning calibration points
        """
        super().__init__()
        
        self.model = model
        self.view = view
        self.config_manager = config_manager
        
        # Initialize the fine-tuning service (use provided one or create new)
        self.fine_tuning_service = fine_tuning_service or CalibrationFineTuningService()
        
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
    def on_add_point(self, side: str, scene_x: float, scene_y: float):
        """
        Handle adding a point when the user clicks on one of the scenes.
        
        Args:
            side (str): Which side the point was added on ('left' or 'right')
            scene_x (float): The x-coordinate in the scene
            scene_y (float): The y-coordinate in the scene
        """
        if side not in ['left', 'right']:
            logger.error(f"Invalid side: {side}")
            return
        
        # Convert scene coordinates to model coordinates (pixels) using view's method
        pixel_x, pixel_y = self.view.scene_to_pixel(side, scene_x, scene_y)
        
        logger.info(f"Adding point on {side} side at scene coordinates: "
                   f"({scene_x:.2f}, {scene_y:.2f}), pixel coordinates: ({pixel_x:.2f}, {pixel_y:.2f})")
        
        # Use add_point to add a new point to the model
        point_id = self.model.add_point(side, (pixel_x, pixel_y))
        if point_id < 0:
            logger.error(f"Failed to add point to {side} side")
            return
            
        # Add the point directly to the view
        self.view.add_point(side, point_id, scene_x, scene_y)
        
        # Update grid lines after adding a point
        self._update_grid_lines()
    
    @Slot(str, int, float, float)
    def on_move_point(self, side: str, point_id: int, scene_x: float, scene_y: float):
        """
        Handle moving a point when the user drags it in the scene.
        
        Args:
            side (str): Which side the point was moved on ('left' or 'right')
            point_id (int): The ID of the point being moved
            scene_x (float): The new x-coordinate in the scene
            scene_y (float): The new y-coordinate in scene coordinates
        """
        if side not in ['left', 'right']:
            logger.error(f"Invalid side: {side}")
            return
        
        # Convert scene coordinates to model coordinates (pixels) using view's method
        pixel_x, pixel_y = self.view.scene_to_pixel(side, scene_x, scene_y)
        
        logger.info(f"Moving point {point_id} on {side} side to scene coordinates: "
                  f"({scene_x:.2f}, {scene_y:.2f}), pixel coordinates: ({pixel_x:.2f}, {pixel_y:.2f})")
        
        # Update the point in the model based on side
        self.model.update_point(side, point_id, (pixel_x, pixel_y))
        
        # 포인트 이동 도중에는 view를 다시 업데이트하지 않음
        # CalibrationPoint 객체의 드래그는 사용자 입력에 의해 이미 처리되고 있음
        # 드래그가 완료된 후에만 필요한 경우 그리드 라인을 업데이트
        
        # Update grid lines after moving a point
        self._update_grid_lines()
    
    @Slot()
    def on_clear_points(self):
        """Handle clearing all points."""
        try:
            # Model에서 포인트 제거
            self.model.clear_points()
            
            # View에서 포인트 제거
            self.view.clear_points()
            
            logger.info("Cleared all calibration points")
        except Exception as e:
            logger.error(f"Error clearing points: {e}")
    
    @Slot()
    def on_fine_tune(self):
        """
        Handle fine-tuning calibration points.
        This processes both left and right images to improve the accuracy of calibration points.
        """
        # Get the current images from view
        left_image = self.view.get_left_image()
        right_image = self.view.get_right_image()
        
        if left_image is None or right_image is None:
            logger.error("No images available for fine-tuning")
            QMessageBox.warning(
                self.view,
                "Fine Tuning Failed",
                "No images available for fine-tuning"
            )
            return
        
        # Get current points from model
        left_points = [self.model.get_point('left', i) for i in range(self.model.get_point_count('left'))]
        right_points = [self.model.get_point('right', i) for i in range(self.model.get_point_count('right'))]
        
        # Filter out None values
        left_points = [p for p in left_points if p is not None]
        right_points = [p for p in right_points if p is not None]
        
        if not left_points and not right_points:
            logger.warning("No calibration points available for fine-tuning")
            QMessageBox.warning(
                self.view,
                "Fine Tuning Failed",
                "No calibration points available for fine-tuning"
            )
            return
        
        # Show progress dialog
        progress_dialog = QMessageBox(self.view)
        progress_dialog.setWindowTitle("Fine-Tuning in Progress")
        progress_dialog.setText("Fine-tuning calibration points...\nPlease wait.")
        progress_dialog.setStandardButtons(QMessageBox.NoButton)
        progress_dialog.show()
        QApplication.processEvents()
        
        try:
            # Each point processing code for left and right sides
            # ...

            # Perform actual fine-tuning
            results = self.fine_tuning_service.fine_tune_calibration_points(
                left_image=left_image,
                right_image=right_image,
                left_points=left_points,
                right_points=right_points
            )
            
            # Close progress dialog and refresh UI
            progress_dialog.close()
            QApplication.processEvents()
            
            # Process results and update model/view
            adjusted_count = 0
            
            # Update left points
            for idx, result in results.get('left', {}).items():
                if result.get('success', False):
                    adjusted_x, adjusted_y = result['adjusted']
                    self.model.update_point('left', idx, (adjusted_x, adjusted_y))
                    adjusted_count += 1
                    
            # Update right points  
            for idx, result in results.get('right', {}).items():
                if result.get('success', False):
                    adjusted_x, adjusted_y = result['adjusted']
                    self.model.update_point('right', idx, (adjusted_x, adjusted_y))
                    adjusted_count += 1
                    
            # Hide ROI
            self.view.hide_roi('left')
            self.view.hide_roi('right')
            
            # Update the view
            self._render_points()
            self._update_grid_lines()
            
            logger.info(f"Fine-tuning complete. Adjusted {adjusted_count} points.")
            
            # Show success message
            QMessageBox.information(
                self.view,
                "Fine Tuning Complete",
                f"Fine-tuning complete. Adjusted {adjusted_count} points."
            )
        except Exception as e:
            # Close progress dialog and refresh UI in case of error
            progress_dialog.close()
            QApplication.processEvents()
            
            # Hide ROI
            self.view.hide_roi('left')
            self.view.hide_roi('right')
            
            logger.error(f"Error during fine-tuning: {e}")
            QMessageBox.critical(
                self.view,
                "Fine Tuning Error",
                f"An error occurred during fine-tuning: {str(e)}"
            )
        finally:
            # Ensure dialog is closed in all circumstances
            if progress_dialog.isVisible():
                progress_dialog.close()
                QApplication.processEvents()
    
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
            self._render_points()
            
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
                self._render_points()
                
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
    
    def _render_points(self):
        """Render all points in the view."""
        # Clear existing points from view
        self.view.clear_points()
        
        # Get number of points from model
        left_point_count = self.model.get_point_count('left')
        right_point_count = self.model.get_point_count('right')
        
        # Render left points
        for index in range(left_point_count):
            point = self.model.get_point('left', index)
            if point is not None:
                # Convert pixel coordinates to scene coordinates using view's method
                scene_x, scene_y = self.view.pixel_to_scene('left', point[0], point[1])
                
                # Add point to view
                self.view.add_point('left', index, scene_x, scene_y)
            
        # Render right points
        for index in range(right_point_count):
            point = self.model.get_point('right', index)
            if point is not None:
                # Convert pixel coordinates to scene coordinates using view's method
                scene_x, scene_y = self.view.pixel_to_scene('right', point[0], point[1])
                
                # Add point to view
                self.view.add_point('right', index, scene_x, scene_y)
        
        # Update grid lines
        self._update_grid_lines()
            
        logger.info(f"Rendered {left_point_count} left points and {right_point_count} right points")
    
    def _update_grid_lines(self, side=None):
        """
        Update grid lines for the specified side or both sides if side is None.
        
        Args:
            side (str, optional): 'left', 'right', or None for both sides
        """
        sides = [side] if side else ['left', 'right']
        
        for current_side in sides:
            # Get points for the current side
            if current_side == 'left':
                point_count = self.model.get_point_count('left')
                points = [self.model.get_point('left', i) for i in range(point_count)]
                # Filter out None values
                points = [p for p in points if p is not None]
            else:  # 'right'
                point_count = self.model.get_point_count('right')
                points = [self.model.get_point('right', i) for i in range(point_count)]
                # Filter out None values
                points = [p for p in points if p is not None]
            
            # 각 포인트를 scene 좌표로 변환
            scene_points = []
            for point in points:
                # 점들을 scene 좌표로 변환
                scene_x, scene_y = self.view.pixel_to_scene(current_side, point[0], point[1])
                scene_points.append((scene_x, scene_y))
            
            # Draw grid lines with custom pattern
            # 지정된 4x4 그리드 레이아웃으로 그리기
            self.view.draw_grid_lines(current_side, scene_points, 4, 4)
    
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
                self._render_points()
                
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
    
    def _render_point(self, side: str, point_id: int):
        """
        Render a specific point in the view.
        
        Args:
            side (str): 'left' or 'right'
            point_id (int): The id of the point to render
        """
        try:
            point = self.model.get_point(side, point_id)
            
            if point:
                # Convert pixel coordinates to scene coordinates
                scene_x, scene_y = self.view.pixel_to_scene(side, point[0], point[1])
                
                # Update the point view directly
                self.view.update_point(side, point_id, scene_x, scene_y)
                
                logger.debug(f"Rendered point {point_id} on {side} side at scene({scene_x:.1f}, {scene_y:.1f}), pixel({point[0]:.1f}, {point[1]:.1f})")
        except Exception as e:
            logger.error(f"Failed to render point {point_id} on {side} side: {e}")
    
    def add_grid_line(self, side: str, start_point: tuple, end_point: tuple, color: str = None):
        """
        Add a grid line to the specified side.
        
        Args:
            side (str): 'left' or 'right' side
            start_point (tuple): (x, y) start point in scene coordinates
            end_point (tuple): (x, y) end point in scene coordinates
            color (str, optional): Color of the line, defaults to None which will use the default color
        
        Returns:
            The added line item
        """
        if side not in ['left', 'right']:
            logger.error(f"Invalid side specified for grid line: {side}")
            return None
            
        try:
            # Create a pen for the line
            pen = QPen(QColor(color) if color else QColor(255, 0, 0, 128))  # Semi-transparent red by default
            pen.setWidth(1)
            
            # Create line in the appropriate scene
            if side == 'left':
                line = self.view.left_scene.addLine(
                    start_point[0], start_point[1], 
                    end_point[0], end_point[1], 
                    pen
                )
                self.view.left_grid_lines.append(line)
                return line
            else:  # 'right'
                line = self.view.right_scene.addLine(
                    start_point[0], start_point[1], 
                    end_point[0], end_point[1], 
                    pen
                )
                self.view.right_grid_lines.append(line)
                return line
        except Exception as e:
            logger.error(f"Failed to add grid line on {side} side: {e}")
            return None
            
    def update_grid_lines(self, side: str = None):
        """
        Update grid lines for the specified side or both sides if side is None.
        
        Args:
            side (str, optional): 'left', 'right', or None for both sides
        """
        # First clear existing grid lines
        self.view.update_grid_lines(side)
        
        # If no side specified or invalid side, update both sides
        sides_to_update = [side] if side in ['left', 'right'] else ['left', 'right']
        
        for current_side in sides_to_update:
            # Get calibration points for the current side
            points = self.model.get_points(current_side)
            if not points or len(points) < 2:
                logger.debug(f"Not enough points to create grid lines for {current_side} side")
                continue
            
            try:
                # Create horizontal and vertical grid lines
                for i in range(len(points)):
                    for j in range(i + 1, len(points)):
                        # Skip if points are not part of a grid structure
                        if not self._are_points_in_grid(points[i], points[j]):
                            continue
                            
                        # Convert to scene coordinates
                        start_x, start_y = self.view.pixel_to_scene(current_side, points[i].x, points[i].y)
                        end_x, end_y = self.view.pixel_to_scene(current_side, points[j].x, points[j].y)
                        
                        # Add the grid line
                        self.add_grid_line(current_side, (start_x, start_y), (end_x, end_y))
            except Exception as e:
                logger.error(f"Error updating grid lines for {current_side} side: {e}")
                
    def _are_points_in_grid(self, point1, point2):
        """
        Determine if two points should be connected in a grid structure.
        
        Args:
            point1: First calibration point
            point2: Second calibration point
            
        Returns:
            bool: True if points should be connected, False otherwise
        """
        # Simplified implementation - connect points that are approximately in the same row or column
        # This can be enhanced with more sophisticated grid detection logic
        GRID_TOLERANCE = 10  # pixels tolerance for alignment
        
        # Check if points are approximately in the same row (y-coordinate)
        same_row = abs(point1.y - point2.y) < GRID_TOLERANCE
        
        # Check if points are approximately in the same column (x-coordinate)
        same_column = abs(point1.x - point2.x) < GRID_TOLERANCE
        
        return same_row or same_column 