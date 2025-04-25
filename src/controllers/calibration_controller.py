#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calibration Controller module.
This module contains the CalibrationController class which connects the model and view.
"""

import logging
import os
from typing import Optional

from PySide6.QtCore import QObject, Slot, QPointF
from PySide6.QtGui import QPixmap

from src.models.calibration_model import CalibrationModel
from src.views.calibration_view import CalibrationView
from src.utils.ui_constants import CalibrationTab


class CalibrationController(QObject):
    """
    Controller for the calibration feature.
    Connects the CalibrationModel and CalibrationView.
    """
    
    def __init__(self, model: CalibrationModel, view: CalibrationView):
        """
        Initialize the calibration controller.
        
        Args:
            model (CalibrationModel): Calibration model
            view (CalibrationView): Calibration view
        """
        super(CalibrationController, self).__init__()
        
        self.model = model
        self.view = view
        
        # Set controller reference in view for direct model access
        self.view.controller = self
        
        # Connect view signals to controller slots
        self.view.point_added.connect(self.add_point)
        self.view.point_moved.connect(self.update_point)
        self.view.fine_tune_requested.connect(self.fine_tune)
        self.view.save_calibration_requested.connect(self.save_calibration)
        self.view.load_calibration_requested.connect(self.load_calibration)
        self.view.clear_points_requested.connect(self.clear_points)
        self.view.load_images_requested.connect(self.load_images)
        self.view.load_current_frame_requested.connect(self.load_current_frame)
        
        # Connect model signals to view update methods
        self.model.points_changed.connect(self.update_view)
        self.model.point_updated.connect(self.view.update_point)
        
        # Reference to main window or frame provider (to be set by main application)
        self.main_window = None
        
        # Flag to indicate whether we should use the config manager (preferred) or direct file I/O
        self.use_config_manager = True
        
        # Current active point index for each side
        self.active_index = {"left": 0, "right": 0}
        
        # Maximum number of points allowed per side
        self.max_points = CalibrationTab.MAX_POINTS
    
    @Slot(str, QPointF)
    def add_point(self, side: str, position: QPointF) -> None:
        """
        Add a calibration point.
        
        Args:
            side (str): 'left' or 'right'
            position (QPointF): Point position
        """
        points = self.model.left_points if side == "left" else self.model.right_points
        
        if len(points) < self.max_points:
            # Add new point if we haven't reached the maximum
            self.model.add_point(side, position)
            # Update active index to the newly added point
            self.active_index[side] = len(points) - 1
            logging.info(f"Added new {side} point at index {self.active_index[side]}")
        else:
            # Maximum reached, overwrite the current active point
            index = self.active_index[side]
            logging.info(f"Maximum points ({self.max_points}) reached, overwriting {side} point at index {index}")
            self.model.update_point(side, index, position)
            # Increment active index for next overwrite
            self.active_index[side] = (index + 1) % self.max_points
    
    @Slot(str, int, QPointF)
    def update_point(self, side: str, index: int, position: QPointF) -> None:
        """
        Update a calibration point.
        
        Args:
            side (str): 'left' or 'right'
            index (int): Point index
            position (QPointF): New position
        """
        self.model.update_point(side, index, position)
    
    @Slot()
    def fine_tune(self) -> None:
        """
        Fine-tune calibration points.
        This is a placeholder that will be implemented in future versions.
        """
        # This will be implemented in the future with the ROI cropper, skeletonizer, etc.
        logging.info("Fine-tune requested - feature not implemented yet")
        
        # Show a message in the view
        # This would be better with a status message system
        pass
    
    @Slot(str)
    def save_calibration(self, file_path: str) -> None:
        """
        Save calibration to file.
        
        Args:
            file_path (str): File path
        """
        if self.use_config_manager and self.model.config_manager:
            # Save using config manager
            if self.model.save_to_config():
                logging.info("Calibration saved to config")
            else:
                logging.error("Failed to save calibration to config")
        else:
            # Fall back to direct file saving if no config manager
            if self.model.save_to_json(file_path):
                logging.info(f"Calibration saved to {file_path}")
            else:
                logging.error(f"Failed to save calibration to {file_path}")
    
    @Slot(str)
    def load_calibration(self, file_path: str) -> None:
        """
        Load calibration from file.
        
        Args:
            file_path (str): File path
        """
        success = False
        
        if self.use_config_manager and self.model.config_manager:
            # Load using config manager
            success = self.model.load_from_config()
            if success:
                logging.info("Calibration loaded from config")
            else:
                logging.error("Failed to load calibration from config")
        else:
            # Fall back to direct file loading if no config manager
            success = self.model.load_from_json(file_path)
            if success:
                logging.info(f"Calibration loaded from {file_path}")
            else:
                logging.error(f"Failed to load calibration from {file_path}")
                
        if success:
            # Try to load images if paths are available and files exist
            if self.model.left_image_path and os.path.exists(self.model.left_image_path):
                try:
                    left_pixmap = QPixmap(self.model.left_image_path)
                    if not left_pixmap.isNull():
                        self.view.set_left_image(left_pixmap)
                        logging.info(f"Successfully loaded left image from: {self.model.left_image_path}")
                    else:
                        logging.warning(f"Failed to create pixmap from left image path: {self.model.left_image_path}")
                except Exception as e:
                    logging.error(f"Error loading left image: {str(e)}")
                
            if self.model.right_image_path and os.path.exists(self.model.right_image_path):
                try:
                    right_pixmap = QPixmap(self.model.right_image_path)
                    if not right_pixmap.isNull():
                        self.view.set_right_image(right_pixmap)
                        logging.info(f"Successfully loaded right image from: {self.model.right_image_path}")
                    else:
                        logging.warning(f"Failed to create pixmap from right image path: {self.model.right_image_path}")
                except Exception as e:
                    logging.error(f"Error loading right image: {str(e)}")
                
            # Update view with points even if image loading failed
            self.update_view("left")
            self.update_view("right")
    
    @Slot()
    def clear_points(self) -> None:
        """Clear all calibration points."""
        self.model.clear_points()
        self.view.clear_points()
    
    @Slot(str, str)
    def load_images(self, left_path: str, right_path: str) -> None:
        """
        Load images.
        
        Args:
            left_path (str): Path to left image
            right_path (str): Path to right image
        """
        # Check if files exist
        if not os.path.isfile(left_path) or not os.path.isfile(right_path):
            logging.error("One or both image files do not exist")
            return
        
        # Set paths in model
        self.model.set_image_paths(left_path, right_path)
        
        # Load images to view
        left_pixmap = QPixmap(left_path)
        right_pixmap = QPixmap(right_path)
        
        if left_pixmap.isNull() or right_pixmap.isNull():
            logging.error("Failed to load one or both images")
            return
        
        # Set image sizes in model for normalized coordinates
        self.model.set_image_sizes(
            (left_pixmap.width(), left_pixmap.height()),
            (right_pixmap.width(), right_pixmap.height())
        )
        
        self.view.set_left_image(left_pixmap)
        self.view.set_right_image(right_pixmap)
        
        logging.info(f"Images loaded: {left_path}, {right_path}")
    
    @Slot(str)
    def update_view(self, side: str) -> None:
        """
        Update view when model changes.
        
        Args:
            side (str): 'left' or 'right' side that changed
        """
        # Call view's rebuild points method to update points from model data
        if side == "left":
            self.view._rebuild_points("left")
        else:
            self.view._rebuild_points("right")
    
    @Slot()
    def load_current_frame(self) -> None:
        """
        Load the current frame images from the main application.
        This method will be triggered when the user clicks the Load Current Frame button.
        """
        # Check if we have a reference to the main window
        if not self.main_window:
            logging.error("Cannot load current frame: No reference to main window")
            return
            
        # Get current frame images from main window
        # This assumes the main window has a method to get the current frame images
        # The implementation might need to be adjusted based on the actual main window structure
        try:
            # Get current pixmaps from main window (to be implemented)
            left_pixmap, right_pixmap = self.main_window.get_current_frame_pixmaps()
            
            if left_pixmap is None or right_pixmap is None:
                logging.error("Failed to get current frame images")
                return
            
            # Set image sizes in model for normalized coordinates
            self.model.set_image_sizes(
                (left_pixmap.width(), left_pixmap.height()),
                (right_pixmap.width(), right_pixmap.height())
            )
                
            # Set the images in the view
            self.view.set_left_image(left_pixmap)
            self.view.set_right_image(right_pixmap)
            
            # Set image paths to None or to actual paths if available
            frame_info = self.main_window.get_current_frame_info()
            if frame_info:
                self.model.set_image_paths(
                    frame_info.get('left_path', None),
                    frame_info.get('right_path', None)
                )
            
            logging.info("Current frame images loaded")
        except Exception as e:
            logging.error(f"Error loading current frame images: {str(e)}") 