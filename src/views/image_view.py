#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image View module.
This module contains the ImageView class for the image view tab in the Stereo Image Player.
"""

import logging
from PySide6.QtWidgets import QWidget, QVBoxLayout

from src.views.image_view_widget import StereoImageViewWidget
from src.views.playback_controls_widget import PlaybackControlsWidget
from src.views.info_view import InfoView
from src.utils.ui_constants import Layout


class ImageView(QWidget):
    """
    Widget for the image view tab in the Stereo Image Player.
    Contains the stereo image view and playback controls.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the image view.
        
        Args:
            parent (QWidget, optional): Parent widget
        """
        super(ImageView, self).__init__(parent)
        
        # Set up UI
        self._setup_ui()
        
        # Mask overlay state
        self.show_mask = False
        self.left_mask = None
        self.right_mask = None
    
    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(Layout.MARGIN, Layout.MARGIN, Layout.MARGIN, Layout.MARGIN)
        main_layout.setSpacing(Layout.SPACING)
        
        # Create info view
        self.info_view = InfoView()
        main_layout.addWidget(self.info_view)
        
        # Create stereo image view
        self.stereo_view = StereoImageViewWidget()
        main_layout.addWidget(self.stereo_view)
        
        # Create playback controls
        self.playback_controls = PlaybackControlsWidget()
        main_layout.addWidget(self.playback_controls)
    
    def enable_controls(self, enable=True):
        """
        Enable or disable all controls.
        
        Args:
            enable (bool): True to enable, False to disable
        """
        self.playback_controls.enable_controls(enable)
    
    def set_images(self, left_image, right_image):
        """
        Set the left and right images.
        
        Args:
            left_image (numpy.ndarray): Left OpenCV image
            right_image (numpy.ndarray): Right OpenCV image
            
        Returns:
            tuple: (left_success, right_success) indicating if each image was successfully set
        """
        return self.stereo_view.set_images(left_image, right_image)
    
    def clear_images(self):
        """Clear both the left and right images."""
        self.stereo_view.clear_images()
        self.info_view.clear_info()
    
    def update_detection_info(self, detection_rate=0.0, pixel_coords=None, position_coords=None):
        """
        Update detection information.
        
        Args:
            detection_rate (float, optional): Detection rate (0.0 to 1.0)
            pixel_coords (tuple, optional): 2D pixel coordinates (x, y)
            position_coords (tuple, optional): 3D position coordinates (x, y, z)
        """
        if detection_rate is not None:
            self.info_view.set_detection_rate(detection_rate)
        
        if pixel_coords is not None:
            x, y = pixel_coords
            self.info_view.set_pixel_coords(x, y)
        
        if position_coords is not None:
            x, y, z = position_coords
            self.info_view.set_position_coords(x, y, z) 
            
    def is_skipping_frames(self):
        """
        Check if frame skipping is enabled.
        
        Returns:
            bool: True if frames should be skipped, False otherwise
        """
        return self.playback_controls.is_skipping_frames()
        
    def set_masks(self, left_mask, right_mask, hsv_settings=None):
        """
        Set the HSV masks for left and right images.
        
        Args:
            left_mask (numpy.ndarray): Binary mask for left image
            right_mask (numpy.ndarray): Binary mask for right image
            hsv_settings (dict, optional): HSV settings for dynamic color visualization
        """
        self.left_mask = left_mask
        self.right_mask = right_mask
        
        # Apply masks to current images if enabled
        if self.show_mask:
            self.stereo_view.set_masks(left_mask, right_mask, hsv_settings)
    
    def set_rois(self, left_roi, right_roi):
        """
        Set the ROIs for left and right images.
        
        Args:
            left_roi (dict): ROI information for left image
            right_roi (dict): ROI information for right image
        """
        # Apply ROIs to current images
        self.stereo_view.set_rois(left_roi, right_roi)
    
    def enable_mask_overlay(self, enabled=True):
        """
        Enable or disable mask overlay on images.
        
        Args:
            enabled (bool): True to enable, False to disable
        """
        self.show_mask = enabled
        
        if enabled:
            # Apply current masks with current HSV settings
            # We need to pass the hsv_settings from the controller
            # Use the stored HSV settings from the most recent mask_updated signal
            self.stereo_view.set_masks(self.left_mask, self.right_mask)
        else:
            # Clear masks
            self.stereo_view.set_masks(None, None)
            
        # Also update the stereo view mask enabled state
        self.stereo_view.enable_mask_overlay(enabled)
            
    def set_circle_images(self, left_circle_image, right_circle_image):
        """
        Set the images with detected circles.
        
        Args:
            left_circle_image (numpy.ndarray): Left image with circles drawn
            right_circle_image (numpy.ndarray): Right image with circles drawn
        """
        if left_circle_image is not None and right_circle_image is not None:
            self.stereo_view.set_images(left_circle_image, right_circle_image)
        elif left_circle_image is not None:
            left_success, _ = self.stereo_view.set_images(left_circle_image, None)
        elif right_circle_image is not None:
            _, right_success = self.stereo_view.set_images(None, right_circle_image)
            
    def connect_ball_tracking_controller(self, controller):
        """
        Connect to a ball tracking controller to receive updates.
        
        Args:
            controller: BallTrackingController instance
        """
        if controller:
            # Connect mask update signal
            controller.mask_updated.connect(self.set_masks)
            
            # Connect ROI update signal
            controller.roi_updated.connect(self.set_rois)
            
            # Connect circles processed signal
            controller.circles_processed.connect(self.set_circle_images)
            
            # Connect info view to controller
            self.info_view.connect_tracking_controller(controller)
            
            logging.info("Connected to ball tracking controller")
    
    def get_left_pixmap(self):
        """
        Get the current left image pixmap.
        
        Returns:
            QPixmap: The left image pixmap or None if not available
        """
        # Get the current pixmap from the stereo view
        if hasattr(self, 'stereo_view') and self.stereo_view:
            return self.stereo_view.left_image_view.image_label.pixmap()
        return None
    
    def get_right_pixmap(self):
        """
        Get the current right image pixmap.
        
        Returns:
            QPixmap: The right image pixmap or None if not available
        """
        # Get the current pixmap from the stereo view
        if hasattr(self, 'stereo_view') and self.stereo_view:
            return self.stereo_view.right_image_view.image_label.pixmap()
        return None
    
    def get_current_frame_info(self):
        """
        Get information about the current frame.
        
        Returns:
            dict: Frame information including file paths, or None if not available
        """
        # This implementation depends on how frame information is stored
        # For now, we'll return a basic structure with just the availability status
        if not self.get_left_pixmap() or not self.get_right_pixmap():
            return None
            
        # In a full implementation, this would include actual file paths
        # and other relevant frame information
        return {
            'has_images': True,
            'left_path': None,  # Would typically come from a frame manager
            'right_path': None  # Would typically come from a frame manager
        } 