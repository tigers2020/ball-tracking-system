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