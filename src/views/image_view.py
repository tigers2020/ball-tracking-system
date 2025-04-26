#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image View module.
This module contains the ImageView class for the image view tab in the Stereo Image Player.
"""

import logging
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSplitter, QTabWidget
from PySide6.QtCore import Qt

from src.views.image_view_widget import StereoImageViewWidget
from src.views.playback_controls_widget import PlaybackControlsWidget
from src.views.info_view import InfoView
from src.views.bounce_overlay import BounceOverlayWidget
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
        
        # Game analyzer reference
        self.game_analyzer = None
    
    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(Layout.MARGIN, Layout.MARGIN, Layout.MARGIN, Layout.MARGIN)
        main_layout.setSpacing(Layout.SPACING)
        
        # Create info view
        self.info_view = InfoView()
        main_layout.addWidget(self.info_view)
        
        # Create splitter for image view and bounce overlay
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Create stereo image view
        self.stereo_view = StereoImageViewWidget()
        self.splitter.addWidget(self.stereo_view)
        
        # Create bounce overlay view (initially hidden)
        self.bounce_overlay = BounceOverlayWidget()
        self.splitter.addWidget(self.bounce_overlay)
        
        # Set default splitter sizes
        self.splitter.setSizes([int(self.width() * 0.7), int(self.width() * 0.3)])
        main_layout.addWidget(self.splitter)
        
        # Create tab widget for analysis views
        self.analysis_tabs = QTabWidget()
        self.analysis_tabs.setTabPosition(QTabWidget.South)
        self.analysis_tabs.setMaximumHeight(200)
        self.analysis_tabs.setVisible(False)  # Initially hidden
        main_layout.addWidget(self.analysis_tabs)
        
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
        
        # Reset bounce overlay
        if hasattr(self, 'bounce_overlay'):
            self.bounce_overlay.reset()
    
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
    
    def enable_bounce_overlay(self, enabled=True):
        """
        Enable or disable bounce overlay view.
        
        Args:
            enabled (bool): True to enable, False to disable
        """
        self.bounce_overlay.setVisible(enabled)
        
        # Adjust splitter sizes when toggling overlay
        if enabled:
            self.splitter.setSizes([int(self.width() * 0.5), int(self.width() * 0.5)])
        else:
            self.splitter.setSizes([int(self.width()), 0])
    
    def enable_analysis_tabs(self, enabled=True):
        """
        Enable or disable analysis tabs.
        
        Args:
            enabled (bool): True to enable, False to disable
        """
        self.analysis_tabs.setVisible(enabled)
    
    def set_circle_images(self, left_circle_image, right_circle_image):
        """
        Set the Hough circle detection images.
        
        Args:
            left_circle_image (numpy.ndarray): Left image with circles
            right_circle_image (numpy.ndarray): Right image with circles
        """
        self.stereo_view.set_images(left_circle_image, right_circle_image)
    
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
            
    def connect_game_analyzer(self, analyzer):
        """
        Connect to a game analyzer controller to receive updates.
        
        Args:
            analyzer: GameAnalyzer instance
        """
        if analyzer:
            # Store reference
            self.game_analyzer = analyzer
            
            # Connect bounce overlay to analyzer
            self.bounce_overlay.connect_game_analyzer(analyzer)
            
            # Show bounce overlay and analysis tabs
            self.enable_bounce_overlay(True)
            self.enable_analysis_tabs(True)
            
            # Connect info view directly to game analyzer
            self.info_view.connect_game_analyzer(analyzer)
            
            # No longer need to connect to court_position_updated separately 
            # since InfoView now connects directly
            
            # Connect bounce events to info view
            analyzer.bounce_detected.connect(self._on_bounce_detected)
            
            logging.info("Connected to game analyzer")
    
    def _on_bounce_detected(self, bounce_event):
        """
        Handle bounce events from game analyzer.
        
        Args:
            bounce_event: BounceEvent object
        """
        # Update info view with bounce information
        position = bounce_event.position
        message = f"Bounce {'IN' if bounce_event.is_inside_court else 'OUT'} at ({position[0]:.2f}, {position[1]:.2f})"
        
        # Update bounce info in info view
        if hasattr(self.info_view, 'set_bounce_info'):
            self.info_view.set_bounce_info(message)
        
        logging.info(f"Bounce detected: {message}") 