#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image View module.
This module contains the ImageView class for the image view tab in the Stereo Image Player.
"""

import logging
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSplitter, QTabWidget
from PySide6.QtCore import Qt, Slot

from src.views.image_view_widget import StereoImageViewWidget
from src.views.playback_controls_widget import PlaybackControlsWidget
from src.views.bounce_overlay import BounceOverlayWidget
from src.views.tracking_overlay import TrackingOverlay
from src.utils.ui_constants import Layout
from src.utils.signal_binder import SignalBinder


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
        
        # Add tracking overlay at the top
        self.tracking_overlay = TrackingOverlay()
        main_layout.addWidget(self.tracking_overlay)
        
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
        
        # Reset bounce overlay
        if hasattr(self, 'bounce_overlay'):
            self.bounce_overlay.reset()
        
        # Reset tracking overlay
        if hasattr(self, 'tracking_overlay'):
            self.tracking_overlay.reset_tracking_info()
    
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
    
    def enable_tracking_overlay(self, enabled=True):
        """
        Enable or disable tracking overlay view.
        
        Args:
            enabled (bool): True to enable, False to disable
        """
        if hasattr(self, 'tracking_overlay'):
            self.tracking_overlay.setVisible(enabled)
    
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
    
    @Slot(int, float, tuple, tuple, tuple)
    def _on_detection_updated(self, frame_idx, detection_rate, left_coords, right_coords, position_coords):
        """
        Handle detection updates from the ball tracking controller.
        
        Args:
            frame_idx (int): Current frame index
            detection_rate (float): Detection processing time in milliseconds
            left_coords (tuple): Left image coordinates (x, y)
            right_coords (tuple): Right image coordinates (x, y)
            position_coords (tuple): 3D world coordinates (x, y, z) in meters
        """
        logging.debug(f"Detection updated - Frame: {frame_idx}, Coords: L={left_coords}, R={right_coords}, 3D={position_coords}")
        
        # Prepare tracking data dictionary
        tracking_data = {
            'frame_idx': frame_idx,
            'left_2d': left_coords,
            'right_2d': right_coords,
            'world_3d': position_coords,
            'process_time': detection_rate,
            'status': 'Tracking' if all(coord is not None for coord in [left_coords, right_coords, position_coords]) else 'Lost',
            'confidence': 100.0 if all(coord is not None for coord in [left_coords, right_coords, position_coords]) else 0.0
        }
        
        # Update tracking overlay
        if hasattr(self, 'tracking_overlay'):
            self.tracking_overlay.update_tracking_info(tracking_data)
            logging.debug("Tracking overlay updated with detection data")
    
    def connect_ball_tracking_controller(self, controller):
        """
        Connect to a ball tracking controller to receive updates.
        
        Args:
            controller: BallTrackingController instance
        """
        if controller:
            # Define signal mappings
            signal_mappings = {
                "mask_updated": self.set_masks,
                "roi_updated": self.set_rois,
                "circles_processed": self.set_circle_images,
                "detection_updated": self._on_detection_updated
            }
            
            # Connect all signals using SignalBinder
            SignalBinder.bind_all(controller, self, signal_mappings)
            
            logging.info("Connected to ball tracking controller")
            
    def connect_game_analyzer(self, analyzer):
        """
        Connect to a game analyzer controller to receive updates.
        
        Args:
            analyzer: GameAnalyzer instance
        """
        if analyzer:
            self.game_analyzer = analyzer
            
            # Connect bounce detection and court position
            analyzer.bounce_detected.connect(self._on_bounce_detected)
            
            # Connect to bounce overlay
            self.bounce_overlay.connect_game_analyzer(analyzer)
            
            logging.info("Connected to game analyzer")
    
    def _on_bounce_detected(self, bounce_event):
        """
        Handle bounce detection.
        
        Args:
            bounce_event (BounceEvent): Bounce event data
        """
        # Log that we received a bounce event
        logging.info(f"Received bounce event at frame {bounce_event.frame_idx}, velocity: {bounce_event.velocity}") 