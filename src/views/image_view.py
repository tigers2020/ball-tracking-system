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
        self.tracking_overlay.setVisible(True)  # Explicitly set to visible
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
            logging.info(f"Setting tracking overlay visibility to {enabled}")
            self.tracking_overlay.setVisible(enabled)
            
            if enabled:
                # Force repaint and raise to top
                self.tracking_overlay.raise_()
                self.tracking_overlay.repaint()
                
                # Log current state
                logging.info(f"Tracking overlay state after enabling: visible={self.tracking_overlay.isVisible()}, size={self.tracking_overlay.size()}")
                
                # Make sure it's still in the layout
                if self.tracking_overlay.parent() != self:
                    logging.warning("Tracking overlay parent is not the image view, re-adding to layout")
                    main_layout = self.layout()
                    main_layout.insertWidget(0, self.tracking_overlay)
    
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
            # DISCONNECT ANY EXISTING CONNECTIONS FIRST
            try:
                controller.detection_updated.disconnect()
            except:
                pass
                
            # Direct signal connection for detection updates
            controller.detection_updated.connect(self._on_detection_updated)
            
            # Connect other signals directly
            controller.mask_updated.connect(self.set_masks)
            controller.roi_updated.connect(self.set_rois)
            controller.circles_processed.connect(self.set_circle_images)
            
            # Log debug information
            logging.critical("DIRECT SIGNAL CONNECTION ESTABLISHED: detection_updated -> _on_detection_updated")
    
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
        logging.critical(f"★★★ RECEIVED SIGNAL! Frame: {frame_idx}, L={left_coords}, R={right_coords}, 3D={position_coords}")
        
        # DIRECT METHOD CALL (bypass signals)
        if hasattr(self, 'tracking_overlay') and self.tracking_overlay is not None:
            # Create dict with correct data
            tracking_data = {
                'frame_idx': frame_idx,
                'left_coords': left_coords if left_coords is not None else (0.0, 0.0),
                'right_coords': right_coords if right_coords is not None else (0.0, 0.0),
                'world_coords': position_coords if position_coords is not None else (0.0, 0.0, 0.0),
                'process_time': detection_rate,
                'status': 'Tracking' if left_coords is not None and right_coords is not None else 'Lost',
                'confidence': 100.0 if left_coords is not None and right_coords is not None else 0.0
            }
            
            # DIRECT METHOD CALL - Forcibly update overlay
            try:
                logging.critical(f"★★★ Calling update_tracking_info directly with data: {tracking_data}")
                self.tracking_overlay.update_tracking_info(tracking_data)
                self.tracking_overlay.repaint()  # Force immediate repaint
            except Exception as e:
                logging.critical(f"ERROR updating tracking overlay: {str(e)}")
                import traceback
                logging.critical(traceback.format_exc())
    
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