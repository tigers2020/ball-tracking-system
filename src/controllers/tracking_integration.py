#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tracking Integration module.
This module provides functions to integrate the tracking controllers with the main application.
"""

import logging
from typing import Optional

from src.controllers.ball_tracking_controller import BallTrackingController
from src.controllers.tracking_coordinates_controller import TrackingCoordinatesController
from src.views.image_view import ImageView
from src.views.tracking_overlay import TrackingOverlay
from src.utils.config_manager import ConfigManager


def setup_tracking_overlay(
    app_window,
    ball_tracking_controller: BallTrackingController,
    config_manager: ConfigManager,
    image_view: Optional[ImageView] = None
) -> TrackingCoordinatesController:
    """
    Set up tracking overlay and coordinate controller.
    
    Args:
        app_window: Main application window
        ball_tracking_controller: BallTrackingController instance
        config_manager: ConfigManager instance
        image_view: ImageView instance (optional)
        
    Returns:
        TrackingCoordinatesController: The initialized tracking coordinates controller
    """
    # Create tracking coordinates controller
    tracking_coord_controller = TrackingCoordinatesController(
        ball_tracking_controller=ball_tracking_controller,
        config_manager=config_manager
    )
    
    # If image_view is provided, connect directly to its tracking overlay
    if image_view is not None:
        tracking_coord_controller.connect_to_view(image_view.tracking_overlay)
        
        # Start updates when tracking is enabled
        ball_tracking_controller.tracking_enabled_changed.connect(
            lambda enabled: tracking_coord_controller.start_updates() if enabled else tracking_coord_controller.stop_updates()
        )
        
        # Start updates if tracking is already enabled
        if ball_tracking_controller.is_enabled:
            tracking_coord_controller.start_updates()
    
    # Store controller reference in app window for future use
    if hasattr(app_window, 'tracking_coord_controller'):
        app_window.tracking_coord_controller = tracking_coord_controller
    
    logging.info("Tracking overlay integration completed")
    return tracking_coord_controller


def connect_existing_tracking_overlay(
    tracking_overlay: TrackingOverlay,
    ball_tracking_controller: BallTrackingController,
    config_manager: ConfigManager
) -> TrackingCoordinatesController:
    """
    Connect an existing tracking overlay to tracking controllers.
    
    Args:
        tracking_overlay: TrackingOverlay instance
        ball_tracking_controller: BallTrackingController instance
        config_manager: ConfigManager instance
        
    Returns:
        TrackingCoordinatesController: The initialized tracking coordinates controller
    """
    # Create tracking coordinates controller
    tracking_coord_controller = TrackingCoordinatesController(
        ball_tracking_controller=ball_tracking_controller,
        config_manager=config_manager
    )
    
    # Connect to tracking overlay
    tracking_coord_controller.connect_to_view(tracking_overlay)
    
    # Start updates when tracking is enabled
    ball_tracking_controller.tracking_enabled_changed.connect(
        lambda enabled: tracking_coord_controller.start_updates() if enabled else tracking_coord_controller.stop_updates()
    )
    
    # Start updates if tracking is already enabled
    if ball_tracking_controller.is_enabled:
        tracking_coord_controller.start_updates()
    
    logging.info("Connected existing tracking overlay")
    return tracking_coord_controller 