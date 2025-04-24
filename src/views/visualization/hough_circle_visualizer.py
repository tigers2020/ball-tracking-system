#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hough Circle Visualizer module.
This module contains the HoughCircleVisualizer class for visualizing detected circles.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, List, Tuple

from src.views.visualization.hough_visualizer import draw_circles


class HoughCircleVisualizer:
    """
    Visualizer for Hough circles.
    Draws detected circles on images.
    """
    
    def __init__(self, controller):
        """
        Initialize the Hough circle visualizer.
        
        Args:
            controller: BallTrackingController instance
        """
        self.controller = controller
        self.left_circles = None
        self.right_circles = None
        
        # Connect to controller signals if it emits circles_processed
        if controller and hasattr(controller, 'circles_processed'):
            controller.circles_processed.connect(self._on_circles_processed)
            logging.info("HoughCircleVisualizer connected to controller")
    
    def _on_circles_processed(self, left_image, right_image):
        """
        Handle circles processed signal from the controller.
        Note: This is a placeholder and might need to be adapted based on the actual 
        controller implementation. The circles should be fetched from the controller or model.
        
        Args:
            left_image: Left camera image with circles (unused, just for signal compatibility)
            right_image: Right camera image with circles (unused, just for signal compatibility)
        """
        # Usually, we would extract circle information from these images,
        # but since we can get it directly from the controller, we'll use that approach
        left_circles, right_circles = self.controller.get_latest_coordinates()
        self.left_circles = [left_circles] if left_circles else None
        self.right_circles = [right_circles] if right_circles else None
        logging.debug("Hough circle visualizer updated")
    
    def draw(self, left_image, right_image):
        """
        Draw detected circles on left and right images.
        
        Args:
            left_image: Left camera image
            right_image: Right camera image
            
        Returns:
            tuple: (left_output, right_output) - Images with circles drawn
        """
        left_output = left_image.copy() if left_image is not None else None
        right_output = right_image.copy() if right_image is not None else None
        
        # Fetch latest circle data from controller if not already updated via signal
        if not self.left_circles or not self.right_circles:
            left_coords, right_coords = self.controller.get_latest_coordinates()
            self.left_circles = [left_coords] if left_coords else None
            self.right_circles = [right_coords] if right_coords else None
        
        if left_output is not None and self.left_circles:
            left_output = draw_circles(
                left_output, 
                self.left_circles, 
                main_color=(0, 255, 0),  # Green
                secondary_color=(255, 0, 0),  # Blue
                center_color=(0, 0, 255),  # Red
                thickness=2
            )
        
        if right_output is not None and self.right_circles:
            right_output = draw_circles(
                right_output, 
                self.right_circles, 
                main_color=(0, 255, 0),  # Green
                secondary_color=(255, 0, 0),  # Blue
                center_color=(0, 0, 255),  # Red
                thickness=2
            )
        
        return left_output, right_output 