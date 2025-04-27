#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hough Circle Visualizer module.
This module contains the HoughCircleVisualizer class for visualizing detected circles.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, List, Tuple

from src.views.visualization import OpenCVVisualizer


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
    
    def visualize(self, left_frame, right_frame):
        """
        Draw detected circles on both frames.
        
        Args:
            left_frame: The left camera frame
            right_frame: The right camera frame
            
        Returns:
            Tuple of frames with circles drawn
        """
        left_output = left_frame.copy() if left_frame is not None else None
        right_output = right_frame.copy() if right_frame is not None else None
        
        detected_circles = self.controller.get_detected_circles()
        
        if left_output is not None and detected_circles and detected_circles[0] is not None:
            left_output = OpenCVVisualizer.draw_circles(left_output, detected_circles[0])
            
        if right_output is not None and detected_circles and detected_circles[1] is not None:
            right_output = OpenCVVisualizer.draw_circles(right_output, detected_circles[1])
            
        return left_output, right_output 