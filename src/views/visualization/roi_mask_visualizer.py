#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ROI Mask Visualizer module.
This module contains the ROIMaskVisualizer class for visualizing ROI.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any

from src.views.visualization.roi_visualizer import draw_roi


class ROIMaskVisualizer:
    """
    Visualizer for Region of Interest (ROI).
    Draws ROI rectangles on images.
    """
    
    def __init__(self, controller):
        """
        Initialize the ROI mask visualizer.
        
        Args:
            controller: BallTrackingController instance
        """
        self.controller = controller
        self.left_roi = None
        self.right_roi = None
        
        # Connect to controller signals
        if controller:
            controller.roi_updated.connect(self._on_roi_updated)
            logging.info("ROIMaskVisualizer connected to controller")
    
    def _on_roi_updated(self, left_roi, right_roi):
        """
        Handle ROI update signal from the controller.
        
        Args:
            left_roi: Left camera ROI
            right_roi: Right camera ROI
        """
        self.left_roi = left_roi
        self.right_roi = right_roi
        logging.debug("ROI mask visualizer updated")
    
    def draw(self, left_image, right_image):
        """
        Draw ROI rectangles on left and right images.
        
        Args:
            left_image: Left camera image
            right_image: Right camera image
            
        Returns:
            tuple: (left_output, right_output) - Images with ROI rectangles drawn
        """
        left_output = left_image.copy() if left_image is not None else None
        right_output = right_image.copy() if right_image is not None else None
        
        if left_output is not None and self.left_roi is not None:
            left_output = draw_roi(
                left_output, 
                self.left_roi, 
                color=(255, 255, 0),  # Yellow
                thickness=2,
                show_center=True
            )
        
        if right_output is not None and self.right_roi is not None:
            right_output = draw_roi(
                right_output, 
                self.right_roi, 
                color=(255, 255, 0),  # Yellow
                thickness=2,
                show_center=True
            )
        
        return left_output, right_output 