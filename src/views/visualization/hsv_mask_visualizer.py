#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HSV Mask Visualizer module.
This module contains the HSVMaskVisualizer class for visualizing HSV masks.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any

from src.views.visualization import OpenCVVisualizer


class HSVMaskVisualizer:
    """
    Visualizer for HSV masks.
    Applies HSV mask overlay to images.
    """
    
    def __init__(self, controller):
        """
        Initialize the HSV mask visualizer.
        
        Args:
            controller: BallTrackingController instance
        """
        self.controller = controller
        self.left_mask = None
        self.right_mask = None
        self.hsv_settings = None
        
        # Connect to controller signals
        if controller:
            controller.mask_updated.connect(self._on_mask_updated)
            logging.info("HSVMaskVisualizer connected to controller")
    
    def _on_mask_updated(self, left_mask, right_mask, hsv_settings):
        """
        Handle mask update signal from the controller.
        
        Args:
            left_mask: Left camera HSV mask
            right_mask: Right camera HSV mask
            hsv_settings: HSV threshold settings
        """
        self.left_mask = left_mask
        self.right_mask = right_mask
        self.hsv_settings = hsv_settings
        logging.debug("HSV mask visualizer updated")
    
    def draw(self, left_image, right_image):
        """
        Draw HSV mask overlay on left and right images.
        
        Args:
            left_image: Left camera image
            right_image: Right camera image
            
        Returns:
            tuple: (left_output, right_output) - Images with HSV mask overlay
        """
        left_output = left_image.copy() if left_image is not None else None
        right_output = right_image.copy() if right_image is not None else None
        
        if left_output is not None and self.left_mask is not None:
            left_output = OpenCVVisualizer.apply_mask_overlay(
                left_output, 
                self.left_mask, 
                alpha=0.3,
                hsv_settings=self.hsv_settings
            )
            
            # Draw HSV centroid if available
            left_coords = self.controller.get_latest_coordinates()[0]
            if left_coords:
                left_output = OpenCVVisualizer.draw_centroid(left_output, (left_coords[0], left_coords[1]))
        
        if right_output is not None and self.right_mask is not None:
            right_output = OpenCVVisualizer.apply_mask_overlay(
                right_output, 
                self.right_mask, 
                alpha=0.3,
                hsv_settings=self.hsv_settings
            )
            
            # Draw HSV centroid if available
            right_coords = self.controller.get_latest_coordinates()[1]
            if right_coords:
                right_output = OpenCVVisualizer.draw_centroid(right_output, (right_coords[0], right_coords[1]))
        
        return left_output, right_output 

    def visualize(self, frame, mask):
        """
        Apply HSV mask overlay to the frame.
        
        Args:
            frame: The original frame
            mask: The HSV mask to overlay
            
        Returns:
            The frame with HSV mask overlay
        """
        if frame is None or mask is None:
            return frame
            
        return OpenCVVisualizer.apply_mask_overlay(frame, mask) 