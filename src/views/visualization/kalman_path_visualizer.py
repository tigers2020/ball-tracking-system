#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kalman Path Visualizer module.
This module contains the KalmanPathVisualizer class for visualizing Kalman filter predictions and trajectory.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, List, Tuple

from src.views.visualization.kalman_visualizer import draw_prediction, draw_trajectory


class KalmanPathVisualizer:
    """
    Visualizer for Kalman filter predictions and trajectories.
    Draws trajectory paths and prediction vectors on images.
    """
    
    def __init__(self, controller):
        """
        Initialize the Kalman path visualizer.
        
        Args:
            controller: BallTrackingController instance
        """
        self.controller = controller
        self.left_predictions = None
        self.right_predictions = None
        
        # Optional: Connect to state_updated signal if available
        if controller and hasattr(controller, 'tracking_updated'):
            controller.tracking_updated.connect(self._on_tracking_updated)
            logging.info("KalmanPathVisualizer connected to controller tracking_updated")
    
    def _on_tracking_updated(self, x, y, z):
        """
        Handle tracking update signal from the controller.
        
        Args:
            x: X coordinate in 3D space
            y: Y coordinate in 3D space
            z: Z coordinate in 3D space
        """
        # This signal only provides 3D coordinates, so we still need to get 
        # the Kalman filter predictions separately
        self._update_predictions()
        logging.debug("Kalman path visualizer updated")
    
    def _update_predictions(self):
        """
        Update Kalman filter predictions from the controller.
        """
        # Get predictions from controller
        if hasattr(self.controller, 'get_predictions'):
            self.left_predictions = self.controller.get_predictions().get('left')
            self.right_predictions = self.controller.get_predictions().get('right')
        else:
            self.left_predictions = None
            self.right_predictions = None
    
    def draw(self, left_image, right_image):
        """
        Draw Kalman filter predictions and trajectories on left and right images.
        
        Args:
            left_image: Left camera image
            right_image: Right camera image
            
        Returns:
            tuple: (left_output, right_output) - Images with Kalman predictions drawn
        """
        left_output = left_image.copy() if left_image is not None else None
        right_output = right_image.copy() if right_image is not None else None
        
        # Update predictions if not already done
        if self.left_predictions is None or self.right_predictions is None:
            self._update_predictions()
        
        # Draw Kalman trajectory using coordinate history
        if hasattr(self.controller, 'get_coordinate_history'):
            left_history = self.controller.get_coordinate_history('left', 20)
            right_history = self.controller.get_coordinate_history('right', 20)
            
            # Convert to position-only list
            if left_history:
                left_positions = [(x, y) for x, y, r, *_ in left_history]
                if left_output is not None:
                    left_output = draw_trajectory(left_output, left_positions, max_points=20)
            
            if right_history:
                right_positions = [(x, y) for x, y, r, *_ in right_history]
                if right_output is not None:
                    right_output = draw_trajectory(right_output, right_positions, max_points=20)
            
        # Draw Kalman predictions if available
        if left_output is not None and self.left_predictions:
            # Extract positions from predictions
            current_pos = (int(self.left_predictions[0]), int(self.left_predictions[1]))
            velocity = (self.left_predictions[2], self.left_predictions[3])
            future_pos = (int(current_pos[0] + velocity[0] * 5), int(current_pos[1] + velocity[1] * 5))
            
            left_output = draw_prediction(
                left_output,
                current_pos,
                future_pos,
                arrow_color=(255, 0, 255),  # Magenta
                thickness=2,
                draw_uncertainty=True,
                world_pos=self.controller.current_position_3d  # 3D 좌표 전달
            )
        
        if right_output is not None and self.right_predictions:
            # Extract positions from predictions
            current_pos = (int(self.right_predictions[0]), int(self.right_predictions[1]))
            velocity = (self.right_predictions[2], self.right_predictions[3])
            future_pos = (int(current_pos[0] + velocity[0] * 5), int(current_pos[1] + velocity[1] * 5))
            
            right_output = draw_prediction(
                right_output,
                current_pos,
                future_pos,
                arrow_color=(255, 0, 255),  # Magenta
                thickness=2,
                draw_uncertainty=True,
                world_pos=self.controller.current_position_3d  # 3D 좌표 전달
            )
        
        return left_output, right_output 