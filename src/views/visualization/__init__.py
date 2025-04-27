#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization package initialization.
Provides access to visualization tools for the application.
"""

# Legacy imports for backward compatibility
from src.views.visualization.hsv_visualizer import apply_mask_overlay, draw_centroid
from src.views.visualization.roi_visualizer import draw_roi
from src.views.visualization.hough_visualizer import draw_circles
from src.views.visualization.kalman_visualizer import draw_prediction, draw_trajectory

# Import the visualizer interface and implementations
from .visualizer import (
    IVisualizer,
    OpenCVVisualizer,
    QtVisualizer,
    VisualizerFactory
)

# Export for public use
__all__ = [
    # Legacy functions
    'apply_mask_overlay', 
    'draw_centroid',
    'draw_roi',
    'draw_circles',
    'draw_prediction',
    'draw_trajectory',
    
    # New visualizer interface
    'IVisualizer',
    'OpenCVVisualizer',
    'QtVisualizer',
    'VisualizerFactory'
] 