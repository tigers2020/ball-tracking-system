#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualizer Factory Module.
This module provides a factory pattern for creating appropriate visualizer instances.
"""

from typing import Dict, Any, Optional, Union
from PySide6.QtWidgets import QGraphicsScene

from src.views.visualization.base import IVisualizer

class VisualizerFactory:
    """
    Factory for creating visualizer instances based on the requested mode.
    """
    
    @staticmethod
    def create_visualizer(mode: str, **kwargs) -> IVisualizer:
        """
        Create and return a visualizer instance based on the specified mode.
        
        Args:
            mode: Visualizer type ('opencv' or 'qt')
            **kwargs: Additional parameters required by specific visualizer types
                      - For Qt visualizer: requires 'scene' parameter
        
        Returns:
            An instance of a class implementing the IVisualizer interface
            
        Raises:
            ValueError: If the requested mode is not supported
            KeyError: If required parameters for a specific mode are missing
        """
        if mode.lower() == 'opencv':
            # Lazy import to avoid circular dependencies
            from src.views.visualization.opencv_visualizer import OpenCVVisualizer
            return OpenCVVisualizer()
        
        elif mode.lower() == 'qt':
            # Check for required parameters
            if 'scene' not in kwargs:
                raise KeyError("Qt visualizer requires 'scene' parameter")
            
            # Lazy import to avoid circular dependencies
            from src.views.visualization.qt_visualizer import QtVisualizer
            return QtVisualizer(kwargs['scene'])
        
        else:
            raise ValueError(f"Unsupported visualizer mode: {mode}")
            
    @staticmethod
    def get_default_visualizer() -> IVisualizer:
        """
        Get the default visualizer (OpenCV-based).
        
        Returns:
            Default OpenCV-based visualizer
        """
        return VisualizerFactory.create_visualizer('opencv') 