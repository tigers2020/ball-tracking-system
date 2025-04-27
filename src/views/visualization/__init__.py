#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization package initialization.
Provides access to visualization tools for the application.
"""

import logging

# Create logger
logger = logging.getLogger(__name__)

# Import the visualizer interface and implementations
from .visualizer import (
    IVisualizer,
    OpenCVVisualizer,
    QtVisualizer,
    VisualizerFactory
)

# Export for public use
__all__ = [
    # New visualizer interface
    'IVisualizer',
    'OpenCVVisualizer',
    'QtVisualizer',
    'VisualizerFactory'
] 