#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ball Tracking Data model.
This module contains data classes for ball tracking information.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np


@dataclass
class BallTrackingData:
    """Data class for 3D tracking data."""
    frame_index: int
    timestamp: float
    detection_rate: float
    position_2d_left: Optional[Tuple[float, float]]
    position_2d_right: Optional[Tuple[float, float]]
    position_3d: Optional[np.ndarray]
    velocity_3d: Optional[np.ndarray]
    is_valid: bool = False
    confidence: float = 0.0
    position_left: Optional[Tuple[float, float]] = None
    position_right: Optional[Tuple[float, float]] = None
    state: Optional[str] = None 