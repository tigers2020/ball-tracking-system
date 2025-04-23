#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Coordinate Utilities module.
This module provides helper functions for coordinate manipulation and fusion.
"""

import logging
from typing import List, Tuple, Optional, Union


def fuse_coordinates(coordinates: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """
    Fuse multiple 2D coordinates into a single coordinate by averaging.
    
    Args:
        coordinates: List of (x, y) coordinate tuples to be fused
        
    Returns:
        Tuple containing the fused (x, y) coordinates or None if no valid coordinates
    """
    # If no coordinates, return None
    if not coordinates:
        return None
        
    # Compute mean coordinate
    xs, ys = zip(*coordinates)
    fused_x = sum(xs) / len(xs)
    fused_y = sum(ys) / len(ys)
    
    logging.debug(f"Fused {len(coordinates)} coordinates: {fused_x:.2f}, {fused_y:.2f}")
    
    return (fused_x, fused_y)


def get_2d_point_from_kalman(kalman_state: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Extract 2D position from Kalman filter state (x, y, vx, vy).
    
    Args:
        kalman_state: Tuple containing (x, y, vx, vy)
        
    Returns:
        Tuple containing just the position (x, y)
    """
    return (kalman_state[0], kalman_state[1]) 