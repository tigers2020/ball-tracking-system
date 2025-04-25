#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Court Frame module.
This module contains constants and utility functions for tennis court coordinate frame.
"""

import numpy as np
import logging
from typing import Tuple, List, Dict, Optional, Any, Union

# Tennis court dimensions in meters (standard singles court)
COURT_LENGTH = 23.77  # Total length
COURT_HALF_WIDTH = 11.885  # Half width
NET_HEIGHT = 0.914  # Height of the net
NET_Y = COURT_LENGTH / 2  # Net position (y-coordinate)

# Coordinate system:
# Origin (0,0,0) at the center of the baseline
# X-axis: left-right (negative is left, positive is right)
# Y-axis: baseline to net (0 at baseline, positive towards net)
# Z-axis: height (0 at court level, positive up)

# Court landmarks with corresponding world coordinates
COURT_LANDMARKS = {
    "center_mark": (0.0, NET_Y, 0.0),
    "net_center": (0.0, NET_Y, NET_HEIGHT/2),
    "left_baseline_corner": (-COURT_HALF_WIDTH, 0.0, 0.0),
    "right_baseline_corner": (COURT_HALF_WIDTH, 0.0, 0.0),
    "left_net_post": (-COURT_HALF_WIDTH, NET_Y, NET_HEIGHT),
    "right_net_post": (COURT_HALF_WIDTH, NET_Y, NET_HEIGHT),
    "left_service_T": (-1.37, NET_Y - 6.4, 0.0),
    "right_service_T": (1.37, NET_Y - 6.4, 0.0),
    "center_baseline": (0.0, 0.0, 0.0),
    "left_service_line_baseline": (-1.37, 0.0, 0.0),
    "right_service_line_baseline": (1.37, 0.0, 0.0),
    "left_doubles_baseline": (-5.485, 0.0, 0.0),
    "right_doubles_baseline": (5.485, 0.0, 0.0),
    "left_net_corner": (-COURT_HALF_WIDTH, NET_Y, 0.0),
    "right_net_corner": (COURT_HALF_WIDTH, NET_Y, 0.0)
}

# Margin for IN/OUT judgment (meters)
INSIDE_EPS = 0.05  # 5 cm margin


def world_pts_from_index(idx: int) -> Tuple[float, float, float]:
    """
    Get world coordinates for a court landmark by index.
    Used for PnP calibration with ordered points.
    
    Args:
        idx: Index of the court landmark
        
    Returns:
        Tuple of (X, Y, Z) world coordinates
    """
    # Define ordered landmarks for calibration
    ordered_landmarks = [
        COURT_LANDMARKS["left_baseline_corner"],
        COURT_LANDMARKS["right_baseline_corner"],
        COURT_LANDMARKS["left_net_corner"],
        COURT_LANDMARKS["right_net_corner"],
        COURT_LANDMARKS["left_service_line_baseline"],
        COURT_LANDMARKS["right_service_line_baseline"],
        COURT_LANDMARKS["left_service_T"],
        COURT_LANDMARKS["right_service_T"],
        COURT_LANDMARKS["center_baseline"],
        COURT_LANDMARKS["center_mark"],
        # Additional points if needed
        COURT_LANDMARKS["left_doubles_baseline"],
        COURT_LANDMARKS["right_doubles_baseline"],
        COURT_LANDMARKS["net_center"],
        COURT_LANDMARKS["left_net_post"],
        COURT_LANDMARKS["right_net_post"]
    ]
    
    if idx < 0 or idx >= len(ordered_landmarks):
        logging.error(f"Invalid landmark index: {idx}")
        return (0.0, 0.0, 0.0)
    
    return ordered_landmarks[idx]


def get_calibration_points() -> List[np.ndarray]:
    """
    Get a list of 3D points for camera calibration (used for solvePnP).
    
    Returns:
        List of 3D points as numpy arrays
    """
    calibration_points = []
    for i in range(10):  # Use first 10 points for calibration
        point = world_pts_from_index(i)
        calibration_points.append(np.array(point, dtype=np.float32))
    
    return calibration_points


def is_point_inside_court(x: float, y: float, z: float = None, with_margin: bool = True) -> bool:
    """
    Check if a point is inside the court boundaries.
    
    Args:
        x: X-coordinate in court frame (meters)
        y: Y-coordinate in court frame (meters)
        z: Z-coordinate in court frame (meters, optional)
        with_margin: Whether to apply a small margin for IN judgment
        
    Returns:
        True if the point is inside the court, False otherwise
    """
    margin = INSIDE_EPS if with_margin else 0.0
    
    in_x = -COURT_HALF_WIDTH - margin <= x <= COURT_HALF_WIDTH + margin
    in_y = -margin <= y <= COURT_LENGTH + margin
    
    return in_x and in_y


def is_net_crossed(prev_y: float, curr_y: float) -> bool:
    """
    Check if the ball crossed the net between two positions.
    
    Args:
        prev_y: Previous Y-coordinate in court frame
        curr_y: Current Y-coordinate in court frame
        
    Returns:
        True if the ball crossed the net, False otherwise
    """
    # Check if positions are on opposite sides of the net
    return (prev_y < NET_Y and curr_y >= NET_Y) or (prev_y >= NET_Y and curr_y < NET_Y)


def calculate_landing_position(pos: np.ndarray, vel: np.ndarray) -> Optional[np.ndarray]:
    """
    Calculate the landing position when ball hits the ground.
    
    Args:
        pos: Current position [x, y, z]
        vel: Current velocity [vx, vy, vz]
        
    Returns:
        Landing position [x, y, 0] or None if not possible to calculate
    """
    # If ball is moving upward, can't calculate landing
    if vel[2] >= 0:
        return None
    
    # Calculate time to hit ground (z=0)
    time_to_hit = -pos[2] / vel[2]
    
    # If time is negative, can't calculate landing
    if time_to_hit < 0:
        return None
    
    # Calculate landing position
    landing_x = pos[0] + vel[0] * time_to_hit
    landing_y = pos[1] + vel[1] * time_to_hit
    
    return np.array([landing_x, landing_y, 0.0]) 