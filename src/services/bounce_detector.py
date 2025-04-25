#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bounce Detector Service Module
This module contains the BounceDetector class for detecting ball bounces.
"""

import logging
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass

from src.geometry.court_frame import is_point_inside_court, calculate_landing_position


@dataclass
class BounceEvent:
    """Data class for bounce events."""
    frame_index: int
    timestamp: float
    position: np.ndarray  # [x, y, z]
    velocity_before: np.ndarray  # [vx, vy, vz]
    velocity_after: np.ndarray  # [vx, vy, vz]
    is_inside_court: bool
    predicted_landing: Optional[np.ndarray] = None  # [x, y, 0]


class BounceDetector:
    """
    Service class for detecting ball bounces based on 3D trajectory.
    """

    def __init__(self, settings: dict = None):
        """
        Initialize bounce detector.
        
        Args:
            settings: Dictionary containing bounce detector settings
        """
        # Set default values
        settings = settings or {}
        
        # Extract settings with defaults
        self.height_threshold = settings.get("height_threshold", 0.03)  # 3 cm above ground
        self.velocity_threshold = settings.get("velocity_threshold", 0.5)  # m/s
        self.min_bounce_interval = settings.get("min_bounce_interval", 3)  # frames
        self.min_ball_radius = settings.get("min_ball_radius", 0.05)  # 5 cm
        self.prediction_enabled = settings.get("prediction_enabled", True)
        
        # State tracking
        self.last_bounce_frame = -self.min_bounce_interval
        self.bounce_events = []
        self.max_events = settings.get("max_events", 50)
        
        # Previous velocity for comparison
        self.prev_velocity = None
        
        logging.info(f"Bounce detector initialized with height_threshold={self.height_threshold}m, "
                     f"velocity_threshold={self.velocity_threshold}m/s, "
                     f"min_bounce_interval={self.min_bounce_interval} frames")

    def update_params(self, settings: dict) -> None:
        """
        Update bounce detector parameters.
        
        Args:
            settings: Dictionary containing updated bounce detector settings
        """
        if "height_threshold" in settings:
            self.height_threshold = settings["height_threshold"]
        if "velocity_threshold" in settings:
            self.velocity_threshold = settings["velocity_threshold"]
        if "min_bounce_interval" in settings:
            self.min_bounce_interval = settings["min_bounce_interval"]
        if "min_ball_radius" in settings:
            self.min_ball_radius = settings["min_ball_radius"]
        if "max_events" in settings:
            self.max_events = settings["max_events"]
        if "prediction_enabled" in settings:
            self.prediction_enabled = settings["prediction_enabled"]
            
        logging.info(f"Bounce detector parameters updated: height_threshold={self.height_threshold}m, "
                     f"velocity_threshold={self.velocity_threshold}m/s")

    def check_bounce(self, frame_index: int, timestamp: float, position: np.ndarray, 
                    velocity: np.ndarray) -> Optional[BounceEvent]:
        """
        Check if a bounce has occurred.
        
        Args:
            frame_index: Current frame index
            timestamp: Current timestamp
            position: Current 3D position [x, y, z]
            velocity: Current 3D velocity [vx, vy, vz]
            
        Returns:
            BounceEvent if bounce detected, None otherwise
        """
        # Initialize previous velocity if None
        if self.prev_velocity is None:
            self.prev_velocity = velocity.copy()
            return None
            
        # Check if enough frames have passed since last bounce
        if frame_index - self.last_bounce_frame < self.min_bounce_interval:
            self.prev_velocity = velocity.copy()
            return None
            
        # Check bounce conditions:
        # 1. Ball is near the ground (z < height_threshold)
        # 2. Vertical velocity changed from negative to positive
        # 3. Magnitude of velocity change is significant
        near_ground = position[2] < self.height_threshold and position[2] >= 0
        velocity_sign_change = self.prev_velocity[2] < -self.velocity_threshold and velocity[2] > self.velocity_threshold
        
        if near_ground and velocity_sign_change:
            # Calculate predicted landing position
            predicted_landing = None
            if self.prediction_enabled:
                # When bounce is detected, we're already at the ground
                # Just use the current xy position with z=0
                predicted_landing = np.array([position[0], position[1], 0.0])
            
            # Check if bounce is inside court
            is_inside = is_point_inside_court(position[0], position[1])
            
            # Create bounce event
            bounce_event = BounceEvent(
                frame_index=frame_index,
                timestamp=timestamp,
                position=position.copy(),
                velocity_before=self.prev_velocity.copy(),
                velocity_after=velocity.copy(),
                is_inside_court=is_inside,
                predicted_landing=predicted_landing
            )
            
            # Update last bounce frame
            self.last_bounce_frame = frame_index
            
            # Add to events history, maintaining max size
            self.bounce_events.append(bounce_event)
            if len(self.bounce_events) > self.max_events:
                self.bounce_events.pop(0)
                
            logging.info(f"Bounce detected at frame {frame_index}, position: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}), "
                         f"in_court: {is_inside}")
                
            # Save previous velocity and return the event
            self.prev_velocity = velocity.copy()
            return bounce_event
            
        # Update previous velocity
        self.prev_velocity = velocity.copy()
        return None

    def predict_landing(self, position: np.ndarray, velocity: np.ndarray) -> Optional[np.ndarray]:
        """
        Predict ball landing position on the ground.
        
        Args:
            position: Current ball position [x, y, z]
            velocity: Current ball velocity [vx, vy, vz]
            
        Returns:
            Predicted landing position [x, y, 0] or None if not possible
        """
        if not self.prediction_enabled:
            return None
            
        # Validate inputs
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            return None
            
        return calculate_landing_position(position, velocity)

    def get_bounce_events(self, count: int = None) -> List[BounceEvent]:
        """
        Get recent bounce events.
        
        Args:
            count: Number of recent events to return (None for all)
            
        Returns:
            List of BounceEvent objects
        """
        if count is None or count >= len(self.bounce_events):
            return self.bounce_events.copy()
        
        return self.bounce_events[-count:].copy()

    def clear_events(self) -> None:
        """Clear all bounce events."""
        self.bounce_events = []
        self.last_bounce_frame = -self.min_bounce_interval
        self.prev_velocity = None
        logging.info("Bounce events cleared")

    def get_latest_bounce(self) -> Optional[BounceEvent]:
        """
        Get the most recent bounce event.
        
        Returns:
            The most recent BounceEvent or None if no bounces detected
        """
        if not self.bounce_events:
            return None
        
        return self.bounce_events[-1]

    def get_bounce_statistics(self) -> Dict[str, Any]:
        """
        Calculate statistics about detected bounces.
        
        Returns:
            Dictionary with bounce statistics
        """
        if not self.bounce_events:
            return {
                "count": 0,
                "in_court_count": 0,
                "out_count": 0,
                "in_court_percentage": 0.0
            }
            
        # Count bounces inside/outside court
        in_court_count = sum(1 for event in self.bounce_events if event.is_inside_court)
        
        return {
            "count": len(self.bounce_events),
            "in_court_count": in_court_count,
            "out_count": len(self.bounce_events) - in_court_count,
            "in_court_percentage": in_court_count / len(self.bounce_events) * 100
        } 