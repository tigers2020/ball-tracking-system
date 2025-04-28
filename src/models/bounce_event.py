#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bounce event model.
This module contains the BounceEvent class, which represents a bounce event detected in a ball trajectory.
"""

import numpy as np
from typing import Dict, Any, Optional, List
import dataclasses


@dataclasses.dataclass
class BounceEvent:
    """
    Bounce event class.
    
    This class represents a bounce event detected in a ball trajectory.
    """
    
    # Required fields
    frame_index: int  # Frame index where the bounce was detected
    timestamp: float  # Timestamp of the bounce event
    position: np.ndarray  # 3D position of the bounce event (x, y, z)
    
    # Optional fields
    velocity_before: Optional[np.ndarray] = None  # Velocity before the bounce
    velocity_after: Optional[np.ndarray] = None  # Velocity after the bounce
    confidence: float = 0.0  # Confidence score for the bounce detection
    detection_method: str = 'combined'  # Method used to detect the bounce
    is_in_bounds: Optional[bool] = None  # Whether the bounce is within the court bounds
    court_section: Optional[str] = None  # Court section where the bounce occurred
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)  # Additional metadata

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the bounce event to a dictionary.
        
        Returns:
            Dictionary representation of the bounce event
        """
        dict_data = {
            'frame_index': self.frame_index,
            'timestamp': self.timestamp,
            'position': self.position.tolist() if self.position is not None else None,
            'velocity_before': self.velocity_before.tolist() if self.velocity_before is not None else None,
            'velocity_after': self.velocity_after.tolist() if self.velocity_after is not None else None,
            'confidence': self.confidence,
            'detection_method': self.detection_method,
            'is_in_bounds': self.is_in_bounds,
            'court_section': self.court_section,
            'metadata': self.metadata
        }
        return dict_data

    @classmethod
    def from_dict(cls, dict_data: Dict[str, Any]) -> 'BounceEvent':
        """
        Create a bounce event from a dictionary.
        
        Args:
            dict_data: Dictionary representation of the bounce event
            
        Returns:
            BounceEvent instance
        """
        # Convert position and velocity arrays from lists to numpy arrays
        position = np.array(dict_data['position']) if dict_data.get('position') is not None else None
        velocity_before = np.array(dict_data['velocity_before']) if dict_data.get('velocity_before') is not None else None
        velocity_after = np.array(dict_data['velocity_after']) if dict_data.get('velocity_after') is not None else None
        
        return cls(
            frame_index=dict_data['frame_index'],
            timestamp=dict_data['timestamp'],
            position=position,
            velocity_before=velocity_before,
            velocity_after=velocity_after,
            confidence=dict_data.get('confidence', 0.0),
            detection_method=dict_data.get('detection_method', 'combined'),
            is_in_bounds=dict_data.get('is_in_bounds'),
            court_section=dict_data.get('court_section'),
            metadata=dict_data.get('metadata', {})
        )

    def calculate_impact_angle(self) -> Optional[float]:
        """
        Calculate the impact angle of the bounce.
        
        Returns:
            Impact angle in degrees or None if velocities are not available
        """
        if self.velocity_before is None or self.velocity_after is None:
            return None
        
        # Calculate the angle between the velocity vectors
        v_before = self.velocity_before
        v_after = self.velocity_after
        
        # Normalize the vectors
        v_before_norm = v_before / np.linalg.norm(v_before)
        v_after_norm = v_after / np.linalg.norm(v_after)
        
        # Calculate the dot product
        dot_product = np.clip(np.dot(v_before_norm, v_after_norm), -1.0, 1.0)
        
        # Calculate the angle in degrees
        angle = np.degrees(np.arccos(dot_product))
        
        return angle

    def calculate_bounce_height(self) -> float:
        """
        Get the height of the bounce.
        
        Returns:
            Bounce height (z-coordinate)
        """
        return float(self.position[2]) if self.position is not None else 0.0

    def __str__(self) -> str:
        """
        Get a string representation of the bounce event.
        
        Returns:
            String representation
        """
        return (f"BounceEvent(frame={self.frame_index}, time={self.timestamp:.3f}, "
                f"pos=({self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f}), "
                f"confidence={self.confidence:.2f}, in_bounds={self.is_in_bounds}, "
                f"section={self.court_section})") 