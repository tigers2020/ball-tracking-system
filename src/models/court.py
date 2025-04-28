#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tennis court model.
This module contains the Court class for representing a tennis court and its dimensions.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import dataclasses
import enum


class CourtSection(str, enum.Enum):
    """
    Enum for different sections of a tennis court.
    """
    DEUCE_SERVICE = "deuce_service"  # Right service box (from server's view)
    AD_SERVICE = "ad_service"  # Left service box (from server's view)
    DEUCE_COURT = "deuce_court"  # Right side of court (from server's view)
    AD_COURT = "ad_court"  # Left side of court (from server's view)
    NET = "net"  # Net area
    OUT_OF_BOUNDS = "out_of_bounds"  # Outside the court bounds
    UNKNOWN = "unknown"  # Unknown section


@dataclasses.dataclass
class CourtLine:
    """
    Line on a tennis court.
    """
    start: np.ndarray  # 3D start position (x, y, z)
    end: np.ndarray  # 3D end position (x, y, z)
    line_type: str  # Type of line (e.g., baseline, service line, center line)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert line to dictionary.
        
        Returns:
            Dictionary representation of the line
        """
        return {
            'start': self.start.tolist(),
            'end': self.end.tolist(),
            'line_type': self.line_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CourtLine':
        """
        Create a line from dictionary.
        
        Args:
            data: Dictionary representation of the line
            
        Returns:
            CourtLine instance
        """
        return cls(
            start=np.array(data['start']),
            end=np.array(data['end']),
            line_type=data['line_type']
        )


class Court:
    """
    Court class for representing a tennis court.
    
    Standard tennis court dimensions in meters:
    - Total length: 23.77m (baseline to baseline)
    - Total width: 10.97m (for doubles), 8.23m (for singles)
    - Service box: 6.4m x 4.115m (from net)
    - Net height: 0.914m (at center), 1.07m (at posts)
    """
    
    # Default court dimensions (ITF standard) in meters
    DEFAULT_DIMENSIONS = {
        'court_length': 23.77,  # Baseline to baseline
        'court_width': 8.23,  # Singles court width
        'doubles_width': 10.97,  # Doubles court width
        'service_line_dist': 6.4,  # Distance from net to service line
        'net_height_center': 0.914,  # Net height at center
        'net_height_post': 1.07,  # Net height at posts
        'center_mark_length': 0.1,  # Length of center mark on baseline
        'service_center_mark_length': 0.1  # Length of center mark on service line
    }
    
    def __init__(self, dimensions: Optional[Dict[str, float]] = None, 
                 origin: Optional[np.ndarray] = None, 
                 rotation: float = 0.0):
        """
        Initialize a new court.
        
        Args:
            dimensions: Court dimensions (defaults to ITF standard)
            origin: Origin point for the court (defaults to [0,0,0])
            rotation: Rotation angle in degrees around z-axis (clockwise)
        """
        # Set dimensions, defaulting to standard dimensions for missing values
        self.dimensions = self.DEFAULT_DIMENSIONS.copy()
        if dimensions:
            self.dimensions.update(dimensions)
        
        # Set origin and rotation
        self.origin = origin if origin is not None else np.array([0.0, 0.0, 0.0])
        self.rotation = rotation
        
        # Calculate rotation matrix
        angle_rad = np.radians(rotation)
        self.rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
        
        # Generate court lines
        self.lines = self._generate_court_lines()
        
        # Additional properties
        self.metadata: Dict[str, Any] = {}
    
    def _generate_court_lines(self) -> List[CourtLine]:
        """
        Generate court lines based on dimensions.
        
        Returns:
            List of court lines
        """
        d = self.dimensions
        half_length = d['court_length'] / 2
        half_width_singles = d['court_width'] / 2
        half_width_doubles = d['doubles_width'] / 2
        service_line_dist = d['service_line_dist']
        
        # Create court lines in local coordinates
        lines = []
        
        # Singles sidelines
        lines.append(CourtLine(
            start=np.array([-half_length, -half_width_singles, 0]),
            end=np.array([half_length, -half_width_singles, 0]),
            line_type='singles_sideline'
        ))
        lines.append(CourtLine(
            start=np.array([-half_length, half_width_singles, 0]),
            end=np.array([half_length, half_width_singles, 0]),
            line_type='singles_sideline'
        ))
        
        # Doubles sidelines
        lines.append(CourtLine(
            start=np.array([-half_length, -half_width_doubles, 0]),
            end=np.array([half_length, -half_width_doubles, 0]),
            line_type='doubles_sideline'
        ))
        lines.append(CourtLine(
            start=np.array([-half_length, half_width_doubles, 0]),
            end=np.array([half_length, half_width_doubles, 0]),
            line_type='doubles_sideline'
        ))
        
        # Baselines
        lines.append(CourtLine(
            start=np.array([-half_length, -half_width_doubles, 0]),
            end=np.array([-half_length, half_width_doubles, 0]),
            line_type='baseline'
        ))
        lines.append(CourtLine(
            start=np.array([half_length, -half_width_doubles, 0]),
            end=np.array([half_length, half_width_doubles, 0]),
            line_type='baseline'
        ))
        
        # Net line (conceptual)
        lines.append(CourtLine(
            start=np.array([0, -half_width_doubles, d['net_height_post']]),
            end=np.array([0, half_width_doubles, d['net_height_post']]),
            line_type='net'
        ))
        
        # Service lines
        lines.append(CourtLine(
            start=np.array([-half_length + service_line_dist, -half_width_singles, 0]),
            end=np.array([-half_length + service_line_dist, half_width_singles, 0]),
            line_type='service_line'
        ))
        lines.append(CourtLine(
            start=np.array([half_length - service_line_dist, -half_width_singles, 0]),
            end=np.array([half_length - service_line_dist, half_width_singles, 0]),
            line_type='service_line'
        ))
        
        # Center service line
        lines.append(CourtLine(
            start=np.array([-half_length + service_line_dist, 0, 0]),
            end=np.array([half_length - service_line_dist, 0, 0]),
            line_type='center_service_line'
        ))
        
        # Center marks on baselines
        center_mark_length = d['center_mark_length']
        lines.append(CourtLine(
            start=np.array([-half_length, -center_mark_length/2, 0]),
            end=np.array([-half_length, center_mark_length/2, 0]),
            line_type='center_mark'
        ))
        lines.append(CourtLine(
            start=np.array([half_length, -center_mark_length/2, 0]),
            end=np.array([half_length, center_mark_length/2, 0]),
            line_type='center_mark'
        ))
        
        # Transform lines to world coordinates
        return [self._transform_line(line) for line in lines]
    
    def _transform_point(self, point: np.ndarray) -> np.ndarray:
        """
        Transform a point from local court coordinates to world coordinates.
        
        Args:
            point: Point in local court coordinates
            
        Returns:
            Point in world coordinates
        """
        # Apply rotation and then translation
        rotated_point = np.dot(self.rotation_matrix, point)
        return rotated_point + self.origin
    
    def _transform_line(self, line: CourtLine) -> CourtLine:
        """
        Transform a line from local court coordinates to world coordinates.
        
        Args:
            line: Line in local court coordinates
            
        Returns:
            Line in world coordinates
        """
        return CourtLine(
            start=self._transform_point(line.start),
            end=self._transform_point(line.end),
            line_type=line.line_type
        )
    
    def point_to_court_coordinates(self, point: np.ndarray) -> np.ndarray:
        """
        Convert a point from world coordinates to local court coordinates.
        
        Args:
            point: Point in world coordinates
            
        Returns:
            Point in local court coordinates
        """
        # Translate to origin and then apply inverse rotation
        translated_point = point - self.origin
        inv_rotation_matrix = self.rotation_matrix.T  # Transpose is inverse for rotation matrices
        return np.dot(inv_rotation_matrix, translated_point)
    
    def get_section(self, point: np.ndarray) -> CourtSection:
        """
        Get the court section containing a point in world coordinates.
        
        Args:
            point: Point in world coordinates
            
        Returns:
            Court section containing the point
        """
        # Convert to court coordinates
        court_point = self.point_to_court_coordinates(point)
        x, y, z = court_point
        
        d = self.dimensions
        half_length = d['court_length'] / 2
        half_width_singles = d['court_width'] / 2
        service_line_dist = d['service_line_dist']
        
        # Check if point is within court bounds
        if (x < -half_length or x > half_length or 
            y < -half_width_singles or y > half_width_singles):
            return CourtSection.OUT_OF_BOUNDS
        
        # Check if point is in service boxes
        if abs(x) < half_length - service_line_dist:
            if y > 0:
                return CourtSection.AD_SERVICE
            else:
                return CourtSection.DEUCE_SERVICE
        
        # Check if point is in ad or deuce court
        if y > 0:
            return CourtSection.AD_COURT
        else:
            return CourtSection.DEUCE_COURT
    
    def is_in_bounds(self, point: np.ndarray, singles: bool = True) -> bool:
        """
        Check if a point is within the court boundaries.
        
        Args:
            point: Point in world coordinates
            singles: Whether to check singles or doubles court bounds
            
        Returns:
            True if the point is within bounds, False otherwise
        """
        # Convert to court coordinates
        court_point = self.point_to_court_coordinates(point)
        x, y, z = court_point
        
        d = self.dimensions
        half_length = d['court_length'] / 2
        half_width = d['court_width'] / 2 if singles else d['doubles_width'] / 2
        
        # Check if point is within court bounds
        return (-half_length <= x <= half_length) and (-half_width <= y <= half_width)
    
    def distance_to_nearest_line(self, point: np.ndarray) -> Tuple[float, str]:
        """
        Calculate the distance from a point to the nearest court line.
        
        Args:
            point: Point in world coordinates
            
        Returns:
            Tuple of (distance, line_type)
        """
        min_distance = float('inf')
        nearest_line_type = None
        
        for line in self.lines:
            # Calculate the distance from the point to the line segment
            v = line.end - line.start
            w = point - line.start
            
            c1 = np.dot(w, v)
            if c1 <= 0:
                # Point is before the line segment
                distance = np.linalg.norm(point - line.start)
            else:
                c2 = np.dot(v, v)
                if c1 >= c2:
                    # Point is after the line segment
                    distance = np.linalg.norm(point - line.end)
                else:
                    # Point is between the line segment
                    b = c1 / c2
                    pb = line.start + b * v
                    distance = np.linalg.norm(point - pb)
            
            if distance < min_distance:
                min_distance = distance
                nearest_line_type = line.line_type
        
        return min_distance, nearest_line_type
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert court to dictionary.
        
        Returns:
            Dictionary representation of the court
        """
        return {
            'dimensions': self.dimensions,
            'origin': self.origin.tolist(),
            'rotation': self.rotation,
            'lines': [line.to_dict() for line in self.lines],
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Court':
        """
        Create a court from dictionary.
        
        Args:
            data: Dictionary representation of the court
            
        Returns:
            Court instance
        """
        court = cls(
            dimensions=data.get('dimensions'),
            origin=np.array(data['origin']) if 'origin' in data else None,
            rotation=data.get('rotation', 0.0)
        )
        
        # Set metadata
        if 'metadata' in data:
            court.metadata = data['metadata']
        
        return court
    
    def __str__(self) -> str:
        """
        Get a string representation of the court.
        
        Returns:
            String representation
        """
        return f"Court(origin={self.origin}, rotation={self.rotation})" 