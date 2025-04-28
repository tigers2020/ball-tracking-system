#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ball track model.
This module contains the BallTrack class and related utilities for representing ball trajectories.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Iterator
import dataclasses
import uuid
from datetime import datetime


@dataclasses.dataclass
class TrackPoint:
    """
    TrackPoint class.
    
    This class represents a single point in a ball trajectory with position, velocity, and timestamp information.
    """
    
    position: np.ndarray  # 3D position (x, y, z)
    timestamp: float  # Time in seconds
    frame_index: int  # Frame index
    velocity: Optional[np.ndarray] = None  # 3D velocity vector (vx, vy, vz)
    acceleration: Optional[np.ndarray] = None  # 3D acceleration vector (ax, ay, az)
    confidence: float = 1.0  # Detection confidence
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)  # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the track point to a dictionary.
        
        Returns:
            Dictionary representation of the track point
        """
        return {
            'position': self.position.tolist() if self.position is not None else None,
            'timestamp': self.timestamp,
            'frame_index': self.frame_index,
            'velocity': self.velocity.tolist() if self.velocity is not None else None,
            'acceleration': self.acceleration.tolist() if self.acceleration is not None else None,
            'confidence': self.confidence,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrackPoint':
        """
        Create a track point from a dictionary.
        
        Args:
            data: Dictionary representation of the track point
            
        Returns:
            TrackPoint instance
        """
        position = np.array(data['position']) if data.get('position') is not None else None
        velocity = np.array(data['velocity']) if data.get('velocity') is not None else None
        acceleration = np.array(data['acceleration']) if data.get('acceleration') is not None else None
        
        return cls(
            position=position,
            timestamp=data['timestamp'],
            frame_index=data['frame_index'],
            velocity=velocity,
            acceleration=acceleration,
            confidence=data.get('confidence', 1.0),
            metadata=data.get('metadata', {})
        )


class BallTrack:
    """
    BallTrack class.
    
    This class represents a trajectory of a ball, consisting of a sequence of track points.
    """
    
    def __init__(self, track_id: Optional[str] = None):
        """
        Initialize a new ball track.
        
        Args:
            track_id: Optional track ID (will be generated if not provided)
        """
        self.id = track_id if track_id else str(uuid.uuid4())
        self.track_points: List[TrackPoint] = []
        self.created_at = datetime.now()
        self.metadata: Dict[str, Any] = {}
    
    def add_point(self, point: TrackPoint) -> None:
        """
        Add a track point to the trajectory.
        
        Args:
            point: TrackPoint to add
        """
        self.track_points.append(point)
        # Sort track points by timestamp to ensure they are in chronological order
        self.track_points.sort(key=lambda p: p.timestamp)
    
    def get_points_by_time_range(self, start_time: float, end_time: float) -> List[TrackPoint]:
        """
        Get track points within a specified time range.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            List of track points within the specified time range
        """
        return [p for p in self.track_points if start_time <= p.timestamp <= end_time]
    
    def get_points_by_frame_range(self, start_frame: int, end_frame: int) -> List[TrackPoint]:
        """
        Get track points within a specified frame range.
        
        Args:
            start_frame: Start frame index
            end_frame: End frame index
            
        Returns:
            List of track points within the specified frame range
        """
        return [p for p in self.track_points if start_frame <= p.frame_index <= end_frame]
    
    def calculate_velocities(self) -> None:
        """
        Calculate velocity for each track point based on position and timestamp.
        """
        if len(self.track_points) < 2:
            return
        
        # For the first point, use forward difference
        p0 = self.track_points[0]
        p1 = self.track_points[1]
        dt = p1.timestamp - p0.timestamp
        if dt > 0:
            p0.velocity = (p1.position - p0.position) / dt
        
        # For the middle points, use central difference
        for i in range(1, len(self.track_points) - 1):
            prev_point = self.track_points[i - 1]
            curr_point = self.track_points[i]
            next_point = self.track_points[i + 1]
            
            dt = next_point.timestamp - prev_point.timestamp
            if dt > 0:
                curr_point.velocity = (next_point.position - prev_point.position) / dt
        
        # For the last point, use backward difference
        p_last = self.track_points[-1]
        p_second_last = self.track_points[-2]
        dt = p_last.timestamp - p_second_last.timestamp
        if dt > 0:
            p_last.velocity = (p_last.position - p_second_last.position) / dt
    
    def calculate_accelerations(self) -> None:
        """
        Calculate acceleration for each track point based on velocity and timestamp.
        """
        if len(self.track_points) < 2:
            return
        
        # Ensure velocities are calculated
        if any(p.velocity is None for p in self.track_points):
            self.calculate_velocities()
        
        # For the first point, use forward difference
        p0 = self.track_points[0]
        p1 = self.track_points[1]
        dt = p1.timestamp - p0.timestamp
        if dt > 0 and p0.velocity is not None and p1.velocity is not None:
            p0.acceleration = (p1.velocity - p0.velocity) / dt
        
        # For the middle points, use central difference
        for i in range(1, len(self.track_points) - 1):
            prev_point = self.track_points[i - 1]
            curr_point = self.track_points[i]
            next_point = self.track_points[i + 1]
            
            dt = next_point.timestamp - prev_point.timestamp
            if dt > 0 and prev_point.velocity is not None and next_point.velocity is not None:
                curr_point.acceleration = (next_point.velocity - prev_point.velocity) / dt
        
        # For the last point, use backward difference
        p_last = self.track_points[-1]
        p_second_last = self.track_points[-2]
        dt = p_last.timestamp - p_second_last.timestamp
        if dt > 0 and p_last.velocity is not None and p_second_last.velocity is not None:
            p_last.acceleration = (p_last.velocity - p_second_last.velocity) / dt
    
    def interpolate_position(self, timestamp: float) -> Optional[np.ndarray]:
        """
        Interpolate the ball position at a given timestamp.
        
        Args:
            timestamp: Timestamp for interpolation
            
        Returns:
            Interpolated 3D position or None if interpolation is not possible
        """
        if not self.track_points:
            return None
        
        # Check if timestamp is within the track's time range
        if timestamp < self.track_points[0].timestamp or timestamp > self.track_points[-1].timestamp:
            return None
        
        # Find the two track points that bracket the requested timestamp
        for i in range(len(self.track_points) - 1):
            p1 = self.track_points[i]
            p2 = self.track_points[i + 1]
            
            if p1.timestamp <= timestamp <= p2.timestamp:
                # Linear interpolation
                t = (timestamp - p1.timestamp) / (p2.timestamp - p1.timestamp)
                return p1.position + t * (p2.position - p1.position)
        
        return None
    
    def get_duration(self) -> float:
        """
        Get the time duration of the track.
        
        Returns:
            Duration in seconds or 0 if the track is empty
        """
        if not self.track_points:
            return 0.0
        
        return self.track_points[-1].timestamp - self.track_points[0].timestamp
    
    def get_start_time(self) -> Optional[float]:
        """
        Get the start time of the track.
        
        Returns:
            Start time in seconds or None if the track is empty
        """
        if not self.track_points:
            return None
        
        return self.track_points[0].timestamp
    
    def get_end_time(self) -> Optional[float]:
        """
        Get the end time of the track.
        
        Returns:
            End time in seconds or None if the track is empty
        """
        if not self.track_points:
            return None
        
        return self.track_points[-1].timestamp
    
    def get_average_velocity(self) -> Optional[np.ndarray]:
        """
        Calculate the average velocity over the entire track.
        
        Returns:
            Average velocity vector or None if calculation is not possible
        """
        if len(self.track_points) < 2:
            return None
        
        # Ensure velocities are calculated
        if any(p.velocity is None for p in self.track_points):
            self.calculate_velocities()
        
        # Calculate average velocity from all points with velocity
        velocities = [p.velocity for p in self.track_points if p.velocity is not None]
        if not velocities:
            return None
        
        return np.mean(velocities, axis=0)
    
    def get_average_speed(self) -> Optional[float]:
        """
        Calculate the average speed over the entire track.
        
        Returns:
            Average speed (magnitude of velocity) or None if calculation is not possible
        """
        avg_velocity = self.get_average_velocity()
        if avg_velocity is None:
            return None
        
        return float(np.linalg.norm(avg_velocity))
    
    def merge_with(self, other_track: 'BallTrack') -> None:
        """
        Merge another track into this one.
        
        Args:
            other_track: Track to merge with this one
        """
        # Add all points from the other track
        for point in other_track.track_points:
            # Skip points that overlap with existing points
            if not any(abs(p.timestamp - point.timestamp) < 1e-6 for p in self.track_points):
                self.add_point(point)
        
        # Update metadata
        self.metadata.update({
            f"merged_with_{other_track.id}": {
                "timestamp": datetime.now().isoformat(),
                "points_count": len(other_track.track_points)
            }
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the track to a dictionary.
        
        Returns:
            Dictionary representation of the track
        """
        return {
            'id': self.id,
            'created_at': self.created_at.isoformat(),
            'track_points': [p.to_dict() for p in self.track_points],
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BallTrack':
        """
        Create a track from a dictionary.
        
        Args:
            data: Dictionary representation of the track
            
        Returns:
            BallTrack instance
        """
        track = cls(track_id=data.get('id'))
        
        # Set creation timestamp
        if 'created_at' in data:
            try:
                track.created_at = datetime.fromisoformat(data['created_at'])
            except (ValueError, TypeError):
                track.created_at = datetime.now()
        
        # Add track points
        if 'track_points' in data and isinstance(data['track_points'], list):
            for point_data in data['track_points']:
                point = TrackPoint.from_dict(point_data)
                track.add_point(point)
        
        # Set metadata
        if 'metadata' in data and isinstance(data['metadata'], dict):
            track.metadata = data['metadata']
        
        return track
    
    def __len__(self) -> int:
        """
        Get the number of track points.
        
        Returns:
            Number of track points
        """
        return len(self.track_points)
    
    def __iter__(self) -> Iterator[TrackPoint]:
        """
        Get an iterator over track points.
        
        Returns:
            Iterator over track points
        """
        return iter(self.track_points)
    
    def __getitem__(self, idx) -> TrackPoint:
        """
        Get a track point by index.
        
        Args:
            idx: Index
            
        Returns:
            Track point at the specified index
        """
        return self.track_points[idx]
    
    def __str__(self) -> str:
        """
        Get a string representation of the track.
        
        Returns:
            String representation
        """
        return (f"BallTrack(id={self.id}, points={len(self.track_points)}, "
                f"duration={self.get_duration():.2f}s)") 