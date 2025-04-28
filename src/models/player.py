#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Player model.
This module contains the Player class for representing a tennis player.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import dataclasses
import uuid
from datetime import datetime


@dataclasses.dataclass
class PlayerStats:
    """
    Statistics for a player in a match or across multiple matches.
    """
    # Match play stats
    aces: int = 0
    double_faults: int = 0
    first_serves_in: int = 0
    first_serves_total: int = 0
    second_serves_in: int = 0
    second_serves_total: int = 0
    first_serve_points_won: int = 0
    second_serve_points_won: int = 0
    break_points_saved: int = 0
    break_points_faced: int = 0
    service_games_played: int = 0
    service_games_won: int = 0
    
    # Return stats
    return_points_won: int = 0
    return_points_played: int = 0
    break_points_converted: int = 0
    break_points_opportunities: int = 0
    return_games_played: int = 0
    return_games_won: int = 0
    
    # Rally stats
    winners: int = 0
    unforced_errors: int = 0
    forced_errors: int = 0
    net_points_won: int = 0
    net_points_total: int = 0
    
    # Totals
    points_won: int = 0
    points_played: int = 0
    games_won: int = 0
    games_played: int = 0
    sets_won: int = 0
    sets_played: int = 0
    matches_won: int = 0
    matches_played: int = 0
    
    def first_serve_percentage(self) -> float:
        """Calculate first serve percentage"""
        return self.first_serves_in / self.first_serves_total if self.first_serves_total > 0 else 0.0
    
    def second_serve_percentage(self) -> float:
        """Calculate second serve percentage"""
        return self.second_serves_in / self.second_serves_total if self.second_serves_total > 0 else 0.0
    
    def first_serve_points_won_percentage(self) -> float:
        """Calculate percentage of points won on first serve"""
        return self.first_serve_points_won / self.first_serves_in if self.first_serves_in > 0 else 0.0
    
    def second_serve_points_won_percentage(self) -> float:
        """Calculate percentage of points won on second serve"""
        return self.second_serve_points_won / self.second_serves_in if self.second_serves_in > 0 else 0.0
    
    def service_points_won_percentage(self) -> float:
        """Calculate percentage of total service points won"""
        total_service_points = self.first_serve_points_won + self.second_serve_points_won
        total_points = self.first_serves_in + self.second_serves_in
        return total_service_points / total_points if total_points > 0 else 0.0
    
    def break_points_saved_percentage(self) -> float:
        """Calculate percentage of break points saved"""
        return self.break_points_saved / self.break_points_faced if self.break_points_faced > 0 else 0.0
    
    def return_points_won_percentage(self) -> float:
        """Calculate percentage of return points won"""
        return self.return_points_won / self.return_points_played if self.return_points_played > 0 else 0.0
    
    def break_points_conversion_percentage(self) -> float:
        """Calculate percentage of break points converted"""
        return self.break_points_converted / self.break_points_opportunities if self.break_points_opportunities > 0 else 0.0
    
    def winners_to_unforced_errors_ratio(self) -> float:
        """Calculate winners to unforced errors ratio"""
        return self.winners / self.unforced_errors if self.unforced_errors > 0 else float('inf')
    
    def net_points_won_percentage(self) -> float:
        """Calculate percentage of net points won"""
        return self.net_points_won / self.net_points_total if self.net_points_total > 0 else 0.0
    
    def points_won_percentage(self) -> float:
        """Calculate percentage of total points won"""
        return self.points_won / self.points_played if self.points_played > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert stats to dictionary.
        
        Returns:
            Dictionary representation of player stats
        """
        return {
            'aces': self.aces,
            'double_faults': self.double_faults,
            'first_serves_in': self.first_serves_in,
            'first_serves_total': self.first_serves_total,
            'second_serves_in': self.second_serves_in,
            'second_serves_total': self.second_serves_total,
            'first_serve_points_won': self.first_serve_points_won,
            'second_serve_points_won': self.second_serve_points_won,
            'break_points_saved': self.break_points_saved,
            'break_points_faced': self.break_points_faced,
            'service_games_played': self.service_games_played,
            'service_games_won': self.service_games_won,
            'return_points_won': self.return_points_won,
            'return_points_played': self.return_points_played,
            'break_points_converted': self.break_points_converted,
            'break_points_opportunities': self.break_points_opportunities,
            'return_games_played': self.return_games_played,
            'return_games_won': self.return_games_won,
            'winners': self.winners,
            'unforced_errors': self.unforced_errors,
            'forced_errors': self.forced_errors,
            'net_points_won': self.net_points_won,
            'net_points_total': self.net_points_total,
            'points_won': self.points_won,
            'points_played': self.points_played,
            'games_won': self.games_won,
            'games_played': self.games_played,
            'sets_won': self.sets_won,
            'sets_played': self.sets_played,
            'matches_won': self.matches_won,
            'matches_played': self.matches_played
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlayerStats':
        """
        Create player stats from dictionary.
        
        Args:
            data: Dictionary with player stats
            
        Returns:
            PlayerStats instance
        """
        return cls(
            aces=data.get('aces', 0),
            double_faults=data.get('double_faults', 0),
            first_serves_in=data.get('first_serves_in', 0),
            first_serves_total=data.get('first_serves_total', 0),
            second_serves_in=data.get('second_serves_in', 0),
            second_serves_total=data.get('second_serves_total', 0),
            first_serve_points_won=data.get('first_serve_points_won', 0),
            second_serve_points_won=data.get('second_serve_points_won', 0),
            break_points_saved=data.get('break_points_saved', 0),
            break_points_faced=data.get('break_points_faced', 0),
            service_games_played=data.get('service_games_played', 0),
            service_games_won=data.get('service_games_won', 0),
            return_points_won=data.get('return_points_won', 0),
            return_points_played=data.get('return_points_played', 0),
            break_points_converted=data.get('break_points_converted', 0),
            break_points_opportunities=data.get('break_points_opportunities', 0),
            return_games_played=data.get('return_games_played', 0),
            return_games_won=data.get('return_games_won', 0),
            winners=data.get('winners', 0),
            unforced_errors=data.get('unforced_errors', 0),
            forced_errors=data.get('forced_errors', 0),
            net_points_won=data.get('net_points_won', 0),
            net_points_total=data.get('net_points_total', 0),
            points_won=data.get('points_won', 0),
            points_played=data.get('points_played', 0),
            games_won=data.get('games_won', 0),
            games_played=data.get('games_played', 0),
            sets_won=data.get('sets_won', 0),
            sets_played=data.get('sets_played', 0),
            matches_won=data.get('matches_won', 0),
            matches_played=data.get('matches_played', 0)
        )


@dataclasses.dataclass
class PlayerTrackingData:
    """
    Tracking data for a player, including position, movement, and posture.
    """
    timestamp: float  # Time in seconds
    frame_index: int  # Frame index
    position: Optional[np.ndarray] = None  # 2D or 3D position (x, y, z)
    velocity: Optional[np.ndarray] = None  # 2D or 3D velocity vector
    orientation: Optional[float] = None  # Orientation angle in degrees
    posture: Optional[str] = None  # e.g., 'standing', 'running', 'hitting'
    confidence: float = 1.0  # Detection confidence
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)  # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tracking data to dictionary"""
        return {
            'timestamp': self.timestamp,
            'frame_index': self.frame_index,
            'position': self.position.tolist() if self.position is not None else None,
            'velocity': self.velocity.tolist() if self.velocity is not None else None,
            'orientation': self.orientation,
            'posture': self.posture,
            'confidence': self.confidence,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlayerTrackingData':
        """Create tracking data from dictionary"""
        position = np.array(data['position']) if data.get('position') is not None else None
        velocity = np.array(data['velocity']) if data.get('velocity') is not None else None
        
        return cls(
            timestamp=data['timestamp'],
            frame_index=data['frame_index'],
            position=position,
            velocity=velocity,
            orientation=data.get('orientation'),
            posture=data.get('posture'),
            confidence=data.get('confidence', 1.0),
            metadata=data.get('metadata', {})
        )


class Player:
    """
    Player class.
    
    This class represents a tennis player, including their personal information,
    current position, tracking data, and statistics.
    """
    
    def __init__(self, name: str, player_id: Optional[str] = None, 
                 handedness: str = 'right', height_cm: Optional[float] = None,
                 ranking: Optional[int] = None):
        """
        Initialize a new player.
        
        Args:
            name: Player name
            player_id: Optional player ID (will be generated if not provided)
            handedness: Player's dominant hand ('right', 'left', or 'ambidextrous')
            height_cm: Player height in centimeters
            ranking: Player's current ranking
        """
        self.name = name
        self.id = player_id if player_id else str(uuid.uuid4())
        self.handedness = handedness
        self.height_cm = height_cm
        self.ranking = ranking
        
        # Current state
        self.current_position: Optional[np.ndarray] = None
        self.current_orientation: Optional[float] = None
        
        # Tracking history
        self.tracking_data: List[PlayerTrackingData] = []
        
        # Statistics
        self.stats = PlayerStats()
        
        # Additional properties
        self.metadata: Dict[str, Any] = {}
        self.created_at = datetime.now()
    
    def update_position(self, position: np.ndarray, timestamp: float, 
                       frame_index: int, orientation: Optional[float] = None,
                       velocity: Optional[np.ndarray] = None, 
                       posture: Optional[str] = None,
                       confidence: float = 1.0,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the player's position and add to tracking data.
        
        Args:
            position: 2D or 3D position (x, y, z)
            timestamp: Time in seconds
            frame_index: Frame index
            orientation: Orientation angle in degrees
            velocity: 2D or 3D velocity vector
            posture: Player posture (e.g., 'standing', 'running', 'hitting')
            confidence: Detection confidence
            metadata: Additional metadata
        """
        # Update current position
        self.current_position = position
        self.current_orientation = orientation
        
        # Add to tracking data
        tracking_point = PlayerTrackingData(
            timestamp=timestamp,
            frame_index=frame_index,
            position=position,
            velocity=velocity,
            orientation=orientation,
            posture=posture,
            confidence=confidence,
            metadata=metadata or {}
        )
        self.tracking_data.append(tracking_point)
        
        # Sort tracking data by timestamp
        self.tracking_data.sort(key=lambda p: p.timestamp)
    
    def get_tracking_data_by_time_range(self, start_time: float, end_time: float) -> List[PlayerTrackingData]:
        """
        Get tracking data within a specified time range.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            List of tracking data points within the specified time range
        """
        return [p for p in self.tracking_data if start_time <= p.timestamp <= end_time]
    
    def get_tracking_data_by_frame_range(self, start_frame: int, end_frame: int) -> List[PlayerTrackingData]:
        """
        Get tracking data within a specified frame range.
        
        Args:
            start_frame: Start frame index
            end_frame: End frame index
            
        Returns:
            List of tracking data points within the specified frame range
        """
        return [p for p in self.tracking_data if start_frame <= p.frame_index <= end_frame]
    
    def calculate_average_position(self, start_time: Optional[float] = None, 
                                  end_time: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Calculate the average position over a specified time range.
        
        Args:
            start_time: Start time in seconds (optional)
            end_time: End time in seconds (optional)
            
        Returns:
            Average position vector or None if calculation is not possible
        """
        # Get relevant tracking data
        if start_time is not None and end_time is not None:
            relevant_data = self.get_tracking_data_by_time_range(start_time, end_time)
        else:
            relevant_data = self.tracking_data
        
        # Calculate average position
        positions = [p.position for p in relevant_data if p.position is not None]
        if not positions:
            return None
        
        return np.mean(positions, axis=0)
    
    def calculate_distance_traveled(self, start_time: Optional[float] = None,
                                   end_time: Optional[float] = None) -> float:
        """
        Calculate the total distance traveled over a specified time range.
        
        Args:
            start_time: Start time in seconds (optional)
            end_time: End time in seconds (optional)
            
        Returns:
            Total distance traveled in the same units as positions
        """
        # Get relevant tracking data
        if start_time is not None and end_time is not None:
            relevant_data = self.get_tracking_data_by_time_range(start_time, end_time)
        else:
            relevant_data = self.tracking_data
        
        # Sort by timestamp to ensure correct order
        relevant_data.sort(key=lambda p: p.timestamp)
        
        # Calculate total distance
        total_distance = 0.0
        for i in range(1, len(relevant_data)):
            prev_pos = relevant_data[i-1].position
            curr_pos = relevant_data[i].position
            
            if prev_pos is not None and curr_pos is not None:
                distance = np.linalg.norm(curr_pos - prev_pos)
                total_distance += distance
        
        return total_distance
    
    def interpolate_position(self, timestamp: float) -> Optional[np.ndarray]:
        """
        Interpolate the player's position at a given timestamp.
        
        Args:
            timestamp: Timestamp for interpolation
            
        Returns:
            Interpolated position or None if interpolation is not possible
        """
        if not self.tracking_data:
            return None
        
        # Check if timestamp is within the tracking data's time range
        if timestamp < self.tracking_data[0].timestamp or timestamp > self.tracking_data[-1].timestamp:
            return None
        
        # Find the two tracking points that bracket the requested timestamp
        for i in range(len(self.tracking_data) - 1):
            p1 = self.tracking_data[i]
            p2 = self.tracking_data[i + 1]
            
            if p1.timestamp <= timestamp <= p2.timestamp:
                # Make sure both points have position data
                if p1.position is None or p2.position is None:
                    continue
                    
                # Linear interpolation
                t = (timestamp - p1.timestamp) / (p2.timestamp - p1.timestamp)
                return p1.position + t * (p2.position - p1.position)
        
        return None
    
    def update_stats(self, new_stats: Dict[str, int]) -> None:
        """
        Update player statistics.
        
        Args:
            new_stats: Dictionary with statistic increments
        """
        stats_dict = self.stats.to_dict()
        
        # Update each statistic
        for key, value in new_stats.items():
            if key in stats_dict:
                current_value = getattr(self.stats, key)
                setattr(self.stats, key, current_value + value)
    
    def calculate_average_speed(self, start_time: Optional[float] = None,
                               end_time: Optional[float] = None) -> Optional[float]:
        """
        Calculate the average speed over a specified time range.
        
        Args:
            start_time: Start time in seconds (optional)
            end_time: End time in seconds (optional)
            
        Returns:
            Average speed in distance units per second or None if calculation is not possible
        """
        # Get relevant tracking data
        if start_time is not None and end_time is not None:
            relevant_data = self.get_tracking_data_by_time_range(start_time, end_time)
        else:
            relevant_data = self.tracking_data
        
        if len(relevant_data) < 2:
            return None
        
        # Calculate total distance and time
        total_distance = self.calculate_distance_traveled(start_time, end_time)
        time_elapsed = relevant_data[-1].timestamp - relevant_data[0].timestamp
        
        if time_elapsed <= 0:
            return None
        
        return total_distance / time_elapsed
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert player to dictionary.
        
        Returns:
            Dictionary representation of the player
        """
        return {
            'id': self.id,
            'name': self.name,
            'handedness': self.handedness,
            'height_cm': self.height_cm,
            'ranking': self.ranking,
            'current_position': self.current_position.tolist() if self.current_position is not None else None,
            'current_orientation': self.current_orientation,
            'tracking_data': [p.to_dict() for p in self.tracking_data],
            'stats': self.stats.to_dict(),
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Player':
        """
        Create a player from dictionary.
        
        Args:
            data: Dictionary representation of the player
            
        Returns:
            Player instance
        """
        player = cls(
            name=data['name'],
            player_id=data.get('id'),
            handedness=data.get('handedness', 'right'),
            height_cm=data.get('height_cm'),
            ranking=data.get('ranking')
        )
        
        # Set current position and orientation
        if 'current_position' in data and data['current_position'] is not None:
            player.current_position = np.array(data['current_position'])
        player.current_orientation = data.get('current_orientation')
        
        # Set tracking data
        if 'tracking_data' in data and isinstance(data['tracking_data'], list):
            for point_data in data['tracking_data']:
                player.tracking_data.append(PlayerTrackingData.from_dict(point_data))
        
        # Set statistics
        if 'stats' in data and isinstance(data['stats'], dict):
            player.stats = PlayerStats.from_dict(data['stats'])
        
        # Set metadata
        if 'metadata' in data and isinstance(data['metadata'], dict):
            player.metadata = data['metadata']
        
        # Set creation timestamp
        if 'created_at' in data:
            try:
                player.created_at = datetime.fromisoformat(data['created_at'])
            except (ValueError, TypeError):
                player.created_at = datetime.now()
        
        return player
    
    def __str__(self) -> str:
        """
        Get a string representation of the player.
        
        Returns:
            String representation
        """
        return f"Player(name={self.name}, id={self.id}, handedness={self.handedness})" 