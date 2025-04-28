#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bounce detector service module.
This module contains the BounceDetectorService class, which integrates
the bounce detector with the rest of the system.
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Any, Tuple

from src.services.ball_tracking.bounce_detector import BounceDetector
from src.models.bounce_event import BounceEvent
from src.models.ball_track import BallTrack


class BounceDetectorService:
    """
    Bounce detector service class.
    
    This class provides a high-level interface for detecting bounces
    in ball trajectories and integrates with the rest of the system.
    """
    
    def __init__(self, settings: Dict[str, Any] = None):
        """
        Initialize the bounce detector service.
        
        Args:
            settings: Dictionary containing bounce detector settings
        """
        # Set default values
        self.settings = settings or {}
        
        # Create bounce detector instance
        self.bounce_detector = BounceDetector(
            court_height=self.settings.get('court_height', 0.0),
            threshold_factor=self.settings.get('threshold_factor', 0.15)
        )
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # State variables
        self.latest_bounce_events: List[BounceEvent] = []
        self.accumulated_tracks: Dict[int, BallTrack] = {}  # Track ID -> BallTrack
        self.court_bounds = self.settings.get('court_bounds', None)
        
    def update_settings(self, settings: Dict[str, Any]):
        """
        Update the bounce detector settings.
        
        Args:
            settings: Dictionary containing bounce detector settings
        """
        self.settings.update(settings)
        
        # Update court height if provided
        if 'court_height' in settings:
            self.bounce_detector.set_court_height(settings['court_height'])
            
        # Update court bounds if provided
        if 'court_bounds' in settings:
            self.court_bounds = settings['court_bounds']
            
    def process_ball_track(self, ball_track: BallTrack) -> List[BounceEvent]:
        """
        Process a ball track and detect bounces.
        
        Args:
            ball_track: Ball track object containing trajectory data
            
        Returns:
            List of detected bounce events
        """
        # Extract positions and timestamps from the ball track
        positions = np.array([point.position for point in ball_track.track_points])
        timestamps = np.array([point.timestamp for point in ball_track.track_points])
        frame_indices = np.array([point.frame_index for point in ball_track.track_points])
        
        # Detect bounces using the bounce detector
        bounce_indices = self.bounce_detector.detect_bounces(positions, timestamps)
        
        # Create bounce events from detected bounces
        bounce_events = []
        for idx, position in bounce_indices:
            # Find the corresponding frame index and timestamp
            if 0 <= idx < len(positions):
                frame_idx = frame_indices[idx]
                timestamp = timestamps[idx]
                
                # Get velocity before and after if available
                velocity_before = None
                velocity_after = None
                
                # Access debug info to get velocities if available
                debug_info = self.bounce_detector.debug_info
                if 'velocities' in debug_info and debug_info['velocities'] is not None:
                    velocities = debug_info['velocities']
                    if idx > 0 and idx < len(velocities):
                        velocity_before = velocities[idx-1]
                    if idx < len(velocities) - 1:
                        velocity_after = velocities[idx]
                
                # Get confidence and detection method if available
                confidence = 0.0
                detection_method = "combined"
                
                if 'scored_candidates' in debug_info and debug_info['scored_candidates'] is not None:
                    # Find the corresponding candidate
                    for candidate_idx, candidate_pos, _, candidate_confidence in debug_info['scored_candidates']:
                        if candidate_idx == idx:
                            confidence = candidate_confidence
                            break
                
                # Create bounce event
                bounce_event = BounceEvent(
                    frame_index=int(frame_idx),
                    timestamp=float(timestamp),
                    position=position,
                    velocity_before=velocity_before,
                    velocity_after=velocity_after,
                    confidence=confidence,
                    detection_method=detection_method
                )
                
                # Check if the bounce is within court bounds
                bounce_event.is_in_bounds = self._check_in_bounds(position)
                
                # Determine court section
                bounce_event.court_section = self._determine_court_section(position)
                
                # Set track ID in metadata
                bounce_event.metadata = {
                    'track_id': ball_track.id
                }
                
                bounce_events.append(bounce_event)
        
        # Update latest bounce events
        self.latest_bounce_events = bounce_events
        
        return bounce_events
    
    def process_multiple_tracks(self, ball_tracks: List[BallTrack]) -> List[BounceEvent]:
        """
        Process multiple ball tracks and detect bounces.
        
        Args:
            ball_tracks: List of ball track objects
            
        Returns:
            List of detected bounce events from all tracks
        """
        all_bounce_events = []
        
        for track in ball_tracks:
            bounce_events = self.process_ball_track(track)
            all_bounce_events.extend(bounce_events)
        
        # Sort bounce events by timestamp
        all_bounce_events.sort(key=lambda x: x.timestamp)
        
        return all_bounce_events
    
    def accumulate_track(self, ball_track: BallTrack):
        """
        Accumulate a ball track for later processing.
        
        Args:
            ball_track: Ball track object to accumulate
        """
        self.accumulated_tracks[ball_track.id] = ball_track
    
    def process_accumulated_tracks(self) -> List[BounceEvent]:
        """
        Process all accumulated tracks and detect bounces.
        
        Returns:
            List of detected bounce events from all accumulated tracks
        """
        all_bounce_events = self.process_multiple_tracks(list(self.accumulated_tracks.values()))
        
        # Clear accumulated tracks
        self.accumulated_tracks.clear()
        
        return all_bounce_events
    
    def get_latest_bounce_events(self) -> List[BounceEvent]:
        """
        Get the latest detected bounce events.
        
        Returns:
            List of latest bounce events
        """
        return self.latest_bounce_events
    
    def _check_in_bounds(self, position: np.ndarray) -> bool:
        """
        Check if a position is within the court bounds.
        
        Args:
            position: 3D position to check
            
        Returns:
            True if the position is within the court bounds, False otherwise
        """
        if self.court_bounds is None:
            return None
        
        # Extract x and y coordinates
        x, y = position[0], position[1]
        
        # Check if within bounds
        min_x, max_x = self.court_bounds['x_range']
        min_y, max_y = self.court_bounds['y_range']
        
        return min_x <= x <= max_x and min_y <= y <= max_y
    
    def _determine_court_section(self, position: np.ndarray) -> Optional[str]:
        """
        Determine the court section where a position is located.
        
        Args:
            position: 3D position to check
            
        Returns:
            Court section name or None if not applicable
        """
        if self.court_bounds is None:
            return None
        
        # Extract x and y coordinates
        x, y = position[0], position[1]
        
        # Get court dimensions
        min_x, max_x = self.court_bounds['x_range']
        min_y, max_y = self.court_bounds['y_range']
        net_y = self.court_bounds.get('net_y', (min_y + max_y) / 2)
        center_x = (min_x + max_x) / 2
        
        # Determine section based on position
        if y <= net_y:
            # Bottom half of the court
            if x <= center_x:
                return "bottom_deuce"
            else:
                return "bottom_ad"
        else:
            # Top half of the court
            if x <= center_x:
                return "top_deuce"
            else:
                return "top_ad" 