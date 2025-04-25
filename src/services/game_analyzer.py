#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tennis Game Analyzer module.
This module contains the GameAnalyzer class, which analyzes the ball tracking data, 
detects bounces, and determines if the ball is in/out of court.
"""

import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import math
import time

from PySide6.QtCore import QObject, Signal, Slot, QTimer

from src.services.geometry.court_frame import CourtFrame
from src.services.ball_tracking.ball_tracking_controller import BallTrackingController, TrackingState


@dataclass
class BounceEvent:
    """Class representing a ball bounce event."""
    position: Tuple[float, float, float]  # x, y, z position in court frame
    velocity: Tuple[float, float, float]  # vx, vy, vz velocity at impact
    timestamp: float  # Timestamp of the bounce
    is_inside_court: bool  # Whether the bounce is inside the court


class GameAnalyzer(QObject):
    """
    Tennis game analyzer class.
    
    This class analyzes the ball tracking data, detects bounces, and determines if the ball is in/out.
    """
    
    # Define signals
    bounce_detected = Signal(object)  # Emits BounceEvent
    ball_position_updated = Signal(float, float, float)  # x, y, z in court frame
    court_position_updated = Signal(float, float, float)  # x, y, z court origin in camera frame
    tracking_reset = Signal()  # Emitted when tracking is reset
    analysis_updated = Signal(dict)  # Emits analysis results dict
    rally_started = Signal()  # Emitted when a new rally is detected
    rally_ended = Signal(int, bool)  # shots count, ended with error

    def __init__(self):
        """Initialize the game analyzer."""
        super(GameAnalyzer, self).__init__()
        
        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Court frame
        self.court_frame = CourtFrame()
        
        # Court dimensions (singles court)
        self.court_width = 8.23  # Width in meters (singles)
        self.court_length = 23.77  # Length in meters
        self.service_line_dist = 6.40  # Distance from baseline to service line (meters)
        
        # Ball tracking controller reference
        self.ball_tracking_controller = None
        
        # Tracking data
        self.positions = []  # List of positions in court frame (x, y, z)
        self.velocities = []  # List of velocities in court frame (vx, vy, vz)
        self.timestamps = []  # List of timestamps
        
        # Bounce detection
        self.bounce_threshold = 0.1  # Minimum z velocity change to detect bounce (m/s²)
        self.bounce_cooldown = 0.5  # Minimum time between bounces (seconds)
        self.last_bounce_time = 0.0  # Time of last detected bounce
        self.bounce_events = []  # List of detected bounces
        
        # Rally detection
        self.rally_in_progress = False
        self.shots_count = 0
        self.start_time = 0.0
        
        # Analysis results
        self.analysis_results = {
            "rally_count": 0,
            "total_shots": 0,
            "average_rally_length": 0.0,
            "longest_rally": 0,
            "bounce_count": 0,
            "in_bounce_count": 0,
            "out_bounce_count": 0,
            "average_speed": 0.0,
            "max_speed": 0.0,
        }
        
        # Setup update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_analysis)
        self.update_timer.start(1000)  # Update analysis every second
        
    def connect_ball_tracking_controller(self, controller: BallTrackingController):
        """
        Connect to the ball tracking controller.
        
        Args:
            controller: BallTrackingController instance
        """
        if controller:
            self.ball_tracking_controller = controller
            
            # Connect signals from ball tracking controller
            controller.ball_3d_position_updated.connect(self._on_ball_position_updated)
            controller.court_position_updated.connect(self._on_court_position_updated)
            controller.tracking_state_changed.connect(self._on_tracking_state_changed)
            
            self.logger.info("Game analyzer connected to ball tracking controller")
            
    def reset(self):
        """Reset the game analyzer."""
        # Clear tracking data
        self.positions = []
        self.velocities = []
        self.timestamps = []
        
        # Reset bounce detection
        self.last_bounce_time = 0.0
        self.bounce_events = []
        
        # Reset rally detection
        self.rally_in_progress = False
        self.shots_count = 0
        
        # Emit reset signal
        self.tracking_reset.emit()
        
        self.logger.info("Game analyzer reset")
        
    def _on_ball_position_updated(self, x: float, y: float, z: float, 
                                 vx: float, vy: float, vz: float, 
                                 timestamp: float):
        """
        Handle ball position updates from the ball tracking controller.
        
        Args:
            x: X coordinate in camera frame
            y: Y coordinate in camera frame
            z: Z coordinate in camera frame
            vx: X velocity in camera frame
            vy: Y velocity in camera frame
            vz: Z velocity in camera frame
            timestamp: Timestamp of the position
        """
        # Transform position to court frame
        court_x, court_y, court_z = self.court_frame.camera_to_court(x, y, z)
        
        # Transform velocity to court frame
        court_vx, court_vy, court_vz = self.court_frame.camera_to_court_vector(vx, vy, vz)
        
        # Add to tracking data
        self.positions.append((court_x, court_y, court_z))
        self.velocities.append((court_vx, court_vy, court_vz))
        self.timestamps.append(timestamp)
        
        # Keep only recent data (last 5 seconds)
        max_buffer = 300  # Maximum buffer size (frames)
        if len(self.positions) > max_buffer:
            self.positions = self.positions[-max_buffer:]
            self.velocities = self.velocities[-max_buffer:]
            self.timestamps = self.timestamps[-max_buffer:]
        
        # Detect bounce
        self._detect_bounce()
        
        # Detect rally events
        self._detect_rally_events()
        
        # Emit position updated signal
        self.ball_position_updated.emit(court_x, court_y, court_z)
        
    def _on_court_position_updated(self, origin_x: float, origin_y: float, origin_z: float,
                                   rotation_matrix: np.ndarray):
        """
        Handle court position updates from the ball tracking controller.
        
        Args:
            origin_x: X coordinate of court origin in camera frame
            origin_y: Y coordinate of court origin in camera frame
            origin_z: Z coordinate of court origin in camera frame
            rotation_matrix: Rotation matrix from court to camera frame
        """
        # Update court frame
        self.court_frame.set_transform(origin_x, origin_y, origin_z, rotation_matrix)
        
        # Emit court position updated signal
        self.court_position_updated.emit(origin_x, origin_y, origin_z)
        
    def _on_tracking_state_changed(self, state: TrackingState):
        """
        Handle tracking state changes from the ball tracking controller.
        
        Args:
            state: New tracking state
        """
        if state == TrackingState.TRACKING_LOST:
            # Ball tracking lost, end rally if in progress
            if self.rally_in_progress:
                self._end_rally(True)  # Ended with error
                
        elif state == TrackingState.RESET:
            # Reset the game analyzer
            self.reset()
            
    def _detect_bounce(self):
        """Detect ball bounces based on velocity changes."""
        # Need at least 3 frames for bounce detection
        if len(self.positions) < 3:
            return
        
        # Get current and previous velocities
        curr_vel = self.velocities[-1]
        prev_vel = self.velocities[-2]
        
        # Get current position and timestamp
        curr_pos = self.positions[-1]
        curr_time = self.timestamps[-1]
        
        # Check if enough time has passed since last bounce
        if curr_time - self.last_bounce_time < self.bounce_cooldown:
            return
            
        # Detect bounce based on vertical velocity change
        # A bounce occurs when vertical velocity changes from negative to positive
        if (prev_vel[2] < -0.5 and  # Was moving downward
            (curr_vel[2] > 0.0 or   # Now moving upward, or
             abs(curr_vel[2] - prev_vel[2]) > self.bounce_threshold)):  # Sudden change
            
            # Calculate bounce position (use the lowest point)
            bounce_pos = self._estimate_bounce_position()
            
            # Check if bounce is inside court
            is_inside = self._is_inside_court(bounce_pos[0], bounce_pos[1])
            
            # Create bounce event
            bounce_event = BounceEvent(
                position=bounce_pos,
                velocity=curr_vel,
                timestamp=curr_time,
                is_inside_court=is_inside
            )
            
            # Add to bounce events
            self.bounce_events.append(bounce_event)
            
            # Update last bounce time
            self.last_bounce_time = curr_time
            
            # Update analysis results
            self.analysis_results["bounce_count"] += 1
            if is_inside:
                self.analysis_results["in_bounce_count"] += 1
            else:
                self.analysis_results["out_bounce_count"] += 1
            
            # Emit bounce detected signal
            self.bounce_detected.emit(bounce_event)
            
            self.logger.debug(f"Bounce detected at {bounce_pos}, inside court: {is_inside}")
            
    def _estimate_bounce_position(self) -> Tuple[float, float, float]:
        """
        Estimate the actual bounce position based on recent positions.
        
        Returns:
            Tuple[float, float, float]: Estimated bounce position (x, y, z)
        """
        # Use the position with the lowest z value in the last few frames
        min_z_idx = -1
        min_z = float('inf')
        
        # Look at the last few positions
        for i in range(max(0, len(self.positions) - 5), len(self.positions)):
            if self.positions[i][2] < min_z:
                min_z = self.positions[i][2]
                min_z_idx = i
        
        # Use that position, but ensure z is close to 0 (court level)
        bounce_pos = list(self.positions[min_z_idx])
        bounce_pos[2] = 0.0  # Set z to court level
        
        return tuple(bounce_pos)
    
    def _is_inside_court(self, x: float, y: float) -> bool:
        """
        Check if a position is inside the tennis court.
        
        Args:
            x: X coordinate in court frame
            y: Y coordinate in court frame
            
        Returns:
            bool: True if inside court, False otherwise
        """
        # Check if within court boundaries
        half_width = self.court_width / 2
        half_length = self.court_length / 2
        
        return -half_width <= x <= half_width and -half_length <= y <= half_length
    
    def _detect_rally_events(self):
        """Detect rally start and end events."""
        # Need at least 10 frames for rally detection
        if len(self.positions) < 10:
            return
            
        # Get current position
        curr_pos = self.positions[-1]
        curr_time = self.timestamps[-1]
        
        # Check if rally in progress
        if not self.rally_in_progress:
            # Detect rally start: ball moving fast and above net
            curr_vel = self.velocities[-1]
            speed = math.sqrt(curr_vel[0]**2 + curr_vel[1]**2 + curr_vel[2]**2)
            
            if speed > 5.0 and curr_pos[2] > 1.0:  # Ball moving fast and above net
                self.rally_in_progress = True
                self.shots_count = 1
                self.start_time = curr_time
                
                # Emit rally started signal
                self.rally_started.emit()
                self.logger.info("Rally started")
        else:
            # Check for ball out of tracking range (rally ended)
            if curr_pos[2] < 0 or abs(curr_pos[0]) > self.court_width or abs(curr_pos[1]) > self.court_length:
                self._end_rally(True)  # Ended with error
            
            # Check for shots
            self._detect_shots()
    
    def _detect_shots(self):
        """Detect shots during a rally."""
        # Need at least 10 frames for shot detection
        if len(self.velocities) < 10:
            return
            
        # Get recent velocities
        curr_vel = self.velocities[-1]
        prev_vel = self.velocities[-5]  # 5 frames ago
        
        # Calculate change in velocity
        dv_x = curr_vel[0] - prev_vel[0]
        dv_y = curr_vel[1] - prev_vel[1]
        dv_z = curr_vel[2] - prev_vel[2]
        
        # Magnitude of velocity change
        dv_magnitude = math.sqrt(dv_x**2 + dv_y**2 + dv_z**2)
        
        # Shot detected if velocity change is significant
        if dv_magnitude > 5.0:  # m/s
            self.shots_count += 1
            
            # Calculate current speed
            speed = math.sqrt(curr_vel[0]**2 + curr_vel[1]**2 + curr_vel[2]**2)
            
            # Update max speed
            if speed > self.analysis_results["max_speed"]:
                self.analysis_results["max_speed"] = speed
                
            self.logger.debug(f"Shot detected, count: {self.shots_count}, speed: {speed:.2f} m/s")
    
    def _end_rally(self, error: bool = False):
        """
        End the current rally.
        
        Args:
            error: Whether the rally ended with an error
        """
        if not self.rally_in_progress:
            return
            
        # Calculate rally duration
        rally_duration = self.timestamps[-1] - self.start_time
        
        # Update analysis results
        self.analysis_results["rally_count"] += 1
        self.analysis_results["total_shots"] += self.shots_count
        
        # Update longest rally
        if self.shots_count > self.analysis_results["longest_rally"]:
            self.analysis_results["longest_rally"] = self.shots_count
        
        # Update average rally length
        self.analysis_results["average_rally_length"] = (
            self.analysis_results["total_shots"] / self.analysis_results["rally_count"]
        )
        
        # Reset rally state
        self.rally_in_progress = False
        
        # Emit rally ended signal
        self.rally_ended.emit(self.shots_count, error)
        
        self.logger.info(f"Rally ended, shots: {self.shots_count}, duration: {rally_duration:.2f}s, error: {error}")
    
    def _update_analysis(self):
        """Update analysis results and emit signal."""
        # Calculate average speed if there are velocities
        if self.velocities:
            # Calculate average of last 10 speeds
            recent_velocities = self.velocities[-min(10, len(self.velocities)):]
            speeds = [math.sqrt(vx**2 + vy**2 + vz**2) for vx, vy, vz in recent_velocities]
            self.analysis_results["average_speed"] = sum(speeds) / len(speeds)
        
        # Emit analysis updated signal
        self.analysis_updated.emit(self.analysis_results)
        
    def get_analysis_results(self) -> Dict[str, Any]:
        """
        Get the current analysis results.
        
        Returns:
            Dict: Analysis results
        """
        return self.analysis_results.copy() 