#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Game Analyzer Controller module.
This module contains the GameAnalyzer class for orchestrating 3D ball tracking and analysis.
"""

import logging
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from PySide6.QtCore import QObject, Signal, Slot, QTimer

from src.services.triangulation_service import TriangulationService
from src.services.kalman3d_service import Kalman3DService
from src.services.bounce_detector import BounceDetector, BounceEvent
from src.services.coordinate_service import CoordinateService
from src.geometry.court_frame import is_point_inside_court, is_net_crossed
from src.utils.constants import ANALYSIS, COURT


@dataclass
class TrackingData:
    """Data class for 3D tracking data."""
    frame_index: int
    timestamp: float
    detection_rate: float
    position_2d_left: Optional[Tuple[float, float]]
    position_2d_right: Optional[Tuple[float, float]]
    position_3d: Optional[np.ndarray]
    velocity_3d: Optional[np.ndarray]
    is_valid: bool = False


class GameAnalyzer(QObject):
    """
    Controller for 3D tennis ball tracking and game analysis.
    """
    
    # Signals for UI updates
    tracking_updated = Signal(int, float, object, object, object)  # frame_idx, timestamp, pos_3d, vel_3d, is_valid
    bounce_detected = Signal(object)  # BounceEvent
    trajectory_updated = Signal(list)  # List of 3D positions
    court_position_updated = Signal(float, float, float)  # x, y, z in court frame
    net_crossed = Signal(bool)  # Direction (True if crossed from baseline to net)
    landing_predicted = Signal(float, float)  # x, y of predicted landing
    in_out_detected = Signal(bool)  # True if in, False if out
    
    def __init__(self, config_manager, event_bus=None):
        """
        Initialize the game analyzer.
        
        Args:
            config_manager: Configuration manager object
            event_bus: Event bus for inter-controller communication
        """
        super(GameAnalyzer, self).__init__()
        
        self.config_manager = config_manager
        self.event_bus = event_bus
        
        # Services
        self.triangulation = TriangulationService()
        self.kalman = Kalman3DService()
        self.bounce_detector = BounceDetector()
        
        # 좌표계 설정 가져오기
        coordinate_config = config_manager.get_section("coordinate_settings", {})
        # coordinate_service 초기화 (설정 파라미터와 함께)
        self.coordinate_service = CoordinateService(coordinate_config)
        
        # State tracking
        self.is_enabled = False
        self.current_frame_index = 0
        self.current_timestamp = 0.0
        self.tracking_history = []
        self.max_history_size = ANALYSIS.MAX_HISTORY_LENGTH  # 4 seconds at 30fps
        self.last_net_cross_frame = -ANALYSIS.NET_CROSSING_FRAMES_THRESHOLD  # Minimum frames between net cross detections
        
        # Initialize services from config
        self._init_from_config()
        
        # Connect to event bus if provided
        if self.event_bus:
            self._connect_to_event_bus()
            
        logging.info("Game analyzer initialized")
        
    def _init_from_config(self):
        """Initialize services from configuration."""
        # Get camera settings for triangulation
        camera_cfg = self.config_manager.get_camera_settings()
        if camera_cfg:
            self.triangulation.set_camera(camera_cfg)
            logging.info("Triangulation service configured from settings")
            
        # Get Kalman filter settings
        kalman_cfg = self.config_manager.get_section("kalman3d", {})
        if kalman_cfg:
            self.kalman.update_params(kalman_cfg)
            logging.info("3D Kalman filter configured from settings")
            
        # Get bounce detector settings
        bounce_cfg = self.config_manager.get_section("bounce_detector", {})
        if bounce_cfg:
            self.bounce_detector.update_params(bounce_cfg)
            logging.info("Bounce detector configured from settings")
            
        # Get history size
        self.max_history_size = self.config_manager.get_value(
            "tracking", "history_size", self.max_history_size)
        
    def _connect_to_event_bus(self):
        """Connect to event bus signals."""
        # Connect to ball detection events
        self.event_bus.subscribe("BALL_DETECTED", self.on_ball_detected)
        
        # Connect to calibration events
        self.event_bus.subscribe("CAMERA_CALIBRATED", self.on_camera_calibrated)
        self.event_bus.subscribe("PNP_CALIBRATED", self.on_pnp_calibrated)
        
        # Connect to frame processing events
        self.event_bus.subscribe("FRAME_PROCESSED", self.on_frame_processed)
        
        logging.info("Game analyzer connected to event bus")
        
    def enable(self, enabled=True):
        """
        Enable or disable the game analyzer.
        
        Args:
            enabled: True to enable, False to disable
        """
        if enabled == self.is_enabled:
            return
            
        self.is_enabled = enabled
        
        if not enabled:
            # Clear state when disabled
            self.reset()
            
        logging.info(f"Game analyzer {'enabled' if enabled else 'disabled'}")
        
    def reset(self):
        """Reset all tracking state."""
        # Reset state tracking
        self.current_frame_index = 0
        self.current_timestamp = 0.0
        self.tracking_history = []
        self.last_net_cross_frame = -ANALYSIS.NET_CROSSING_FRAMES_THRESHOLD
        
        # Reset services
        self.kalman.reset()
        self.bounce_detector.clear_events()
        
        logging.info("Game analyzer reset")
        
    def calibrate_from_pnp(self, left_court_points, right_court_points):
        """
        Calibrate camera poses using PnP with court landmarks.
        
        Args:
            left_court_points: List of 2D points in left image
            right_court_points: List of 2D points in right image
            
        Returns:
            True if calibration successful, False otherwise
        """
        if not left_court_points or not right_court_points:
            logging.error("Cannot calibrate: empty point lists")
            return False
            
        # Convert lists to numpy arrays
        left_points = np.array(left_court_points, dtype=np.float32)
        right_points = np.array(right_court_points, dtype=np.float32)
        
        # Perform PnP calibration
        success = self.triangulation.calibrate_from_pnp(left_points, right_points)
        
        if success:
            logging.info("PnP calibration successful")
            # Emit event to notify other components
            if self.event_bus:
                self.event_bus.emit("PNP_CALIBRATION_COMPLETED", True)
        else:
            logging.error("PnP calibration failed")
            
        return success
        
    @Slot(int, float, float, object, object)
    def on_ball_detected(self, frame_index, timestamp, detection_rate, left_point, right_point):
        """
        Handle ball detection events from BallTrackingController.
        
        Args:
            frame_index: Frame index
            timestamp: Timestamp in seconds
            detection_rate: Detection confidence rate (0-1)
            left_point: (x, y) coordinates in left image, or None
            right_point: (x, y) coordinates in right image, or None
        """
        # Log that we received the call
        logging.debug(f"GameAnalyzer.on_ball_detected called: frame={frame_index}, left={left_point}, right={right_point}")
        
        if not self.is_enabled:
            logging.debug(f"GameAnalyzer is disabled, ignoring detection for frame {frame_index}")
            return
            
        # Update current frame info
        self.current_frame_index = frame_index
        self.current_timestamp = timestamp
        
        # Process 2D detections if both points are available
        if left_point is not None and right_point is not None:
            self._process_detections(frame_index, timestamp, detection_rate, left_point, right_point)
        else:
            # If points are missing, add invalid tracking data to maintain continuity
            tracking_data = TrackingData(
                frame_index=frame_index,
                timestamp=timestamp,
                detection_rate=detection_rate,
                position_2d_left=left_point,
                position_2d_right=right_point,
                position_3d=None,
                velocity_3d=None,
                is_valid=False
            )
            self._add_to_history(tracking_data)
            
            logging.debug(f"Skipping frame {frame_index}: missing detection")
            
    def _process_detections(self, frame_index, timestamp, detection_rate, left_point, right_point):
        """
        Process ball detections in both cameras.
        
        Args:
            frame_index: Frame index
            timestamp: Timestamp in seconds
            detection_rate: Detection confidence rate (0-1)
            left_point: (x, y) coordinates in left image
            right_point: (x, y) coordinates in right image
        """
        # Convert to numpy arrays for triangulation
        left_np = np.array([left_point], dtype=np.float32)
        right_np = np.array([right_point], dtype=np.float32)
        
        # Log coordinates
        logging.debug(f"Triangulating points - left: {left_point}, right: {right_point}")
        
        # Triangulate 3D position
        points_3d = self.triangulation.triangulate_points(left_np, right_np)
        
        if points_3d.size == 0:
            logging.warning(f"Triangulation failed for frame {frame_index}")
            return
            
        position_3d = points_3d[0]
        
        # Log triangulated result
        logging.debug(f"Triangulated 3D point: ({position_3d[0]:.2f}, {position_3d[1]:.2f}, {position_3d[2]:.2f})")
        
        # Initialize confidence score
        confidence_score = detection_rate  # Start with detection rate as base confidence
        
        # Use the coordinate service to validate position and adjust confidence
        # Keep original position_3d intact and only adjust confidence
        _, confidence_score = self.coordinate_service.validate_3d_position(position_3d, confidence_score)
        
        # Save current position and timestamp
        self.prev_position_3d = position_3d.copy()
        self.prev_timestamp = timestamp
        
        # Update position and velocity estimation (Kalman filter) with original position
        logging.debug(f"Updating Kalman with confidence score: {confidence_score:.2f}")
        
        # Pass original position data to Kalman filter (no clamping)
        kalman_result = self.kalman.update(position_3d, confidence=confidence_score)
        position_filtered = kalman_result["position"]
        velocity = kalman_result["velocity"]
        
        # Only log warnings for filtered results exceeding limits (no clamping)
        if position_filtered[2] > ANALYSIS.MAX_VALID_HEIGHT:
            logging.info(f"Filtered height exceeds maximum valid value: {position_filtered[2]:.2f}m")
        
        # Convert to court coordinates
        court_x, court_y, court_z = self.coordinate_service.world_to_court(position_filtered)
        
        # Create tracking data object using filtered position
        tracking_data = TrackingData(
            frame_index=frame_index,
            timestamp=timestamp,
            detection_rate=detection_rate,
            position_2d_left=left_point,
            position_2d_right=right_point,
            position_3d=position_filtered,  # Use filtered position
            velocity_3d=velocity,
            is_valid=True
        )
        
        # Add to history
        self._add_to_history(tracking_data)
        
        # Check for bounce
        bounce_event = self.bounce_detector.check_bounce(
            frame_index, timestamp, position_filtered, velocity)
        
        if bounce_event:
            # Emit bounce event
            self.bounce_detected.emit(bounce_event)
            
            # Also emit in/out signal
            self.in_out_detected.emit(bounce_event.is_inside_court)
            
        # Check for net crossing
        self._check_net_crossing(frame_index, position_filtered)
        
        # Calculate predicted landing position
        landing_position = self.bounce_detector.predict_landing(position_filtered, velocity)
        if landing_position is not None:
            self.landing_predicted.emit(landing_position[0], landing_position[1])
            
        # Emit tracking updates (using filtered values)
        self.tracking_updated.emit(
            frame_index, timestamp, position_filtered, velocity, True)
            
        # Emit court position update
        self.court_position_updated.emit(court_x, court_y, court_z)
            
        # Update trajectory visualization
        if frame_index % ANALYSIS.TRAJECTORY_UPDATE_INTERVAL == 0:  # Update trajectory every N frames to reduce overhead
            self.trajectory_updated.emit(self.get_recent_positions(ANALYSIS.TRAJECTORY_DISPLAY_POINTS))
            
        logging.debug(f"Processed frame {frame_index}: "
                     f"pos=({position_filtered[0]:.2f}, {position_filtered[1]:.2f}, {position_filtered[2]:.2f}), "
                     f"vel=({velocity[0]:.2f}, {velocity[1]:.2f}, {velocity[2]:.2f})")
            
    def _add_to_history(self, tracking_data: TrackingData):
        """
        Add tracking data to history, maintaining maximum size.
        
        Args:
            tracking_data: TrackingData object
        """
        self.tracking_history.append(tracking_data)
        
        # Keep history size limited
        while len(self.tracking_history) > self.max_history_size:
            self.tracking_history.pop(0)
            
    def _check_net_crossing(self, frame_index, position):
        """
        Check if the ball has crossed the net.
        
        Args:
            frame_index: Current frame index
            position: Current 3D position [x, y, z]
        """
        # Need at least two positions to check crossing
        if len(self.tracking_history) < 2 or not self.tracking_history[-2].is_valid:
            return
            
        # Get previous position
        prev_position = self.tracking_history[-2].position_3d
        
        # Check for net crossing
        if (frame_index - self.last_net_cross_frame) > ANALYSIS.NET_CROSSING_FRAMES_THRESHOLD and is_net_crossed(prev_position[1], position[1]):
            # Direction: True if crossed from baseline to net
            direction = prev_position[1] < position[1]
            
            # Emit signal
            self.net_crossed.emit(direction)
            
            # Update last crossing frame
            self.last_net_cross_frame = frame_index
            
            logging.info(f"Net crossed at frame {frame_index}, direction: {'baseline->net' if direction else 'net->baseline'}")
            
    @Slot(dict)
    def on_camera_calibrated(self, camera_settings):
        """
        Handle camera calibration events.
        
        Args:
            camera_settings: Camera settings dictionary
        """
        if camera_settings:
            self.triangulation.set_camera(camera_settings)
            logging.info("Triangulation service updated with new camera calibration")
            
    @Slot(dict, dict)
    def on_pnp_calibrated(self, left_pose, right_pose):
        """
        Handle PnP calibration events.
        
        Args:
            left_pose: Left camera pose (rvec, tvec)
            right_pose: Right camera pose (rvec, tvec)
        """
        # Currently not used directly, as calibration is performed through calibrate_from_pnp()
        logging.info("Received PnP calibration update")
            
    @Slot(int, float)
    def on_frame_processed(self, frame_index, timestamp):
        """
        Handle frame processing events.
        
        Args:
            frame_index: Frame index
            timestamp: Timestamp in seconds
        """
        # Currently not used directly
        pass
        
    def get_recent_positions(self, count=None):
        """
        Get recent 3D positions for trajectory visualization.
        
        Args:
            count: Number of recent positions to return (None for all)
            
        Returns:
            List of 3D position arrays
        """
        valid_history = [td.position_3d for td in self.tracking_history if td.is_valid]
        
        if count is None or count >= len(valid_history):
            return valid_history
            
        return valid_history[-count:]
        
    def get_recent_velocities(self, count=None):
        """
        Get recent 3D velocities.
        
        Args:
            count: Number of recent velocities to return (None for all)
            
        Returns:
            List of 3D velocity arrays
        """
        valid_history = [td.velocity_3d for td in self.tracking_history if td.is_valid]
        
        if count is None or count >= len(valid_history):
            return valid_history
            
        return valid_history[-count:]
        
    def get_bounce_events(self, count=None):
        """
        Get recent bounce events.
        
        Args:
            count: Number of recent bounce events to return (None for all)
            
        Returns:
            List of BounceEvent objects
        """
        return self.bounce_detector.get_bounce_events(count)
        
    def get_latest_position(self):
        """
        Get the latest 3D position.
        
        Returns:
            Latest 3D position or None if not available
        """
        for td in reversed(self.tracking_history):
            if td.is_valid:
                return td.position_3d
        return None
        
    def get_latest_velocity(self):
        """
        Get the latest 3D velocity.
        
        Returns:
            Latest 3D velocity or None if not available
        """
        for td in reversed(self.tracking_history):
            if td.is_valid:
                return td.velocity_3d
        return None
        
    def get_tracking_statistics(self):
        """
        Get tracking statistics.
        
        Returns:
            Dictionary with tracking statistics
        """
        total_frames = len(self.tracking_history)
        valid_frames = sum(1 for td in self.tracking_history if td.is_valid)
        
        if total_frames == 0:
            return {
                "total_frames": 0,
                "valid_frames": 0,
                "valid_percentage": 0.0,
                "bounce_count": 0
            }
            
        bounce_stats = self.bounce_detector.get_bounce_statistics()
        
        return {
            "total_frames": total_frames,
            "valid_frames": valid_frames,
            "valid_percentage": valid_frames / total_frames * 100,
            "bounce_count": bounce_stats["count"]
        }
        
    def update_settings(self, settings_dict):
        """
        Update settings for all services.
        
        Args:
            settings_dict: Dictionary with settings for each service
        """
        # Update triangulation settings
        if "triangulation" in settings_dict:
            self.triangulation.set_camera(settings_dict["triangulation"])
            
        # Update Kalman filter settings
        if "kalman3d" in settings_dict:
            self.kalman.update_params(settings_dict["kalman3d"])
            
        # Update bounce detector settings
        if "bounce_detector" in settings_dict:
            self.bounce_detector.update_params(settings_dict["bounce_detector"])
            
        # Update history size
        if "history_size" in settings_dict:
            self.max_history_size = settings_dict["history_size"]
            
        logging.info("Game analyzer settings updated") 