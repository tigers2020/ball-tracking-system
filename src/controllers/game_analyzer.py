#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Game Analyzer Controller module.
This module contains the GameAnalyzer class for orchestrating 3D ball tracking and analysis.
"""

import logging
import os
import csv
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from PySide6.QtCore import QObject, Signal, Slot, QTimer
from datetime import datetime

from src.controllers.ball_tracking_controller import TrackingState
from src.models.net_zone import NetZone
from src.models.tracking_data_model import TrackingDataModel
from src.services.kalman3d_service import Kalman3DService
from src.services.bounce_detector import BounceDetector, BounceEvent
from src.services.coordinate_service import CoordinateService
from src.geometry.court_frame import is_point_inside_court, is_net_crossed
from src.utils.constants import (
    ANALYSIS,
    COURT,
    LEFT_CAMERA_INDEX,
    RIGHT_CAMERA_INDEX,
    TRACKING_DATA_DIR,
    DEFAULT_STORAGE_PATH
)

# Add mock GameState class
class GameState:
    """
    Mock implementation of GameState to replace the deleted module.
    """
    def __init__(self):
        self.left_score = 0
        self.right_score = 0
        self.serving_side = "left"  # 'left' or 'right'
        self.game_active = False
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing GameState mock")
        
    def reset(self):
        """Reset the game state."""
        self.left_score = 0
        self.right_score = 0
        self.game_active = False
        
    def start_game(self):
        """Start the game."""
        self.game_active = True
        
    def end_game(self):
        """End the game."""
        self.game_active = False
        
    def switch_serving_side(self):
        """Switch the serving side."""
        self.serving_side = "right" if self.serving_side == "left" else "left"


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


# Class for 3D mock triangulation
class TriangulationServiceMock:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing TriangulationServiceMock")
        
    def set_camera(self, camera_index, matrix, distortion, rotation=None, translation=None):
        self.logger.info(f"Mock setting camera {camera_index}")
        # Just store the camera parameters
        pass
        
    def triangulate(self, x_left, y_left, x_right, y_right):
        # Simple mock triangulation - returns average of inputs
        # In a real system, this would use proper stereo triangulation
        x = (x_left + x_right) / 2.0
        y = (y_left + y_right) / 2.0
        z = 1000.0  # Arbitrary depth
        return x, y, z


# Mock implementation of BallPositions class
class BallPositions:
    """
    Simple mock implementation of BallPositions to replace the deleted module.
    """
    def __init__(self):
        self.positions = []
        self.timestamps = []
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing BallPositions mock")
        
    def add_position(self, position, timestamp):
        """Add a position with timestamp."""
        self.positions.append(position)
        self.timestamps.append(timestamp)
        
        # Keep only recent positions
        max_positions = 100  # Adjust as needed
        if len(self.positions) > max_positions:
            self.positions.pop(0)
            self.timestamps.pop(0)
    
    def get_recent_positions(self, count=None):
        """Get recent positions."""
        if count is None or count >= len(self.positions):
            return self.positions
        return self.positions[-count:]
    
    def get_recent_timestamps(self, count=None):
        """Get recent timestamps."""
        if count is None or count >= len(self.timestamps):
            return self.timestamps
        return self.timestamps[-count:]
        
    def clear(self):
        """Clear all positions."""
        self.positions = []
        self.timestamps = []


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
    game_state_updated = Signal(GameState)
    tracking_data_saved = Signal(str)
    
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
        self.kalman = Kalman3DService()
        self.bounce_detector = BounceDetector()
        
        # Initialize coordinate service with the ConfigManager instance
        self.coordinate_service = CoordinateService(self.config_manager)
        
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
        
        # 실제 삼각측량 서비스 생성
        try:
            # 먼저 ConfigManager에서 기존에 생성된 triangulator가 있는지 확인
            self.triangulator = self.config_manager.get_triangulator()
            
            # 기존 triangulator가 있는 경우 사용
            if self.triangulator is not None:
                logging.critical("GameAnalyzer: Using pre-existing triangulator from ConfigManager")
            else:
                # 기존 삼각측량 서비스가 없는 경우 새로 생성
                # 카메라 설정 가져오기
                camera_settings = self.config_manager.get_camera_settings()
                
                # 기본 linear 방식의 triangulator 생성 (DLT 알고리즘 사용)
                triangulation_config = {
                    'method': 'linear',
                    'sub_method': 'dlt'
                }
                
                # 실제 삼각측량 서비스 생성
                from src.core.geometry.triangulation.factory import TriangulationFactory
                self.triangulator = TriangulationFactory.create_triangulator_from_config(
                    triangulation_config,
                    camera_settings
                )
                
                # 생성 성공시 ConfigManager에 등록
                if self.triangulator:
                    self.config_manager.set_triangulator(self.triangulator)
                    logging.critical("GameAnalyzer: Created actual triangulation service with camera settings and registered with ConfigManager")
        except Exception as e:
            # 에러 발생시 로깅 후 모의 서비스로 폴백
            logging.error(f"GameAnalyzer: Error creating triangulator: {e}, falling back to mock")
        self.triangulator = TriangulationServiceMock()
            
        # 삼각측량 서비스 검증
        if hasattr(self.triangulator, 'triangulate'):
            logging.critical("GameAnalyzer: Triangulator has 'triangulate' method")
        else:
            logging.critical("GameAnalyzer: Triangulator missing 'triangulate' method - may cause errors")
            
        # 필요시 triangulate 메서드 래퍼 추가 - 없는 경우에만
        if not hasattr(self.triangulator, 'triangulate') or not callable(getattr(self.triangulator, 'triangulate')):
            def triangulate_wrapper(uL, vL, uR, vR):
                points_2d = [(float(uL), float(vL)), (float(uR), float(vR))]
                logging.critical(f"GameAnalyzer triangulating points: {points_2d}")
                
                # 삼각측량 실행
                point_3d = self.triangulator.triangulate_point(points_2d)
                logging.critical(f"GameAnalyzer triangulation result: {point_3d}")
                
                if point_3d is None:
                    # 실패 시 기본 값 반환
                    logging.critical("GameAnalyzer triangulation failed, returning default coordinates")
                    # 기본값을 numpy 배열로 변환
                    return np.array([(uL + uR) / 2.0, vL, abs(uL - uR) / 10.0])
                
                # 결과를 numpy 배열로 변환
                if isinstance(point_3d, tuple):
                    return np.array([float(point_3d[0]), float(point_3d[1]), float(point_3d[2])])
                return point_3d
                
            # triangulate 메서드 추가
            self.triangulator.triangulate = triangulate_wrapper
            logging.critical("GameAnalyzer: Added triangulate wrapper method to triangulator")
        
        self.tracking_data = []
        self.current_game_state = GameState()
        self.ball_positions = BallPositions()
        
        # Timer for periodic game state updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_game_state)
        self.update_timer.setInterval(200)  # Update every 200ms
        
        # Initialize tracking file
        self.tracking_file = None
        self.csv_writer = None
        
        # Start the update timer
        self.update_timer.start()
        
    def _init_from_config(self):
        """Initialize services from configuration."""
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
        
        if enabled:
            # 디버깅용 코드 제거 - 초기 위치를 중앙으로 설정하지 않음
            logging.info("Game analyzer enabled")
        else:
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
        success = self.triangulator.calibrate_from_pnp(left_points, right_points)
        
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
        
        # Log input coordinates for debugging
        logging.critical(f"[COORD DEBUG] Frame {frame_index} - Input coordinates: left={left_point}, right={right_point}")
        
        # Triangulate 3D position - 직접 값을 넘기도록 수정
        try:
            # 삼각측량 서비스 직접 호출
            points_3d = self.triangulator.triangulate(
                float(left_np[0][0]), 
                float(left_np[0][1]), 
                float(right_np[0][0]), 
                float(right_np[0][1])
            )
            
            logging.critical(f"[COORD DEBUG] Frame {frame_index} - Triangulation result: {points_3d}")
            
            if isinstance(points_3d, np.ndarray) and points_3d.size > 0:
                position_3d = points_3d
                logging.critical(f"[COORD DEBUG] Frame {frame_index} - Using numpy array result")
            elif isinstance(points_3d, tuple) and len(points_3d) >= 3:
                position_3d = np.array(points_3d[:3])
                logging.critical(f"[COORD DEBUG] Frame {frame_index} - Converted tuple to numpy array")
            else:
                logging.warning(f"Triangulation returned unexpected type: {type(points_3d)}")
                return
        except Exception as e:
            logging.error(f"Triangulation error: {e}")
            import traceback
            logging.critical(traceback.format_exc())
            return
            
        if points_3d is None or (isinstance(points_3d, np.ndarray) and points_3d.size == 0):
            logging.warning(f"Triangulation failed for frame {frame_index}")
            return
        
        # Log triangulated result with detailed coordinates
        logging.critical(f"[COORD DEBUG] Frame {frame_index} - Original triangulated (world): "
                    f"x={position_3d[0]:.3f}, y={position_3d[1]:.3f}, z={position_3d[2]:.3f}")
        
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
        
        # Log Kalman filtered coordinates
        logging.critical(f"[COORD DEBUG] Frame {frame_index} - After Kalman (world): "
                    f"x={position_filtered[0]:.3f}, y={position_filtered[1]:.3f}, z={position_filtered[2]:.3f}, "
                    f"vx={velocity[0]:.3f}, vy={velocity[1]:.3f}, vz={velocity[2]:.3f}")
        
        # Only log warnings for filtered results exceeding limits (no clamping)
        if position_filtered[2] > ANALYSIS.MAX_VALID_HEIGHT:
            logging.info(f"Filtered height {position_filtered[2]:.2f}m exceeds threshold (debug-bypass for now)")
        
        # Convert to court coordinates
        court_x, court_y, court_z = self.coordinate_service.world_to_court(position_filtered)
        
        # Log court coordinates (after conversion from world)
        logging.critical(f"[COORD DEBUG] Frame {frame_index} - Court coordinates: "
                    f"x={court_x:.3f}, y={court_y:.3f}, z={court_z:.3f} | "
                    f"World->Court transformation applied")
        
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
            self.triangulator.set_camera(LEFT_CAMERA_INDEX, camera_settings["left_camera_matrix"], camera_settings["left_camera_distortion"])
            self.triangulator.set_camera(RIGHT_CAMERA_INDEX, camera_settings["right_camera_matrix"], camera_settings["right_camera_distortion"])
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
        
    def get_latest_tracking_data(self):
        """
        Get the latest valid tracking data.
        
        Returns:
            Latest TrackingData object or None if not available
        """
        for td in reversed(self.tracking_history):
            if td.is_valid:
                return td
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
        
    def initialize_tracking_file(self):
        # Create directory if it doesn't exist
        if not os.path.exists(TRACKING_DATA_DIR):
            os.makedirs(TRACKING_DATA_DIR)
            
        # Generate filename based on current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(TRACKING_DATA_DIR, f"tracking_data_{timestamp}.csv")
        
        try:
            self.tracking_file = open(filename, 'w', newline='')
            self.csv_writer = csv.writer(self.tracking_file)
            
            # Write header
            self.csv_writer.writerow([
                'timestamp', 'x_3d', 'y_3d', 'z_3d', 
                'x_left', 'y_left', 'x_right', 'y_right',
                'confidence', 'state'
            ])
            
            self.logger.info(f"Tracking file initialized: {filename}")
            return filename
        except Exception as e:
            self.logger.error(f"Failed to initialize tracking file: {e}")
            return None
    
    @Slot(TrackingDataModel)
    def add_tracking_data(self, data):
        """Add new tracking data to the analyzer."""
        self.logger.debug(f"Adding tracking data: {data}")
        self.tracking_data.append(data)
        
        # Update ball positions
        if data.position_3d is not None:
            self.ball_positions.add_position(data.position_3d, data.timestamp)
        
        # Write to CSV if file is open
        if self.csv_writer:
            try:
                x_3d, y_3d, z_3d = data.position_3d if data.position_3d else (None, None, None)
                x_left, y_left = data.position_left if data.position_left else (None, None)
                x_right, y_right = data.position_right if data.position_right else (None, None)
                
                self.csv_writer.writerow([
                    data.timestamp, x_3d, y_3d, z_3d, 
                    x_left, y_left, x_right, y_right,
                    data.confidence, data.state.name
                ])
            except Exception as e:
                self.logger.error(f"Failed to write tracking data: {e}")
    
    @Slot()
    def update_game_state(self):
        """Update the game state based on current tracking data."""
        # Skip if no tracking data available
        if not self.tracking_data:
            return
        
        # Get latest tracking data
        latest_data = self.tracking_data[-1]
        
        # Skip if no 3D position
        if not latest_data.position_3d:
            return
        
        # Update game state
        x, y, z = latest_data.position_3d
        
        # Determine net zone
        net_zone = self._determine_net_zone(x, y, z)
        
        # Check for scoring conditions
        if self._check_scoring_conditions(x, y, z, net_zone):
            # Update score
            self._update_score(net_zone)
        
        # Emit updated game state
        self.game_state_updated.emit(self.current_game_state)
    
    def _determine_net_zone(self, x, y, z):
        """Determine the net zone based on 3D position."""
        # Simple mock implementation
        # In a real system, this would use actual court dimensions
        if z < 0:
            return NetZone.LEFT
        else:
            return NetZone.RIGHT
    
    def _check_scoring_conditions(self, x, y, z, net_zone):
        """Check if scoring conditions are met."""
        # Simple mock implementation
        # In a real system, this would check ball trajectory and court boundaries
        return False
    
    def _update_score(self, net_zone):
        """Update the score based on the net zone."""
        if net_zone == NetZone.LEFT:
            self.current_game_state.right_score += 1
        else:
            self.current_game_state.left_score += 1
    
    def close(self):
        """Close tracking file and clean up resources."""
        if self.tracking_file:
            self.tracking_file.close()
            self.tracking_file = None
            self.csv_writer = None
        
        self.update_timer.stop()
        self.logger.info("GameAnalyzer closed") 