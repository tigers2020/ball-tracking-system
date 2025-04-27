#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Info View module.
This module contains the InfoView class for displaying tracking information in a widget.
"""

import logging
import os
import re
from collections import deque
from datetime import datetime
from pathlib import Path
import time
import traceback

import numpy as np
from PySide6.QtCore import Qt, Slot, QMetaObject, Signal, QTimer
from PySide6.QtGui import QFont, QColor, QPalette, QPixmap, QIcon, QImage, QPainter, QBrush, QPen
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QGroupBox, QFormLayout, QPushButton, QGridLayout,
    QSizePolicy, QFrame, QScrollArea
)

# Fix import errors using correct paths
from src.utils.constants import LAYOUT
from src.views.visualization.hsv_mask_visualizer import HSVMaskVisualizer
from src.views.visualization.roi_mask_visualizer import ROIMaskVisualizer
from src.views.visualization.hough_circle_visualizer import HoughCircleVisualizer
from src.views.visualization.kalman_path_visualizer import KalmanPathVisualizer
from src.views.widgets.inout_indicator import InOutLED
from src.utils.signal_binder import SignalBinder
from src.models.net_zone import NetZone
from src.utils.format_utils import format_time_delta
from src.views.widgets.panel_label import PanelLabel

# 로거 설정
logger = logging.getLogger("InfoView")

class InfoView(QWidget):
    """
    Widget for displaying detection information in the Stereo Image Player.
    Displays detection rate, 2D pixel coordinates, and 3D position coordinates.
    """
    
    # 시그널 정의
    screenshot_requested = Signal(str)
    capture_requested = Signal(str)
    
    def __init__(self, parent=None):
        """
        Initialize the info view.
        
        Args:
            parent (QWidget, optional): Parent widget
        """
        super().__init__(parent)
        
        # Default values
        self.detection_rate = 0.0  # Percentage
        self.left_pixel_coords = {"x": 0, "y": 0, "r": 0}   # Left camera 2D coordinates
        self.right_pixel_coords = {"x": 0, "y": 0, "r": 0}  # Right camera 2D coordinates
        self.position_coords_3d = {"x": 0.0, "y": 0.0, "z": 0.0}
        
        # ROI values
        self.left_roi = {"x": 0, "y": 0, "width": 0, "height": 0}
        self.right_roi = {"x": 0, "y": 0, "width": 0, "height": 0}
        
        # Kalman state values
        self.left_state = {"pos": (0.0, 0.0), "vel": (0.0, 0.0)}
        self.right_state = {"pos": (0.0, 0.0), "vel": (0.0, 0.0)}
        
        # Controller reference (will be set later)
        self.tracking_controller = None
        self.game_analyzer = None
        
        # Visualizers list
        self._visualizers = []
        
        # 데이터 저장용 변수 초기화
        self.left_coords = None
        self.right_coords = None
        self.position_3d = None
        self.tracking_rate = 0.0
        self.fps = 0.0
        self.frame_count = 0
        self.detection_count = 0
        self.prediction_data = {}
        self.rally_count = 0
        self.last_update_time = time.time()
        self._session_start_time = datetime.now()
        self.total_session_time = "00:00:00"
        self.current_rally_duration = "00:00:00"
        
        # 점수 관련 변수
        self.score = {"player1": 0, "player2": 0}
        self.current_game = 1
        self.games_won = {"player1": 0, "player2": 0}
        
        # 거리 및 속도 관련 변수
        self.ball_speed = 0.0
        self.travel_distance = 0.0
        
        # 유효/아웃 판정 관련 변수
        self.is_in = True
        
        # 코트 위치 데이터
        self.ball_position_percentages = {
            "left": 0,
            "top": 0,
            "right": 0,
            "bottom": 0
        }
        
        # 이전 위치 정보 저장 (속도 계산용)
        self.position_history = deque(maxlen=10)
        
        # UI 업데이트 타이머
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_ui)
        self.update_timer.start(500)  # 500ms마다 UI 업데이트
        
        # UI 업데이트 큐 (최근 업데이트 시간 추적용)
        self.update_queue = deque(maxlen=5)
        
        # Set up UI
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(LAYOUT.MARGIN, LAYOUT.MARGIN, LAYOUT.MARGIN, LAYOUT.MARGIN)
        main_layout.setSpacing(LAYOUT.SPACING)
        
        # Detection rate group
        detection_group = self._create_group_box("Detection Rate")
        detection_layout = QVBoxLayout()
        self.detection_label = QLabel("0.00%")
        self.detection_label.setAlignment(Qt.AlignCenter)
        detection_layout.addWidget(self.detection_label)
        detection_group.setLayout(detection_layout)
        
        # Left 2D pixel coordinates group
        left_pixel_group = QGroupBox("Left Camera (2D)")
        self.left_pixel_x_label, self.left_pixel_y_label, self.left_pixel_r_label = self._setup_coordinate_form(
            left_pixel_group, ["X:", "Y:", "R:"], ["0", "0", "0"]
        )
        
        # Right 2D pixel coordinates group
        right_pixel_group = QGroupBox("Right Camera (2D)")
        self.right_pixel_x_label, self.right_pixel_y_label, self.right_pixel_r_label = self._setup_coordinate_form(
            right_pixel_group, ["X:", "Y:", "R:"], ["0", "0", "0"]
        )
        
        # 3D position coordinates group
        position_group = QGroupBox("3D Ball World Coordinate")
        position_layout = QFormLayout()
        self.position_x_label = QLabel("0.000")
        self.position_y_label = QLabel("0.000")
        self.position_z_label = QLabel("0.000")
        
        # Add IN/OUT indicator
        self.in_out_led = InOutLED()
        
        position_layout.addRow("X (m):", self.position_x_label)
        position_layout.addRow("Y (m):", self.position_y_label)
        position_layout.addRow("Z (m):", self.position_z_label)
        position_layout.addRow("IN/OUT:", self.in_out_led)
        position_group.setLayout(position_layout)
        
        # ROI information group
        roi_group = QGroupBox("ROI")
        self.left_roi_label, self.right_roi_label = self._setup_coordinate_form(
            roi_group, ["Left:", "Right:"], ["(0, 0, 0, 0)", "(0, 0, 0, 0)"]
        )
        
        # Kalman state group
        kalman_group = QGroupBox("Kalman State")
        self.left_state_label, self.right_state_label = self._setup_coordinate_form(
            kalman_group, ["Left:", "Right:"], ["pos=(0, 0), vel=(0, 0)", "pos=(0, 0), vel=(0, 0)"]
        )
        
        # Add all groups to main layout
        main_layout.addWidget(detection_group)
        main_layout.addWidget(left_pixel_group)
        main_layout.addWidget(right_pixel_group)
        main_layout.addWidget(position_group)
        main_layout.addWidget(roi_group)
        main_layout.addWidget(kalman_group)
        
        # Create current time and session time labels
        self.current_time_label = QLabel()
        self.session_time_label = QLabel("Session: 00:00:00")
    
    def _create_group_box(self, title):
        """
        그룹 박스를 생성하는 팩토리 메서드
        
        Args:
            title (str): 그룹 박스 제목
            
        Returns:
            QGroupBox: 생성된 그룹 박스
        """
        group_box = QGroupBox(title)
        return group_box
    
    def _setup_coordinate_form(self, group_box, row_labels, default_values):
        """
        좌표 폼 레이아웃을 설정하는 유틸리티 메서드
        
        Args:
            group_box (QGroupBox): 폼이 추가될 그룹 박스
            row_labels (list): 행 레이블 목록
            default_values (list): 기본값 목록
            
        Returns:
            tuple: 생성된 레이블 위젯 튜플
        """
        form_layout = QFormLayout()
        labels = []
        
        for label, value in zip(row_labels, default_values):
            lbl = QLabel(value)
            form_layout.addRow(label, lbl)
            labels.append(lbl)
        
        group_box.setLayout(form_layout)
        return tuple(labels)
    
    def set_detection_rate(self, rate):
        """탐지율 설정 (0.0 ~ 1.0)"""
        try:
            self.detection_rate = rate
            percentage = int(rate * 100)
            self.detection_label.setText(f"{percentage}%")
            logger.debug(f"Detection rate set: {rate:.2f} ({percentage}%)")
        except Exception as e:
            logger.error(f"Error setting detection rate: {e}")
    
    def set_left_pixel_coords(self, x, y, r=0):
        """
        Set the left camera 2D pixel coordinates.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            r (int, optional): Radius
        """
        logging.debug(f"[InfoView DEBUG] set_left_pixel_coords called with x={x}, y={y}, r={r}")
        self.left_pixel_coords["x"] = x
        self.left_pixel_coords["y"] = y
        self.left_pixel_coords["r"] = r
        self.left_pixel_x_label.setText(str(x))
        self.left_pixel_y_label.setText(str(y))
        self.left_pixel_r_label.setText(str(r))
        self.update()  # Force UI refresh
    
    def set_right_pixel_coords(self, x, y, r=0):
        """
        Set the right camera 2D pixel coordinates.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            r (int, optional): Radius
        """
        logging.debug(f"[InfoView DEBUG] set_right_pixel_coords called with x={x}, y={y}, r={r}")
        self.right_pixel_coords["x"] = x
        self.right_pixel_coords["y"] = y
        self.right_pixel_coords["r"] = r
        self.right_pixel_x_label.setText(str(x))
        self.right_pixel_y_label.setText(str(y))
        self.right_pixel_r_label.setText(str(r))
        self.update()  # Force UI refresh
    
    def set_position_coords(self, x, y, z):
        """
        Set the 3D world position coordinates in meters.
        
        Args:
            x (float): X coordinate in world space (meters)
            y (float): Y coordinate in world space (meters)
            z (float): Z coordinate in world space (meters)
        """
        logging.debug(f"[InfoView DEBUG] set_position_coords called with x={x:.3f}, y={y:.3f}, z={z:.3f}")
        self.position_coords_3d["x"] = x
        self.position_coords_3d["y"] = y
        self.position_coords_3d["z"] = z
        self.position_x_label.setText(f"{x:.3f}")
        self.position_y_label.setText(f"{y:.3f}")
        self.position_z_label.setText(f"{z:.3f}")
        
        # Store for internal use
        self.position_3d = (x, y, z)
        
        # Add debug log to verify coordinates are being set
        logging.info(f"[UI UPDATE] 3D coordinates set to: x={x:.3f}, y={y:.3f}, z={z:.3f}")
        self.update()  # Force UI refresh
    
    def set_left_roi(self, x, y, width, height):
        """
        Set the left ROI coordinates.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            width (int): Width
            height (int): Height
        """
        self.left_roi = {"x": x, "y": y, "width": width, "height": height}
        self.left_roi_label.setText(f"({x}, {y}, {width}, {height})")
    
    def set_right_roi(self, x, y, width, height):
        """
        Set the right ROI coordinates.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            width (int): Width
            height (int): Height
        """
        self.right_roi = {"x": x, "y": y, "width": width, "height": height}
        self.right_roi_label.setText(f"({x}, {y}, {width}, {height})")
    
    def set_left_state(self, pos_x, pos_y, vel_x, vel_y):
        """
        Set the left camera Kalman state.
        
        Args:
            pos_x (float): Position X coordinate
            pos_y (float): Position Y coordinate
            vel_x (float): Velocity X component
            vel_y (float): Velocity Y component
        """
        self.left_state = {"pos": (pos_x, pos_y), "vel": (vel_x, vel_y)}
        self.left_state_label.setText(f"pos=({pos_x:.1f}, {pos_y:.1f}), vel=({vel_x:.1f}, {vel_y:.1f})")
    
    def set_right_state(self, pos_x, pos_y, vel_x, vel_y):
        """
        Set the right camera Kalman state.
        
        Args:
            pos_x (float): Position X coordinate
            pos_y (float): Position Y coordinate
            vel_x (float): Velocity X component
            vel_y (float): Velocity Y component
        """
        self.right_state = {"pos": (pos_x, pos_y), "vel": (vel_x, vel_y)}
        self.right_state_label.setText(f"pos=({pos_x:.1f}, {pos_y:.1f}), vel=({vel_x:.1f}, {vel_y:.1f})")
    
    def clear_info(self):
        """Reset all information displays."""
        self.set_detection_rate(0.0)
        self.set_left_pixel_coords(0, 0, 0)
        self.set_right_pixel_coords(0, 0, 0)
        self.set_position_coords(0.0, 0.0, 0.0)
        self.set_left_roi(0, 0, 0, 0)
        self.set_right_roi(0, 0, 0, 0)
        self.set_left_state(0.0, 0.0, 0.0, 0.0)
        self.set_right_state(0.0, 0.0, 0.0, 0.0)
    
    def connect_tracking_controller(self, controller):
        """
        Connect to a ball tracking controller to receive updates.
        
        Args:
            controller: BallTrackingController instance
        """
        if controller:
            self.tracking_controller = controller
            
            # Make sure we're getting 3D position data if available
            try:
                # Check if controller has 3D position data
                has_3d = hasattr(controller, "get_3d_position") and callable(getattr(controller, "get_3d_position"))
                logging.info(f"Controller has 3D position method: {has_3d}")
            except Exception as e:
                logging.error(f"Error checking controller 3D capabilities: {e}")
            
            # Connect detection_updated signal to handler method directly to ensure proper connection
            try:
                # Connect detection_updated signal to handler method (including 3D coordinate update)
                # Use Qt.QueuedConnection to ensure the slot is executed in the main event loop
                controller.detection_updated.connect(
                    self._on_detection_updated, 
                    Qt.QueuedConnection
                )
                logging.info("Connected BallTrackingController.detection_updated to InfoView._on_detection_updated")
                
                # ROI update signal directly
                if hasattr(controller, "roi_updated"):
                    controller.roi_updated.connect(self._on_roi_updated, Qt.QueuedConnection)
                    logging.info("Connected BallTrackingController.roi_updated to InfoView._on_roi_updated")
                
                # Connect prediction update signal for Kalman state
                if hasattr(controller, "prediction_updated"):
                    # Try direct connection first
                    controller.prediction_updated.connect(self._on_kalman_predicted, Qt.QueuedConnection)
                    logging.info("Connected BallTrackingController.prediction_updated to InfoView._on_kalman_predicted")
                
                # Also try SignalBinder as backup
                SignalBinder.bind(controller, "roi_updated", self, "_on_roi_updated")
                SignalBinder.bind(controller, "prediction_updated", self, "_on_kalman_predicted")
            except Exception as e:
                logging.error(f"Error connecting to BallTrackingController signals: {e}")
                # Fallback connection method
                try:
                    controller.detection_updated.connect(
                        lambda frame_idx, rate, left, right, pos=None: 
                        self._on_detection_updated(frame_idx, rate, left, right, pos),
                        Qt.QueuedConnection
                    )
                    logging.info("Fallback connection to detection_updated established")
                except Exception as e2:
                    logging.error(f"Fallback connection also failed: {e2}")
            
            # Set up visualizers with controller
            self._setup_visualizers_with_controller(controller)
            
            # Print the object ID to verify it's the same instance used elsewhere
            logging.info(f"InfoView instance {id(self)} connected to controller {id(controller)}")
    
    def connect_game_analyzer(self, analyzer):
        """
        Connect to a game analyzer to receive court position updates.
        
        Args:
            analyzer: GameAnalyzer instance
        """
        if analyzer:
            self.game_analyzer = analyzer
            
            # Connect signals using SignalBinder
            signal_mappings = {
                "tracking_updated": self._on_tracking_updated,
                "in_out_detected": self.in_out_led.on_in_out,
                "court_position_updated": self._on_court_position_updated  # Make sure this signal is connected
            }
            
            # Connect all signals directly to ensure proper connection
            try:
                # Use Qt.QueuedConnection to ensure the slot is executed in the main event loop
                from PySide6.QtCore import Qt
                analyzer.tracking_updated.connect(self._on_tracking_updated, Qt.QueuedConnection)
                analyzer.in_out_detected.connect(self.in_out_led.on_in_out, Qt.QueuedConnection)
                if hasattr(analyzer, "court_position_updated"):
                    analyzer.court_position_updated.connect(self._on_court_position_updated, Qt.QueuedConnection)
                
                # Also try using SignalBinder for redundancy
                SignalBinder.bind_all(analyzer, self, signal_mappings)
                
                # Print the object ID to verify it's the same instance used elsewhere
                logging.info(f"InfoView instance {id(self)} connected to analyzer {id(analyzer)}")
                logging.debug(f"Signal connections established: tracking_updated, in_out_detected, court_position_updated")
            except Exception as e:
                logging.error(f"Error connecting to game analyzer signals: {e}")
                # Fallback direct connection for tracking_updated
                try:
                    analyzer.tracking_updated.connect(self._on_tracking_updated, Qt.QueuedConnection)
                    logging.info("Fallback connection to tracking_updated established")
                except Exception as e2:
                    logging.error(f"Fallback connection failed: {e2}")
    
    def _setup_visualizers_with_controller(self, controller):
        """
        Set up visualizers with the connected controller.
        
        Args:
            controller: BallTrackingController instance
        """
        # Create visualizers with the controller
        self._visualizers = [
            HSVMaskVisualizer(controller),
            ROIMaskVisualizer(controller),
            HoughCircleVisualizer(controller),
            KalmanPathVisualizer(controller)
        ]
        
        logging.info(f"Set up {len(self._visualizers)} visualizers")
    
    @Slot(int, float, tuple, tuple, tuple)
    def _on_detection_updated(self, frame_idx, detection_rate, left_coords, right_coords, position_3d):
        """
        탐지 결과 업데이트 시 호출되는 슬롯
        
        매개변수:
            frame_idx (int): 프레임 인덱스
            detection_rate (float): 탐지 성공률 (0.0 ~ 1.0)
            left_coords (tuple): 좌측 카메라 좌표 (x, y, r) 또는 None
            right_coords (tuple): 우측 카메라 좌표 (x, y, r) 또는 None
            position_3d (tuple): 3D 위치 좌표 (x, y, z) 또는 None
        """
        try:
            # 좌표 정보 업데이트
            if left_coords:
                self.set_left_coordinates(left_coords)
                # Direct update of UI labels
                self.set_left_pixel_coords(left_coords[0], left_coords[1], left_coords[2] if len(left_coords) > 2 else 0)
            
            if right_coords:
                self.set_right_coordinates(right_coords)
                # Direct update of UI labels
                self.set_right_pixel_coords(right_coords[0], right_coords[1], right_coords[2] if len(right_coords) > 2 else 0)
            
            if position_3d:
                logger.debug(f"Received 3D position: {position_3d}")
                self.set_position_3d(position_3d)
                # Direct update of UI labels
                self.set_position_coords(position_3d[0], position_3d[1], position_3d[2])
            
            # 탐지율 업데이트
            self.set_detection_rate(detection_rate)
            
            # 30 프레임마다 로그 기록 (디버그 레벨)
            if frame_idx % 30 == 0:
                logger.debug(f"Detection updated - Frame: {frame_idx}, "
                           f"Left: {left_coords}, "
                           f"Right: {right_coords}, "
                           f"3D: {position_3d}")
        
        except Exception as e:
            logger.error(f"Error in detection update handler: {e}")
            logger.debug(traceback.format_exc())
    
    def _on_roi_updated(self, left_roi, right_roi):
        """
        Handle ROI update signal from ball tracking controller.
        
        Args:
            left_roi (dict): Left camera ROI dictionary
            right_roi (dict): Right camera ROI dictionary
        """
        if left_roi:
            self.set_left_roi(left_roi.get('x', 0), left_roi.get('y', 0), 
                              left_roi.get('width', 0), left_roi.get('height', 0))
        else:
            self.set_left_roi(0, 0, 0, 0)
            
        if right_roi:
            self.set_right_roi(right_roi.get('x', 0), right_roi.get('y', 0), 
                               right_roi.get('width', 0), right_roi.get('height', 0))
        else:
            self.set_right_roi(0, 0, 0, 0)
            
        logging.debug(f"ROI info updated: left={left_roi}, right={right_roi}")
    
    def _on_tracking_updated(self, frame_index, timestamp, position, velocity, is_valid):
        """
        Handle tracking updates from game analyzer to get the original world coordinates.
        
        Args:
            frame_index (int): Frame index
            timestamp (float): Timestamp
            position (numpy.ndarray): 3D position in world space [x, y, z]
            velocity (numpy.ndarray): 3D velocity in world space [vx, vy, vz]
            is_valid (bool): Whether the tracking data is valid
        """
        if position is not None and is_valid:
            try:
                # Display original world coordinates directly
                self.set_position_coords(position[0], position[1], position[2])
                
                # Detailed logging for UI displayed coordinates
                logging.info(f"[UI COORD DEBUG] Frame {frame_index} - Displaying world coordinates in UI: "
                           f"x={position[0]:.3f}m, y={position[1]:.3f}m, z={position[2]:.3f}m | "
                           f"velocity: vx={velocity[0]:.3f}, vy={velocity[1]:.3f}, vz={velocity[2]:.3f}")
                
                logging.debug(f"World position updated: ({position[0]:.3f}m, {position[1]:.3f}m, {position[2]:.3f}m)")
            except (IndexError, TypeError, AttributeError) as e:
                logging.error(f"Error processing 3D position from tracking_updated: {e}, position={position}, type={type(position)}")
        elif not is_valid:
            logging.debug(f"Received invalid tracking data for frame {frame_index}")
        else:
            logging.debug(f"Received None position for frame {frame_index}")
    
    def _on_kalman_predicted(self, camera, x, y, vx, vy):
        """
        Handle Kalman prediction update signal from ball tracking controller.
        
        Args:
            camera (str): Camera identifier ('left' or 'right')
            x (float): Position X coordinate
            y (float): Position Y coordinate
            vx (float): Velocity X component
            vy (float): Velocity Y component
        """
        # Update the appropriate state label based on camera
        if camera.lower() == "left":
            self.set_left_state(x, y, vx, vy)
        elif camera.lower() == "right":
            self.set_right_state(x, y, vx, vy)
            
        logging.debug(f"Kalman state updated for {camera} camera: pos=({x:.1f}, {y:.1f}), vel=({vx:.1f}, {vy:.1f})")
    
    def _on_court_position_updated(self, x, y, z):
        """
        Handle court position updates from game analyzer.
        
        Args:
            x (float): X coordinate in court space (meters)
            y (float): Y coordinate in court space (meters)
            z (float): Z coordinate in court space (meters)
        """
        # Update position display
        self.set_position_coords(x, y, z)
        logging.info(f"[COURT COORD] Court position updated: x={x:.3f}, y={y:.3f}, z={z:.3f}")
    
    def get_visualizers(self):
        """
        Get the list of visualizers.
        
        Returns:
            list: List of visualizer objects
        """
        return self._visualizers 

    def _update_ui(self):
        """UI 위젯들의 값을 업데이트"""
        try:
            # 좌표 정보 업데이트
            if self.left_coords:
                x, y, r = self.left_coords
                self.left_pixel_x_label.setText(f"{x:.2f}")
                self.left_pixel_y_label.setText(f"{y:.2f}")
                self.left_pixel_r_label.setText(f"{r:.2f}")
            
            if self.right_coords:
                x, y, r = self.right_coords
                self.right_pixel_x_label.setText(f"{x:.2f}")
                self.right_pixel_y_label.setText(f"{y:.2f}")
                self.right_pixel_r_label.setText(f"{r:.2f}")
            
            if self.position_3d:
                x, y, z = self.position_3d
                self.position_x_label.setText(f"{x:.2f}")
                self.position_y_label.setText(f"{y:.2f}")
                self.position_z_label.setText(f"{z:.2f}")
                logger.debug(f"Updated UI with 3D position: ({x:.2f}, {y:.2f}, {z:.2f})")
            
            # 탐지율 업데이트
            self.set_detection_rate(self.detection_rate)
            
            # 현재 시간 업데이트
            current_time = datetime.now()
            self.current_time_label.setText(current_time.strftime("%H:%M:%S"))
            
            # 세션 시간 업데이트
            elapsed = current_time - self._session_start_time
            self.total_session_time = format_time_delta(elapsed)
            self.session_time_label.setText(f"Session: {self.total_session_time}")
            
            # UI 강제 리페인트
            self._force_repaint()
            
        except Exception as e:
            logger.error(f"Error updating UI: {e}")
            logger.debug(traceback.format_exc())
    
    @Slot()
    def _force_repaint(self):
        """UI를 강제로 다시 그리도록 함"""
        try:
            # Qt의 repaint 메서드 호출
            QMetaObject.invokeMethod(
                self, "repaint", Qt.QueuedConnection
            )
            # 모든 자식 위젯에도 적용
            for child in self.findChildren(QWidget):
                QMetaObject.invokeMethod(
                    child, "repaint", Qt.QueuedConnection
                )
            
            logger.debug("Forced UI repaint")
        except Exception as e:
            logger.error(f"Error forcing repaint: {e}")
            logger.debug(traceback.format_exc())

    def request_screenshot(self, filename_prefix=None):
        """
        스크린샷 요청 함수
        
        매개변수:
            filename_prefix (str): 파일명 접두사
        """
        if not filename_prefix:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_prefix = f"info_view_{timestamp}"
        
        logger.info(f"Screenshot requested with prefix: {filename_prefix}")
        self.capture_requested.emit(filename_prefix)

    def set_left_coordinates(self, coords):
        """
        좌측 카메라 좌표 설정
        
        매개변수:
            coords (tuple): (x, y, r) 좌표값 튜플
        """
        if coords and len(coords) == 3:
            self.left_coords = coords
            logger.debug(f"Left coordinates set: {coords}")
        else:
            logger.warning(f"Invalid left coordinates format: {coords}")
    
    def set_right_coordinates(self, coords):
        """
        우측 카메라 좌표 설정
        
        매개변수:
            coords (tuple): (x, y, r) 좌표값 튜플
        """
        if coords and len(coords) == 3:
            self.right_coords = coords
            logger.debug(f"Right coordinates set: {coords}")
        else:
            logger.warning(f"Invalid right coordinates format: {coords}")
    
    def set_position_3d(self, position):
        """
        3D 위치 설정
        
        매개변수:
            position (tuple): (x, y, z) 3D 위치값 튜플
        """
        try:
            if position and len(position) == 3:
                self.position_3d = position
                logger.info(f"3D position set: {position}")
                
                # 좌표가 복사되는지 확인 (깊은 복사 vs 얕은 복사)
                copied_position = tuple(float(p) for p in position)
                logger.debug(f"3D position copied: {copied_position}")
                
                # 업데이트 큐에 현재 시간 추가
                self.update_queue.append(time.time())
                
                # UI 즉시 업데이트 요청
                QMetaObject.invokeMethod(
                    self, "_update_ui", Qt.QueuedConnection
                )
            else:
                logger.warning(f"Invalid 3D position format: {position}")
        except Exception as e:
            logger.error(f"Error setting 3D position: {e}")
            logger.debug(traceback.format_exc()) 