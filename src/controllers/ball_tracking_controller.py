#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ball Tracking Controller module.
This module contains the BallTrackingController class for handling ball tracking functionality.
"""

import logging
import cv2
import numpy as np
import json
import os
import time
import xml.etree.ElementTree as ET
import atexit
from pathlib import Path
from PySide6.QtCore import QObject, Signal, Slot
from typing import Dict, List, Tuple, Optional, Any, Union
import traceback
from enum import Enum

from src.models.tracking_data_model import TrackingDataModel
from src.services.hsv_mask_generator import HSVMaskGenerator
from src.services.roi_computer import ROIComputer
from src.services.circle_detector import CircleDetector
from src.services.kalman_processor import KalmanProcessor
from src.services.data_saver import DataSaver
from src.utils.config_manager import ConfigManager
from src.utils.coord_utils import fuse_coordinates
from src.utils.constants import HSV, ROI, COLOR, STEREO, TRACKING, ANALYSIS, HOUGH
# 누락된 STATUS를 추가하기 위한 임시 클래스
class STATUS(Enum):
    """Constants for status messages"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

# 모의 객체를 위한 임시 클래스 추가
class TriangulationServiceMock:
    """삭제된 TriangulationService의 임시 대체 클래스"""
    
    def __init__(self, camera_settings=None):
        self.is_calibrated = True
        self.cfg = {}
        if camera_settings:
            self.set_camera(camera_settings)
            
    def set_camera(self, settings):
        """카메라 설정 처리"""
        self.cfg = settings
            
    def triangulate(self, uL, vL, uR, vR):
        """간단한 좌표 반환"""
        import numpy as np
        # 기본적인 삼각측량 시뮬레이션: 단순히 평균 위치 반환
        x = (uL + uR) / 2.0
        y = vL  # 수직 좌표는 왼쪽 카메라 값 사용
        z = abs(uL - uR) / 10.0  # 디스패리티에 기반한 깊이 추정
        return np.array([x, y, z])


class TrackingState(Enum):
    """Enum representing the state of ball tracking."""
    TRACKING = 0       # Tracking is active and ball is being detected
    TRACKING_LOST = 1  # Tracking is active but ball is not detected
    RESET = 2          # Tracking has been reset
    DISABLED = 3       # Tracking is disabled


class BallTrackingController(QObject):
    """
    Controller class for ball tracking functionality.
    Handles orchestration of various services for ball detection and tracking.
    """
    
    # Signals
    mask_updated = Signal(np.ndarray, np.ndarray, dict)  # left_mask, right_mask, hsv_settings
    roi_updated = Signal(dict, dict)  # left_roi, right_roi
    detection_updated = Signal(int, float, tuple, tuple, tuple)  # frame_idx, detection_rate, left_coords, right_coords, position_coords
    circles_processed = Signal(np.ndarray, np.ndarray)  # left_circle_image, right_circle_image
    tracking_updated = Signal(float, float, float)  # x, y, z
    prediction_updated = Signal(str, float, float, float, float)  # camera, x, y, vx, vy
    tracking_state_changed = Signal(object)  # TrackingState
    tracking_enabled_changed = Signal(bool)  # enabled flag
    
    def __init__(self, model: Any, config_manager: ConfigManager):
        """
        Initialize the ball tracking controller.
        
        Args:
            model: Data model for ball tracking
            config_manager: Configuration manager for accessing settings
        """
        super(BallTrackingController, self).__init__()
        
        self.model = model
        self.config_manager = config_manager
        
        # Initialize internal state
        self._out_of_bounds_counter = 0
        self._last_3d_coordinates = None
        self._frame_counter = 0
        self._detection_counter = 0
        self.state = TrackingState.DISABLED  # Initial state is disabled
        
        # Initialize internal state variables for compatibility
        self._enabled = False
        self._coordinate_history = {"left": [], "right": []}
        self._world_coordinate_history = []
        
        # Ensure model has required basic attributes
        if not hasattr(self.model, 'left_image'):
            self.model.left_image = None
        if not hasattr(self.model, 'right_image'):
            self.model.right_image = None
        if not hasattr(self.model, 'left_mask'):
            self.model.left_mask = None
        if not hasattr(self.model, 'right_mask'):
            self.model.right_mask = None
        if not hasattr(self.model, 'left_roi'):
            self.model.left_roi = None
        if not hasattr(self.model, 'right_roi'):
            self.model.right_roi = None
        if not hasattr(self.model, 'left_circles'):
            self.model.left_circles = None
        if not hasattr(self.model, 'right_circles'):
            self.model.right_circles = None
        if not hasattr(self.model, 'left_prediction'):
            self.model.left_prediction = None
        if not hasattr(self.model, 'right_prediction'):
            self.model.right_prediction = None
        
        # Set enabled state if model supports it
        if hasattr(self.model, 'is_enabled'):
            self.model.is_enabled = False
        
        # Load settings from configuration
        hsv_values = self.config_manager.get_hsv_settings()
        roi_settings = self.config_manager.get_roi_settings()
        hough_settings = self.config_manager.get_hough_circle_settings()
        
        # Store settings internally for model compatibility
        self._hsv_values = hsv_values
        self._roi_settings = roi_settings
        self._hough_settings = hough_settings
        
        # Update model if it supports these settings
        if hasattr(self.model, 'set_hsv_values') and callable(getattr(self.model, 'set_hsv_values')):
            self.model.set_hsv_values(hsv_values)
        elif hasattr(self.model, 'hsv_values'):
            self.model.hsv_values = hsv_values
            
        if hasattr(self.model, 'set_roi_settings') and callable(getattr(self.model, 'set_roi_settings')):
            self.model.set_roi_settings(roi_settings)
        elif hasattr(self.model, 'roi_settings'):
            self.model.roi_settings = roi_settings
            
        if hasattr(self.model, 'set_hough_settings') and callable(getattr(self.model, 'set_hough_settings')):
            self.model.set_hough_settings(hough_settings)
        elif hasattr(self.model, 'hough_settings'):
            self.model.hough_settings = hough_settings
        
        # Create services with proper configuration
        self.hsv_mask_generator = HSVMaskGenerator(hsv_values)
        self.roi_computer = ROIComputer(roi_settings)
        
        # Initialize CircleDetector with original settings from config_manager
        # This ensures we don't create a new detector for each frame
        self.circle_detector = CircleDetector(hough_settings)
        
        # Get Kalman settings and initialize processor
        self.kalman_settings = self.config_manager.get_kalman_settings()
        self.kalman_processor = KalmanProcessor(self.kalman_settings)
        
        # Initialize triangulation service with camera settings
        self.camera_settings = self.config_manager.get_camera_settings()
        self.triangulator = TriangulationServiceMock(self.camera_settings)
        
        self.data_saver = DataSaver()
        
        # Register finalize_xml with atexit to ensure XML is properly closed at exit
        atexit.register(self.data_saver.finalize_xml)
        
        # Timestamp tracking for Kalman filter dt calculation
        self.last_update_time = {"left": None, "right": None}
        
        # Initialize previous ROI variables
        self.previous_left_roi = None
        self.previous_right_roi = None
        
        # Initialize cropped_images if not present in model
        if not hasattr(self.model, 'cropped_images'):
            self.model.cropped_images = {
                "left": None,
                "right": None
            }
            
    @property
    def is_enabled(self):
        """
        Get the enabled state of ball tracking.
        
        Returns:
            bool: True if ball tracking is enabled, False otherwise
        """
        # Handle different model types
        if hasattr(self.model, 'is_enabled'):
            return self.model.is_enabled
        else:
            # Return our internal enabled flag
            return self._enabled
    
    @property
    def detection_stats(self):
        """
        Get detection statistics.
        
        Returns:
            dict: Detection statistics including tracking status and counters
        """
        return self.model.detection_stats
    
    @property
    def xml_root(self):
        """
        Get XML root element from data saver.
        
        Returns:
            Element: XML root element or None if not initialized
        """
        return self.data_saver.xml_root
    
    @property
    def left_image(self):
        """
        Get the current left image.
        
        Returns:
            numpy.ndarray: Left image or None if not available
        """
        return self.model.left_image
    
    @property
    def right_image(self):
        """
        Get the current right image.
        
        Returns:
            numpy.ndarray: Right image or None if not available
        """
        return self.model.right_image
    
    def set_hsv_values(self, hsv_values):
        """
        Set HSV threshold values for ball detection.
        
        Args:
            hsv_values (dict): Dictionary containing HSV min/max values
        """
        # Store internally for model compatibility
        self._hsv_values = hsv_values
        
        # Update model if it supports this method
        if hasattr(self.model, 'set_hsv_values') and callable(getattr(self.model, 'set_hsv_values')):
            self.model.set_hsv_values(hsv_values)
        elif hasattr(self.model, 'hsv_values'):
            self.model.hsv_values = hsv_values
        
        # Update configuration
        self.config_manager.set_hsv_settings(hsv_values)
        
        # Update service
        self.hsv_mask_generator.update_hsv_values(hsv_values)
        
        logging.info(f"HSV values updated and saved: {hsv_values}")
        
        # Apply updated HSV values if enabled
        if self.is_enabled and (self.model.left_image is not None or self.model.right_image is not None):
            self._process_images()
    
    def set_images(self, left_image, right_image):
        """
        Set the current stereo images for processing.
        
        Args:
            left_image (numpy.ndarray): Left OpenCV image
            right_image (numpy.ndarray): Right OpenCV image
        """
        # Update model - handle both TrackingDataModel and StereoImageModel
        if hasattr(self.model, 'set_images'):
            # TrackingDataModel method
            self.model.set_images(left_image, right_image)
        else:
            # Direct attribute access for StereoImageModel
            self.model.left_image = left_image
            self.model.right_image = right_image
        
        # Process images if enabled
        if self.is_enabled:
            self._process_images()
    
    def enable(self, enabled: bool = True) -> None:
        """
        Enable or disable ball tracking.
        
        Args:
            enabled (bool): True to enable tracking, False to disable
        """
        # 모델 속성이 존재하는지 확인
        if enabled:
            has_left_image = hasattr(self.model, 'left_image')
            has_right_image = hasattr(self.model, 'right_image')
            
            if not has_left_image or not has_right_image:
                logging.warning("Cannot enable ball tracking: model missing image attributes")
                return
                
            # 이미지가 존재하는지 확인
            if self.model.left_image is None and self.model.right_image is None:
                logging.warning("Cannot enable ball tracking: no images available")
                return
        
        # 상태 업데이트
        self._enabled = enabled
        
        # 모델 상태 업데이트
        if hasattr(self.model, 'set_tracking_enabled'):
            self.model.set_tracking_enabled(enabled)
        else:
            self.model.tracking_enabled = enabled
        
        # 버튼 상태 업데이트 신호 발생
        self.tracking_enabled_changed.emit(enabled)
        
        # Emit tracking state changed signal
        if enabled:
            self.tracking_state_changed.emit(TrackingState.TRACKING)
        else:
            self.tracking_state_changed.emit(TrackingState.DISABLED)
        
        # 추적 활성화 시에만 이미지 처리 시작
        if enabled:
            logging.info(f"Ball tracking enabled")
            # 이미지 처리 시작
            self._process_images()
        else:
            logging.info("Ball tracking disabled")
            # 마스크 초기화
            if hasattr(self.model, 'left_mask'):
                self.model.left_mask = None
            if hasattr(self.model, 'right_mask'):
                self.model.right_mask = None
            if hasattr(self, 'mask_updated'):
                self.mask_updated.emit(None, None, None)
    
    def _process_images(self):
        """
        Process the current images from the model.
        Detects circles, applies Kalman filtering, and updates the visualizations.
        """
        try:
            from src.views.visualization import VisualizerFactory
            
            # Get current values for processing
            left_image = self.model.left_image
            right_image = self.model.right_image
            
            # 현재 프레임이 없으면 처리하지 않음
            if left_image is None or right_image is None:
                return
            
            # Get current settings
            hsv_values = self.get_hsv_values()
            roi_settings = self.get_roi_settings()
            
            # 1. Apply HSV threshold to create masks
            left_mask, right_mask, pixel_count_left, pixel_count_right = self._apply_hsv_threshold(
                left_image, right_image, hsv_values
            )
            
            # Store masks for later visualization
            self.left_mask = left_mask
            self.right_mask = right_mask
            
            # 2. Compute ROIs (if enabled)
            self.left_roi, self.right_roi = self._compute_rois(
                left_image, right_image, roi_settings
            )
            
            # 3. Crop images to ROI if enabled
            left_cropped_image, right_cropped_image = None, None
            left_cropped_mask, right_cropped_mask = None, None
            
            if roi_settings.get('enabled', False):
                # Crop both images to ROIs for detection
                left_cropped_image, left_cropped_mask = self._crop_to_roi(
                    left_image, left_mask, self.left_roi)
                right_cropped_image, right_cropped_mask = self._crop_to_roi(
                    right_image, right_mask, self.right_roi)
                
                # Debug the cropped images and masks
                logging.debug(f"Left cropped image shape: {left_cropped_image.shape if left_cropped_image is not None else None}")
                logging.debug(f"Right cropped image shape: {right_cropped_image.shape if right_cropped_image is not None else None}")
            
            # Detect circles - use cropped images if available, otherwise use full images
            if roi_settings.get('enabled', False) and left_cropped_image is not None and right_cropped_image is not None:
                # Detect circles in cropped images within ROIs
                left_circles, right_circles = self._detect_circles_in_cropped_images(
                    left_cropped_image, right_cropped_image, 
                    left_cropped_mask, right_cropped_mask, 
                    self.left_roi, self.right_roi
                )
            else:
                # Fall back to original approach for full-image detection
                left_circles, right_circles = self._detect_circles(
                    left_image, right_image, left_mask, right_mask, roi_settings
                )
            
            # Ensure we have at least empty lists, not None
            left_circles = [] if left_circles is None else left_circles  
            right_circles = [] if right_circles is None else right_circles
            
            # Set the detected circles to the model
            self.model.left_circles = left_circles
            self.model.right_circles = right_circles
            
            # Update circle lists for internal use
            self.left_circles = left_circles
            self.right_circles = right_circles
            
            # Apply Kalman filtering (if enabled)
            left_prediction, right_prediction = self._process_predictions(left_circles, right_circles)
            
            # Set the predictions to the model
            self.model.left_prediction = left_prediction
            self.model.right_prediction = right_prediction
            
            # Create copies for visualization
            left_viz_image = left_image.copy()
            right_viz_image = right_image.copy()
            
            # Create visualizers
            left_visualizer = VisualizerFactory.create(backend="opencv")
            right_visualizer = VisualizerFactory.create(backend="opencv")
            
            # 1. Always draw ROI on images (regardless of whether ROI is enabled)
            if self.left_roi:
                # Use the visualizer to draw ROI
                left_viz_image = left_visualizer.draw_roi(
                    left_viz_image, 
                    self.left_roi, 
                    color=COLOR.GREEN,
                    thickness=TRACKING.ROI_THICKNESS,
                    show_center=True
                )
                logging.debug(f"Drew left ROI: {self.left_roi}")
            
            if self.right_roi:
                # Use the visualizer to draw ROI
                right_viz_image = right_visualizer.draw_roi(
                    right_viz_image, 
                    self.right_roi, 
                    color=COLOR.GREEN,
                    thickness=TRACKING.ROI_THICKNESS,
                    show_center=True
                )
                logging.debug(f"Drew right ROI: {self.right_roi}")
            
            # 2. Draw circles on images if detected
            if left_circles:
                # Use the visualizer to draw circles
                left_viz_image = left_visualizer.draw_circles(
                    left_viz_image, 
                    left_circles, 
                    color=TRACKING.MAIN_CIRCLE_COLOR,
                    thickness=TRACKING.CIRCLE_THICKNESS,
                    label_circles=True
                )
                logging.debug(f"Drew {len(left_circles)} circles on left image")
            
            if right_circles:
                # Use the visualizer to draw circles
                right_viz_image = right_visualizer.draw_circles(
                    right_viz_image, 
                    right_circles, 
                    color=TRACKING.MAIN_CIRCLE_COLOR,
                    thickness=TRACKING.CIRCLE_THICKNESS,
                    label_circles=True
                )
                logging.debug(f"Drew {len(right_circles)} circles on right image")
            
            # 3. Draw Kalman prediction and trajectory on images if available
            if left_prediction:
                # Extract current position from circles and predicted position from Kalman
                current_pos = (int(left_circles[0][0]), int(left_circles[0][1])) if left_circles and len(left_circles) > 0 else None
                pred_pos = (int(left_prediction[0]), int(left_prediction[1]))
                
                # Draw prediction arrow with thicker line
                left_viz_image = left_visualizer.draw_prediction(
                    left_viz_image, 
                    current_pos, 
                    pred_pos, 
                    arrow_color=TRACKING.PREDICTION_ARROW_COLOR,
                    thickness=TRACKING.PREDICTION_THICKNESS,
                    draw_uncertainty=True,
                    uncertainty_radius=TRACKING.UNCERTAINTY_RADIUS
                )
                
                # Draw trajectory if available
                left_history = self.kalman_processor.get_position_history("left")
                if left_history and len(left_history) > 1:
                    left_viz_image = left_visualizer.draw_trajectory(
                        left_viz_image, 
                        left_history, 
                        color=(255, 255, 0),  # Yellow trajectory
                        thickness=5,
                        max_points=20
                    )
                    logging.debug(f"Drew left trajectory with {len(left_history)} points")
                    
            if right_prediction:
                # Extract current position from circles and predicted position from Kalman
                current_pos = (int(right_circles[0][0]), int(right_circles[0][1])) if right_circles and len(right_circles) > 0 else None
                pred_pos = (int(right_prediction[0]), int(right_prediction[1]))
                
                # Draw prediction arrow with thicker line
                right_viz_image = right_visualizer.draw_prediction(
                    right_viz_image, 
                    current_pos, 
                    pred_pos, 
                    arrow_color=(0, 255, 255),  # Yellow-green arrow
                    thickness=4,
                    draw_uncertainty=True,
                    uncertainty_radius=20
                )
                
                # Draw trajectory if available
                right_history = self.kalman_processor.get_position_history("right")
                if right_history and len(right_history) > 1:
                    right_viz_image = right_visualizer.draw_trajectory(
                        right_viz_image, 
                        right_history, 
                        color=(255, 255, 0),  # Yellow trajectory
                        thickness=5,
                        max_points=20
                    )
                    logging.debug(f"Drew right trajectory with {len(right_history)} points")
            
            # 4. Emit processed images signal with visualization
            self.circles_processed.emit(left_viz_image, right_viz_image)
            
            # 5. Emit mask and ROI signals (these will be drawn by the ImageViewWidget)
            # Include HSV settings for dynamic color visualization
            self.mask_updated.emit(self.left_mask, self.right_mask, hsv_values)
            
            # 6. Always emit ROI signal regardless of whether it's enabled for detection
            self.roi_updated.emit(self.left_roi, self.right_roi)
            
            # Update internal predictions
            self.left_prediction = left_prediction
            self.right_prediction = right_prediction
            
            # Increment frame count for detection rate calculation
            self._frame_counter += 1
            if left_circles is not None and len(left_circles) > 0:
                self._detection_counter += 1
            
            # Get the current frame index and timestamp
            frame_idx = self.model.current_frame_index if hasattr(self.model, 'current_frame_index') else 0
            timestamp = time.time()
            
            # Get best circles (or None if not found)
            left_best = self._get_best_circle(left_circles, "left") if left_circles is not None else None
            right_best = self._get_best_circle(right_circles, "right") if right_circles is not None else None
            
            # Update detection information to be displayed in the UI
            self._update_detection_signal(frame_idx, timestamp, left_best, right_best)
            
            # Emit signal for image processing complete
            if hasattr(self, 'image_processed') and self.image_processed is not None:
                self.image_processed.emit()
            
        except Exception as e:
            logging.error(f"Error in _process_images: {e}")
            import traceback
            logging.error(traceback.format_exc())
    
    def _apply_hsv_threshold(self, left_image, right_image, hsv_values):
        """
        Apply HSV threshold to create binary masks from left and right images.
        
        Args:
            left_image (numpy.ndarray): Left input image
            right_image (numpy.ndarray): Right input image
            hsv_values (dict): Dictionary containing HSV min/max values
            
        Returns:
            tuple: (left_mask, right_mask) - Binary masks for both images,
                   (pixel_count_left, pixel_count_right) - White pixel counts for both masks
        """
        # Process left image
        left_hsv = cv2.cvtColor(left_image, cv2.COLOR_BGR2HSV)
        left_mask = cv2.inRange(left_hsv, 
                              (hsv_values.get('h_min', HSV.h_min), 
                               hsv_values.get('s_min', HSV.s_min), 
                               hsv_values.get('v_min', HSV.v_min)),
                              (hsv_values.get('h_max', HSV.h_max), 
                               hsv_values.get('s_max', HSV.s_max), 
                               hsv_values.get('v_max', HSV.v_max)))
        
        # Process right image
        right_hsv = cv2.cvtColor(right_image, cv2.COLOR_BGR2HSV)
        right_mask = cv2.inRange(right_hsv,
                               (hsv_values.get('h_min', HSV.h_min), 
                                hsv_values.get('s_min', HSV.s_min), 
                                hsv_values.get('v_min', HSV.v_min)),
                               (hsv_values.get('h_max', HSV.h_max), 
                                hsv_values.get('s_max', HSV.s_max), 
                                hsv_values.get('v_max', HSV.v_max)))
        
        # Count white pixels in each mask
        pixel_count_left = cv2.countNonZero(left_mask)
        pixel_count_right = cv2.countNonZero(right_mask)
        
        return left_mask, right_mask, pixel_count_left, pixel_count_right
    
    def _compute_rois(self, left_image, right_image, roi_settings):
        """
        컴퓨터 ROI 및 적용 마스크
        
        Args:
            left_image: 왼쪽 카메라 이미지
            right_image: 오른쪽 카메라 이미지
            roi_settings: ROI 설정
            
        Returns:
            left_roi, right_roi: 왼쪽 및 오른쪽 ROI 영역 튜플 (x, y, width, height)
        """
        # ROI 계산
        left_roi = self.roi_computer.compute_roi(self.left_mask, left_image)
        right_roi = self.roi_computer.compute_roi(self.right_mask, right_image)
        
        # 이전 ROI가 있는 경우 안정성 검사 추가
        if hasattr(self, 'previous_left_roi') and self.previous_left_roi is not None:
            # 이전 ROI의 중심점 계산
            x = self.previous_left_roi.get('x', 0)
            y = self.previous_left_roi.get('y', 0)
            w = self.previous_left_roi.get('width', ROI.DEFAULT_WIDTH)
            h = self.previous_left_roi.get('height', ROI.DEFAULT_HEIGHT)
            prev_left_x = x + w // 2
            prev_left_y = y + h // 2
            
            # 현재 ROI의 중심점 계산
            x = left_roi.get('x', 0)
            y = left_roi.get('y', 0)
            w = left_roi.get('width', ROI.DEFAULT_WIDTH)
            h = left_roi.get('height', ROI.DEFAULT_HEIGHT)
            curr_left_x = x + w // 2
            curr_left_y = y + h // 2
            
            # 중심점의 변화 계산
            dx = abs(curr_left_x - prev_left_x)
            dy = abs(curr_left_y - prev_left_y)
            
            # 변화가 너무 큰 경우 (이미지 너비의 30% 초과) 제한
            # 급격한 ROI 이동 방지
            frame_width = left_image.shape[1]
            frame_height = left_image.shape[0]
            
            if dx > frame_width * TRACKING.MAX_ROI_JUMP_FACTOR or dy > frame_height * TRACKING.MAX_ROI_JUMP_FACTOR:
                logging.warning(f"Large ROI jump detected: dx={dx}px, dy={dy}px. Limiting movement.")
                
                # 너무 큰 이동 제한 - 이전 ROI와 현재 ROI의 중간으로 제한
                max_dx = int(frame_width * TRACKING.MAX_ROI_JUMP_FACTOR)
                max_dy = int(frame_height * TRACKING.MAX_ROI_JUMP_FACTOR)
                
                # 새 중심점 계산
                new_left_x = prev_left_x + np.clip(curr_left_x - prev_left_x, -max_dx, max_dx)
                new_left_y = prev_left_y + np.clip(curr_left_y - prev_left_y, -max_dy, max_dy)
                
                # 새 ROI 계산
                half_width = w // 2
                half_height = h // 2
                left_roi = {
                    'x': max(0, new_left_x - half_width),
                    'y': max(0, new_left_y - half_height),
                    'width': min(w, frame_width - (new_left_x - half_width)),
                    'height': min(h, frame_height - (new_left_y - half_height))
                }
                
                # 오른쪽 ROI도 동일하게 처리
                if hasattr(self, 'previous_right_roi') and self.previous_right_roi is not None:
                    # 우측 ROI 중심점 계산 
                    x = self.previous_right_roi.get('x', 0)
                    y = self.previous_right_roi.get('y', 0)
                    w = self.previous_right_roi.get('width', ROI.DEFAULT_WIDTH)
                    h = self.previous_right_roi.get('height', ROI.DEFAULT_HEIGHT)
                    prev_right_x = x + w // 2
                    prev_right_y = y + h // 2
                    
                    x = right_roi.get('x', 0)
                    y = right_roi.get('y', 0)
                    w = right_roi.get('width', ROI.DEFAULT_WIDTH)
                    h = right_roi.get('height', ROI.DEFAULT_HEIGHT)
                    curr_right_x = x + w // 2
                    curr_right_y = y + h // 2
                    
                    # 중심점 제한
                    new_right_x = prev_right_x + np.clip(curr_right_x - prev_right_x, -max_dx, max_dx)
                    new_right_y = prev_right_y + np.clip(curr_right_y - prev_right_y, -max_dy, max_dy)
                    
                    # 새 ROI 계산
                    half_width = w // 2
                    half_height = h // 2
                    right_roi = {
                        'x': max(0, new_right_x - half_width),
                        'y': max(0, new_right_y - half_height),
                        'width': min(w, frame_width - (new_right_x - half_width)),
                        'height': min(h, frame_height - (new_right_y - half_height))
                    }
        
        # 현재 ROI 저장
        self.previous_left_roi = left_roi
        self.previous_right_roi = right_roi
        
        return left_roi, right_roi
    
    def _apply_roi_mask(self, mask, roi):
        """
        Apply ROI mask to the current mask.
        
        Args:
            mask: Input mask
            roi: Region of interest
            
        Returns:
            Masked image with areas outside ROI set to 0
        """
        if mask is None or roi is None:
            return mask
            
        # Create empty mask of same size
        h, w = mask.shape[:2]
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Get ROI coordinates
        x, y = roi.get('x', 0), roi.get('y', 0)
        width, height = roi.get('width', 0), roi.get('height', 0)
        
        # Draw filled white rectangle in ROI area
        cv2.rectangle(roi_mask, (x, y), (x + width, y + height), 255, -1)
        
        # Apply mask - only keep pixels in the ROI area
        result = cv2.bitwise_and(mask, roi_mask)
        
        return result
    
    def _crop_to_roi(self, image, mask, roi):
        """
        Crop image and mask to the specified Region of Interest.
        
        Args:
            image: Input image
            mask: Input mask
            roi: Region of interest dictionary with x, y, width, height
            
        Returns:
            Tuple of (cropped_image, cropped_mask) or (None, None) if ROI is invalid
        """
        if image is None or roi is None:
            return None, None
            
        # Validate ROI
        if not all(k in roi for k in ['x', 'y', 'width', 'height']):
            logging.warning("Invalid ROI format for cropping")
            return None, None
            
        # Get ROI coordinates
        x, y = roi.get('x', 0), roi.get('y', 0)
        width, height = roi.get('width', 0), roi.get('height', 0)
        
        # Make sure coordinates are valid
        if width <= 0 or height <= 0:
            logging.warning(f"Invalid ROI dimensions: {width}x{height}")
            return None, None
            
        # Check if ROI is within image bounds
        if image is not None:
            img_height, img_width = image.shape[:2]
            if x < 0 or y < 0 or x + width > img_width or y + height > img_height:
                # Adjust ROI to fit within image bounds
                x = max(0, x)
                y = max(0, y)
                width = min(width, img_width - x)
                height = min(height, img_height - y)
                
                if width <= 0 or height <= 0:
                    logging.warning("ROI outside image bounds")
                    return None, None
                    
                logging.debug(f"Adjusted ROI to fit image: {x}, {y}, {width}, {height}")
        else:
            logging.warning("Cannot crop to ROI: image is None")
            return None, None
        
        # Crop image
        cropped_image = image[y:y+height, x:x+width]
        
        # Crop mask if provided
        cropped_mask = None
        if mask is not None:
            cropped_mask = mask[y:y+height, x:x+width]
            
        return cropped_image, cropped_mask
    
    def _detect_circles(self, left_image, right_image, left_mask, right_mask, roi_settings):
        """
        Detect circles in the masked images within ROIs.
        
        Args:
            left_image (numpy.ndarray): Left input image
            right_image (numpy.ndarray): Right input image
            left_mask (numpy.ndarray): Left binary mask
            right_mask (numpy.ndarray): Right binary mask
            roi_settings (dict): Dictionary containing ROI settings
            
        Returns:
            tuple: (left_circles, right_circles) - Lists of detected circles
        """
        # Use existing CircleDetector instance - Never recreate for each frame
        # Only update settings if roi_settings changed
        current_hough_settings = self._hough_settings.copy()
        
        # If adaptive is enabled in settings, let the CircleDetector handle it
        # but don't create a new detector instance each time
        adaptive_enabled = current_hough_settings.get('adaptive', False)
        
        # Ensure min_radius is small enough for tennis ball detection
        # This fixes the problem where calculated min_radius is too large (16px)
        # and excludes real tennis balls (13-15px)
        if adaptive_enabled:
            # Keep user-defined min_radius as lower bound, don't let adaptive override entirely
            if 'min_radius' in current_hough_settings:
                current_hough_settings['min_radius'] = max(8, min(current_hough_settings['min_radius'], 14))
            
            # Ensure param2 (accumulator threshold) isn't too high
            if 'param2' in current_hough_settings:
                current_hough_settings['param2'] = min(current_hough_settings['param2'], 25)
        
        # Update detector with modified settings if needed
        if current_hough_settings != self._hough_settings:
            self.circle_detector.update_settings(current_hough_settings)
            logging.debug(f"Updated circle detector settings: {current_hough_settings}")
            # Store the updated settings
            self._hough_settings = current_hough_settings.copy()
        
        # Detect circles using the existing detector instance
        left_result = self.circle_detector.detect_circles(
            img=left_image, 
            mask=left_mask, 
            roi=self.left_roi if roi_settings.get("enabled", False) else None,
            side="left"  # Add side parameter for better logging
        )
        
        right_result = self.circle_detector.detect_circles(
            img=right_image, 
            mask=right_mask, 
            roi=self.right_roi if roi_settings.get("enabled", False) else None,
            side="right"  # Add side parameter for better logging
        )
        
        # Extract circles from results and ensure they are not None
        left_circles = left_result['circles'] if left_result['circles'] is not None else []
        right_circles = right_result['circles'] if right_result['circles'] is not None else []
        
        return left_circles, right_circles
    
    def _process_predictions(self, left_circles, right_circles):
        """
        Process detected circles through Kalman filter to generate predictions for current frame.
        
        Args:
            left_circles (list): List of circles detected in left image
            right_circles (list): List of circles detected in right image
            
        Returns:
            tuple: (left_prediction, right_prediction) - Updated Kalman predictions
        """
        # Skip if Kalman processor is disabled
        if not self.kalman_settings.get('enabled', True):
            # Return None for predictions
            return None, None
        
        # Get adaptive detection settings
        confidence_threshold = self.kalman_settings.get('confidence_threshold', 0.7)
        distance_threshold = self.kalman_settings.get('distance_threshold', 100)
        
        # Get the best circle from each image
        left_best = self._get_best_circle(left_circles, "left")
        right_best = self._get_best_circle(right_circles, "right")
        
        # Process left prediction
        left_prediction = None
        if left_best:
            # Extract coordinates and radius
            x, y, r = left_best
            
            # Check if ROI is enabled and if we have valid ROI coordinates
            if self.get_roi_settings().get('enabled', False) and self.left_roi:
                # If circles are already in full-image coordinates (from _detect_circles_in_cropped_images),
                # we don't need to adjust them further
                pass
                
            # Calculate time since last update for this camera
            current_time = time.time()
            if self.last_update_time["left"] is None:
                dt = 1/30.0  # Default 30 FPS if first update
            else:
                dt = current_time - self.last_update_time["left"]
                dt = max(0.01, min(dt, 0.5))  # Limit dt to reasonable range

            # Update last update time
            self.last_update_time["left"] = current_time
                
            # Update Kalman filter with new measurement
            left_prediction = self.kalman_processor.update(
                camera="left",
                x=float(x),
                y=float(y),
                dt=dt
            )
            
            # Emit prediction updated signal
            if left_prediction is not None:
                self.prediction_updated.emit("left", left_prediction[0], left_prediction[1], 
                                            left_prediction[2], left_prediction[3])
            
        else:
            # If no circle detected, just get prediction based on previous state
            left_prediction = self.kalman_processor.get_prediction("left")
            
            # Emit prediction updated signal if available
            if left_prediction is not None:
                self.prediction_updated.emit("left", left_prediction[0], left_prediction[1], 
                                            left_prediction[2], left_prediction[3])
            
        # Process right prediction
        right_prediction = None
        if right_best:
            # Extract coordinates and radius
            x, y, r = right_best
            
            # Check if ROI is enabled and if we have valid ROI coordinates
            if self.get_roi_settings().get('enabled', False) and self.right_roi:
                # If circles are already in full-image coordinates (from _detect_circles_in_cropped_images),
                # we don't need to adjust them further
                pass
                
            # Calculate time since last update for this camera
            current_time = time.time()
            if self.last_update_time["right"] is None:
                dt = 1/30.0  # Default 30 FPS if first update
            else:
                dt = current_time - self.last_update_time["right"]
                dt = max(0.01, min(dt, 0.5))  # Limit dt to reasonable range
                
            # Update last update time
            self.last_update_time["right"] = current_time
                
            # Update Kalman filter with new measurement
            right_prediction = self.kalman_processor.update(
                camera="right",
                x=float(x),
                y=float(y),
                dt=dt
            )
            
            # Emit prediction updated signal
            if right_prediction is not None:
                self.prediction_updated.emit("right", right_prediction[0], right_prediction[1], 
                                            right_prediction[2], right_prediction[3])
        else:
            # If no circle detected, just get prediction based on previous state
            right_prediction = self.kalman_processor.get_prediction("right")
            
            # Emit prediction updated signal if available
            if right_prediction is not None:
                self.prediction_updated.emit("right", right_prediction[0], right_prediction[1], 
                                            right_prediction[2], right_prediction[3])
        
        return left_prediction, right_prediction
    
    def _get_best_circle(self, circles, side):
        """
        Get the best circle from a list of detected circles.
        
        Args:
            circles: List of detected circles
            side: Side identifier (e.g., "left" or "right")
            
        Returns:
            tuple: Best circle (x, y, r) or None if no circles
        """
        if not circles or len(circles) == 0:
            return None
            
        # For now, simply return the first circle
        # TODO: Implement more sophisticated selection based on confidence scores
        return circles[0]
    
    def _fuse_coordinates(self, left_coords, right_coords):
        """Calculate 3D position from 2D coordinates using triangulation."""
        if left_coords is None or right_coords is None:
            logging.debug("Skipping coordinate fusion: One or both coordinates are None")
            return
        
        # Log input coordinates
        logging.debug(f"Fusing coordinates: left={left_coords}, right={right_coords}")
        
        # Check disparity threshold (acceptable difference between y-coordinates)
        disparity_y = abs(left_coords[1] - right_coords[1])
        if disparity_y > 30:  # Threshold value can be adjusted
            logging.warning(f"Large y-disparity between left and right coordinates: {disparity_y}")
            return
        
        try:
            # Use our mock triangulation service to get 3D coordinates
            x_left, y_left = left_coords[0], left_coords[1]
            x_right, y_right = right_coords[0], right_coords[1]
            
            # Do the triangulation
            x, y, z = self.triangulator.triangulate(x_left, y_left, x_right, y_right)
            
            # Store the calculated 3D coordinates for the detection signal
            self._last_3d_coordinates = (float(x), float(y), float(z))
            
            # Log the calculated position
            logging.debug(f"Calculated 3D position: ({x:.2f}, {y:.2f}, {z:.2f})")
            
            # Emit tracking update signal with the 3D position
            self.tracking_updated.emit(float(x), float(y), float(z))
            
            # Return 3D coordinates for other uses
            return self._last_3d_coordinates
        
        except Exception as e:
            logging.error(f"Error during coordinate fusion: {str(e)}")
            return None
    
    def _check_out_of_bounds(self):
        """Check if ball is predicted to be out of bounds and update tracking state."""
        if not self.model.detection_stats["is_tracking"]:
            return
            
        left_pred = self.kalman_processor.get_prediction("left")
        right_pred = self.kalman_processor.get_prediction("right")
        
        if left_pred is None and right_pred is None:
            return
            
        # Define image boundaries
        left_bounds = (0, 0, self.model.left_image.shape[1], self.model.left_image.shape[0]) if self.model.left_image is not None else None
        right_bounds = (0, 0, self.model.right_image.shape[1], self.model.right_image.shape[0]) if self.model.right_image is not None else None
        
        is_out_of_bounds = False
        
        # Add margin to avoid false positives (15% of width/height)
        margin_x = 0
        margin_y = 0
        
        if left_bounds:
            margin_x = int(left_bounds[2] * 0.15)
            margin_y = int(left_bounds[3] * 0.15)
        elif right_bounds:
            margin_x = int(right_bounds[2] * 0.15)
            margin_y = int(right_bounds[3] * 0.15)
        
        # Check left prediction
        left_out = False
        if left_pred is not None and left_bounds is not None:
            px, py, vx, vy = left_pred
            
            # Log current prediction and bounds for debugging
            logging.debug(f"Left prediction: pos=({px}, {py}), vel=({vx}, {vy})")
            
            # Add margin to bounds for checking
            effective_bounds = (-margin_x, -margin_y, left_bounds[2] + margin_x, left_bounds[3] + margin_y)
            
            # Check if current position is outside bounds
            current_out = (px < effective_bounds[0] or px >= effective_bounds[2] or 
                          py < effective_bounds[1] or py >= effective_bounds[3])
                
            # Only consider rapid motion for out-of-bounds prediction
            if abs(vx) > 2.0 or abs(vy) > 2.0:
                future_steps = 3  # Look 3 frames ahead (more conservative)
                future_x = px + vx * future_steps
                future_y = py + vy * future_steps
                
                # Check if future position is outside bounds
                future_out = (future_x < -3*margin_x or future_x >= left_bounds[2] + 3*margin_x or 
                             future_y < -3*margin_y or future_y >= left_bounds[3] + 3*margin_y)
                
                # Consider out of bounds only if both current and future positions are out
                if current_out and future_out:
                    logging.debug(f"Left prediction significantly out of bounds: current=({px}, {py}), future=({future_x}, {future_y})")
                    left_out = True
            
        # Check right prediction (similar to left)
        right_out = False
        if right_pred is not None and right_bounds is not None:
            px, py, vx, vy = right_pred
            
            # Log current prediction and bounds for debugging
            logging.debug(f"Right prediction: pos=({px}, {py}), vel=({vx}, {vy})")
            
            # Add margin to bounds for checking
            effective_bounds = (-margin_x, -margin_y, right_bounds[2] + margin_x, right_bounds[3] + margin_y)
            
            # Check if current position is outside bounds
            current_out = (px < effective_bounds[0] or px >= effective_bounds[2] or 
                          py < effective_bounds[1] or py >= effective_bounds[3])
                
            # Only consider rapid motion for out-of-bounds prediction
            if abs(vx) > 2.0 or abs(vy) > 2.0:
                future_steps = 3  # Look 3 frames ahead (more conservative)
                future_x = px + vx * future_steps
                future_y = py + vy * future_steps
                
                # Check if future position is outside bounds
                future_out = (future_x < -3*margin_x or future_x >= right_bounds[2] + 3*margin_x or 
                             future_y < -3*margin_y or future_y >= right_bounds[3] + 3*margin_y)
                
                # Consider out of bounds only if both current and future positions are out
                if current_out and future_out:
                    logging.debug(f"Right prediction significantly out of bounds: current=({px}, {py}), future=({future_x}, {future_y})")
                    right_out = True
        
        # Use a counter for consecutive out-of-bounds detections
        if not hasattr(self, '_out_of_bounds_counter'):
            self._out_of_bounds_counter = 0
        
        # Consider out of bounds only if both sides are out or a single active side is out
        is_left_active = left_pred is not None and self.kalman_processor.is_filter_ready("left")
        is_right_active = right_pred is not None and self.kalman_processor.is_filter_ready("right")
        
        if ((is_left_active and is_right_active and left_out and right_out) or   # Both sides active and out
            (is_left_active and not is_right_active and left_out) or             # Only left active and out
            (is_right_active and not is_left_active and right_out)):             # Only right active and out
            self._out_of_bounds_counter += 1
            logging.debug(f"Out of bounds counter: {self._out_of_bounds_counter}")
        else:
            # Reset counter if ball is within bounds
            self._out_of_bounds_counter = 0
        
        # Only stop tracking after multiple consecutive out-of-bounds detections
        if self._out_of_bounds_counter >= 3:  # Adjust threshold as needed
            logging.info("Ball predicted out of bounds for multiple frames, stopping tracking")
            self.model.detection_stats["is_tracking"] = False
            self._out_of_bounds_counter = 0  # Reset counter
    
    def _update_detection_signal(self, frame_idx, timestamp, ball_center_left, ball_center_right):
        """Update detection signal with current ball tracking status."""
        if self.state != TrackingState.TRACKING:
            return

        # Get ball coordinates or None if not detected
        left_coords = ball_center_left if ball_center_left is not None else None
        right_coords = ball_center_right if ball_center_right is not None else None
        
        # Position coordinates from _last_3d_coordinates (from _fuse_coordinates)
        position_coords = self._last_3d_coordinates if hasattr(self, '_last_3d_coordinates') and self._last_3d_coordinates is not None else (0.0, 0.0, 0.0)
            
        # Log information before emitting the signal
        logging.debug(f"Emitting detection_updated: frame={frame_idx}, time={timestamp:.3f}, " 
                     f"left={left_coords}, right={right_coords}, pos={position_coords}")
        
        # Emit the signal with the coordinates
        self.detection_updated.emit(frame_idx, timestamp, left_coords, right_coords, position_coords)
    
    def detect_circles_in_roi(self):
        """
        Manually detect circles in the current ROIs and update the model.
        This is a public wrapper for the internal _detect_circles method.
        
        Returns:
            tuple: (left_processed_image, right_processed_image)
        """
        from src.views.visualization import VisualizerFactory
        
        if self.model.is_enabled and (self.model.left_image is not None or self.model.right_image is not None):
            # Get current values for detection
            left_image = self.model.left_image
            right_image = self.model.right_image
            left_mask = self.left_mask if hasattr(self, 'left_mask') else None
            right_mask = self.right_mask if hasattr(self, 'right_mask') else None
            roi_settings = self.get_roi_settings()
            
            # Call _detect_circles with all required parameters
            left_circles, right_circles = self._detect_circles(
                left_image, right_image, left_mask, right_mask, roi_settings)
                
            # Set the detected circles to the model
            self.model.left_circles = left_circles
            self.model.right_circles = right_circles
            
            # Create visualizers
            left_visualizer = VisualizerFactory.create(backend="opencv")
            right_visualizer = VisualizerFactory.create(backend="opencv") 
            
            # Create circle visualizations
            left_viz = None
            right_viz = None
            
            if self.model.left_image is not None and self.model.left_roi is not None:
                left_viz = self.model.left_image.copy()
                
                # Draw ROI if available
                if self.model.left_roi:
                    try:
                        left_viz = left_visualizer.draw_roi(
                            left_viz,
                            self.model.left_roi,
                            color=COLOR.GREEN,
                            thickness=TRACKING.ROI_THICKNESS,
                            show_center=True
                        )
                    except Exception as e:
                        logging.error(f"Error drawing left ROI in visualization: {e}")
                
                # Draw circles if available
                if self.model.left_circles:
                    try:
                        left_viz = left_visualizer.draw_circles(
                            left_viz,
                            self.model.left_circles,
                            color=COLOR.RED,
                            thickness=TRACKING.CIRCLE_THICKNESS,
                            label_circles=True
                        )
                    except Exception as e:
                        logging.error(f"Error drawing left circles in visualization: {e}")
            
            if self.model.right_image is not None and self.model.right_roi is not None:
                right_viz = self.model.right_image.copy()
                
                # Draw ROI if available
                if self.model.right_roi:
                    try:
                        right_viz = right_visualizer.draw_roi(
                            right_viz,
                            self.model.right_roi,
                            color=COLOR.GREEN,
                            thickness=TRACKING.ROI_THICKNESS,
                            show_center=True
                        )
                    except Exception as e:
                        logging.error(f"Error drawing right ROI in visualization: {e}")
                
                # Draw circles if available
                if self.model.right_circles:
                    try:
                        right_viz = right_visualizer.draw_circles(
                            right_viz,
                            self.model.right_circles,
                            color=COLOR.RED,
                            thickness=TRACKING.CIRCLE_THICKNESS,
                            label_circles=True
                        )
                    except Exception as e:
                        logging.error(f"Error drawing right circles in visualization: {e}")
            
            return left_viz, right_viz
        
        return None, None
    
    def get_predictions(self):
        """
        Get the current Kalman filter predictions.
        
        Returns:
            tuple: (left_prediction, right_prediction) where each is (x, y, vx, vy) or None
        """
        left_pred = self.kalman_processor.get_prediction("left")
        right_pred = self.kalman_processor.get_prediction("right")
        return left_pred, right_pred
    
    def get_current_masks(self):
        """
        Get the current masks.
            
        Returns:
            tuple: (left_mask, right_mask)
        """
        return self.model.left_mask, self.model.right_mask
    
    def get_hsv_values(self):
        """
        Get the current HSV values.
        
        Returns:
            dict: Current HSV values
        """
        if hasattr(self.model, 'hsv_values'):
            return self.model.hsv_values
        else:
            # Return our internal HSV values
            return self._hsv_values
    
    def get_roi_settings(self):
        """
        Get the current ROI settings.
        
        Returns:
            dict: Current ROI settings
        """
        if hasattr(self.model, 'roi_settings'):
            return self.model.roi_settings
        else:
            # Return our internal ROI settings
            return self._roi_settings
    
    def set_roi_settings(self, roi_settings):
        """
        Set ROI settings for ball tracking.
        
        Args:
            roi_settings (dict): Dictionary containing ROI settings
        """
        # Store internally for model compatibility
        self._roi_settings = roi_settings
        
        # Update model if it supports this method
        if hasattr(self.model, 'set_roi_settings') and callable(getattr(self.model, 'set_roi_settings')):
            self.model.set_roi_settings(roi_settings)
        elif hasattr(self.model, 'roi_settings'):
            self.model.roi_settings = roi_settings
        
        # Update configuration
        self.config_manager.set_roi_settings(roi_settings)
        
        # Update service
        self.roi_computer.update_roi_settings(roi_settings)
        
        logging.info(f"ROI settings updated and saved: {roi_settings}")
        
        # Reprocess images if enabled to update ROIs
        if self.is_enabled and (self.model.left_image is not None or self.model.right_image is not None):
            self._process_images()
    
    def get_current_rois(self):
        """
        Get the current ROIs.
        
        Returns:
            tuple: (left_roi, right_roi)
        """
        return self.model.left_roi, self.model.right_roi
    
    def get_cropped_roi_images(self):
        """
        Get cropped images based on current ROIs.
            
        Returns:
            tuple: (left_cropped_image, right_cropped_image)
        """
        return self.model.cropped_images["left"], self.model.cropped_images["right"]
    
    def get_detection_rate(self):
        """
        Get the current detection rate.
        
        Returns:
            float: Detection rate (0.0 to 1.0) or None if not tracking
        """
        if hasattr(self.model, 'get_detection_rate'):
            return self.model.get_detection_rate()
        else:
            # For StereoImageModel, return 0.0 or calculate based on internal counter
            if hasattr(self, '_detection_counter') and hasattr(self, '_frame_counter'):
                if self._frame_counter > 0:
                    return self._detection_counter / self._frame_counter
            return 0.0
    
    def get_latest_coordinates(self):
        """
        Get the latest coordinates from both left and right images.
        
        Returns:
            tuple: (left_coords, right_coords) where each is (x, y, r) or None if not available
        """
        if hasattr(self.model, 'get_latest_coordinates'):
            return self.model.get_latest_coordinates()
        else:
            # For StereoImageModel, use our internal coordinate history
            left_coords = None
            right_coords = None
            
            if hasattr(self, '_coordinate_history'):
                if self._coordinate_history["left"]:
                    left_coords = self._coordinate_history["left"][-1][:3]  # Remove timestamp
                if self._coordinate_history["right"]:
                    right_coords = self._coordinate_history["right"][-1][:3]  # Remove timestamp
            
            return left_coords, right_coords
    
    def get_coordinate_history(self, side="both", count=None):
        """
        Get the coordinate history.
        
        Args:
            side (str): "left", "right" or "both" to indicate which side to return
            count (int, optional): Number of most recent coordinates to return. All if None.
            
        Returns:
            dict or list: Coordinate history for the specified side(s)
        """
        if side == "both":
            if count is None:
                return self.model.coordinate_history
            else:
                return {
                    "left": self.model.coordinate_history["left"][-count:] if self.model.coordinate_history["left"] else [],
                    "right": self.model.coordinate_history["right"][-count:] if self.model.coordinate_history["right"] else []
                }
        elif side in ["left", "right"]:
            if count is None:
                return self.model.coordinate_history[side]
            else:
                return self.model.coordinate_history[side][-count:] if self.model.coordinate_history[side] else []
        else:
            logging.error(f"Invalid side: {side}")
            return None

    def clear_coordinate_history(self):
        """Clear the coordinate history."""
        self.model.clear_coordinate_history()
    
    def reset_tracking(self):
        """
        Reset the tracking by clearing history and resetting counters.
        """
        # Clear coordinate history
        self._coordinate_history = {
            'left': [],
            'right': [],
            'fused': []
        }
        
        # Reset counters and detection stats
        self._frames_analyzed = 0
        self._circles_detected = 0
        self._last_detection_timestamp = None
        
        # Reset kalman filter
        if hasattr(self, 'kalman_processor') and self.kalman_processor:
            self.kalman_processor.reset()
        
        # Reset frame counter
        self._frame_counter = 0
        
        # Clear data model tracking data
        if hasattr(self.model, 'clear_tracking_data'):
            self.model.clear_tracking_data()
        
        # Reset ROI if dynamic ROI is enabled
        roi_settings = self.get_roi_settings()
        if roi_settings.get('dynamic', False):
            # Reset to default ROIs
            self.left_roi = None
            self.right_roi = None
            
            if hasattr(self.model, 'left_roi'):
                self.model.left_roi = None
            if hasattr(self.model, 'right_roi'):
                self.model.right_roi = None
                
            # Emit ROI updated signal with empty ROIs
            if hasattr(self, 'roi_updated'):
                self.roi_updated.emit({}, {})
        
        # Reset 3D coordinates
        self._last_3d_coordinates = None
        
        # Save reset state to model
        if hasattr(self.model, 'set_tracking_reset'):
            self.model.set_tracking_reset(True)
            
        # Emit tracking state changed signal
        self.tracking_state_changed.emit(TrackingState.RESET)
            
        logging.info("Ball tracking reset complete")
    
    def save_coordinate_history(self, filename):
        """
        Save the coordinate history to a JSON file.
        
        Args:
            filename (str): Path to the output JSON file
        """
        try:
            history = self.get_coordinate_history()
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'w') as f:
                json.dump(history, f, indent=2)
            logging.info(f"Coordinate history saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving coordinate history: {e}")
            
    def save_tracking_data_to_json(self, folder_path=None, filename=None):
        """
        Save tracking data (original coordinates and Kalman predictions) to a JSON file.
        
        Args:
            folder_path (str, optional): Path to the output folder. Default is 'tracking_data' in current directory.
            filename (str, optional): Base filename without extension. Default uses timestamp.
        
        Returns:
            str: Path to the saved file or None if failed
        """
        try:
            # Prepare data dictionary
            tracking_data = {
                "timestamp": time.time(),
                "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
                "detection_rate": self.model.get_detection_rate(),
                "tracking_active": self.model.detection_stats["is_tracking"],
                "frames_total": self.model.detection_stats["total_frames"],
                "detections_count": self.model.detection_stats["detection_count"],
                "coordinate_data": {
                    "left": {
                        "hsv_centers": [],
                        "hough_centers": [],
                        "kalman_predictions": [],
                        "fused_centers": []
                    },
                    "right": {
                        "hsv_centers": [],
                        "hough_centers": [],
                        "kalman_predictions": [],
                        "fused_centers": []
                    }
                },
                "coordinate_history": {
                    "left": [],
                    "right": []
                }
            }
            
            # Process left side
            if self.model.left_mask is not None:
                # Get HSV mask centroid
                left_hsv_center = self.roi_computer.compute_mask_centroid(self.model.left_mask)
                
                # Get latest Hough circle center if available
                left_hough_center = None
                if self.model.left_circles and len(self.model.left_circles) > 0:
                    left_hough_center = (float(self.model.left_circles[0][0]), float(self.model.left_circles[0][1]))
                
                # Get Kalman prediction if available
                left_kalman_pred = self.kalman_processor.get_prediction("left")
                
                # Add to tracking data
                if left_hsv_center:
                    tracking_data["coordinate_data"]["left"]["hsv_centers"].append({
                        "x": float(left_hsv_center[0]),
                        "y": float(left_hsv_center[1])
                    })
                
                if left_hough_center:
                    tracking_data["coordinate_data"]["left"]["hough_centers"].append({
                        "x": float(left_hough_center[0]),
                        "y": float(left_hough_center[1])
                    })
                
                if left_kalman_pred:
                    tracking_data["coordinate_data"]["left"]["kalman_predictions"].append({
                        "x": float(left_kalman_pred[0]),
                        "y": float(left_kalman_pred[1]),
                        "vx": float(left_kalman_pred[2]),
                        "vy": float(left_kalman_pred[3])
                    })
            
            # Process right side (similar to left)
            if self.model.right_mask is not None:
                # Get HSV mask centroid
                right_hsv_center = self.roi_computer.compute_mask_centroid(self.model.right_mask)
                
                # Get latest Hough circle center if available
                right_hough_center = None
                if self.model.right_circles and len(self.model.right_circles) > 0:
                    right_hough_center = (float(self.model.right_circles[0][0]), float(self.model.right_circles[0][1]))
                
                # Get Kalman prediction if available
                right_kalman_pred = self.kalman_processor.get_prediction("right")
                
                # Add to tracking data
                if right_hsv_center:
                    tracking_data["coordinate_data"]["right"]["hsv_centers"].append({
                        "x": float(right_hsv_center[0]),
                        "y": float(right_hsv_center[1])
                    })
                
                if right_hough_center:
                    tracking_data["coordinate_data"]["right"]["hough_centers"].append({
                        "x": float(right_hough_center[0]),
                        "y": float(right_hough_center[1])
                    })
                
                if right_kalman_pred:
                    tracking_data["coordinate_data"]["right"]["kalman_predictions"].append({
                        "x": float(right_kalman_pred[0]),
                        "y": float(right_kalman_pred[1]),
                        "vx": float(right_kalman_pred[2]),
                        "vy": float(right_kalman_pred[3])
                    })
            
            # Convert coordinate history to serializable format
            for side in ["left", "right"]:
                history = self.model.coordinate_history[side]
                for entry in history:
                    # Each entry is (x, y, r, timestamp)
                    tracking_data["coordinate_history"][side].append({
                        "x": float(entry[0]),
                        "y": float(entry[1]),
                        "radius": float(entry[2]),
                        "timestamp": entry[3]
                    })
            
            # Use data saver service to save the data
            return self.data_saver.save_json_summary(tracking_data, folder_path, filename)
            
        except Exception as e:
            logging.error(f"Error saving tracking data: {e}")
            return None
    
    def save_tracking_data_for_frame(self, frame_number, folder_path=None):
        """
        Save tracking data for a specific frame, using the frame number as part of the filename.
        This allows overwriting data for the same frame when processed again.
        
        Args:
            frame_number (int): Current frame number
            folder_path (str, optional): Path to the output folder
            
        Returns:
            str: Path to the saved file or None if failed
        """
        try:
            # Prepare data dictionary
            frame_data = {
                "frame_number": frame_number,
                "timestamp": time.time(),
                "tracking_active": self.model.detection_stats["is_tracking"],
                "left": {
                    "hsv_center": None,
                    "hough_center": None,
                    "kalman_prediction": None,
                    "fused_center": None
                },
                "right": {
                    "hsv_center": None,
                    "hough_center": None, 
                    "kalman_prediction": None,
                    "fused_center": None
                }
            }
            
            # Process left side
            if self.model.left_mask is not None:
                # Get HSV mask centroid
                left_hsv_center = self.roi_computer.compute_mask_centroid(self.model.left_mask)
                if left_hsv_center:
                    frame_data["left"]["hsv_center"] = {
                        "x": float(left_hsv_center[0]),
                        "y": float(left_hsv_center[1])
                    }
                
                # Get latest Hough circle center if available
                left_hough_center = None
                if self.model.left_circles and len(self.model.left_circles) > 0:
                    left_hough_center = (float(self.model.left_circles[0][0]), float(self.model.left_circles[0][1]))
                    frame_data["left"]["hough_center"] = {
                        "x": left_hough_center[0],
                        "y": left_hough_center[1],
                        "radius": float(self.model.left_circles[0][2])
                    }
                
                # Get Kalman prediction if available
                left_kalman_pred = self.kalman_processor.get_prediction("left")
                if left_kalman_pred is not None:
                    left_kalman_pos = (float(left_kalman_pred[0]), float(left_kalman_pred[1]))
                    frame_data["left"]["kalman_prediction"] = {
                        "x": left_kalman_pos[0],
                        "y": left_kalman_pos[1],
                        "vx": float(left_kalman_pred[2]),
                        "vy": float(left_kalman_pred[3])
                    }
                
                # Calculate fused coordinates
                coords_to_fuse = []
                if left_hsv_center:
                    coords_to_fuse.append(left_hsv_center)
                if left_hough_center:
                    coords_to_fuse.append(left_hough_center)
                if left_kalman_pred:
                    coords_to_fuse.append((left_kalman_pred[0], left_kalman_pred[1]))
                
                if coords_to_fuse:
                    fused_coords = fuse_coordinates(coords_to_fuse)
                    if fused_coords:
                        frame_data["left"]["fused_center"] = {
                            "x": float(fused_coords[0]),
                            "y": float(fused_coords[1])
                        }
                        logging.debug(f"Left fused coordinates: {fused_coords}")
            
            # Process right side (similar to left)
            if self.model.right_mask is not None:
                # Get HSV mask centroid
                right_hsv_center = self.roi_computer.compute_mask_centroid(self.model.right_mask)
                if right_hsv_center:
                        frame_data["right"]["hsv_center"] = {
                            "x": float(right_hsv_center[0]),
                            "y": float(right_hsv_center[1])
                        }
                
                # Get latest Hough circle center if available
                right_hough_center = None
                if self.model.right_circles and len(self.model.right_circles) > 0:
                    right_hough_center = (float(self.model.right_circles[0][0]), float(self.model.right_circles[0][1]))
                    frame_data["right"]["hough_center"] = {
                        "x": right_hough_center[0],
                        "y": right_hough_center[1],
                        "radius": float(self.model.right_circles[0][2])
                    }
                
                # Get Kalman prediction if available
                right_kalman_pred = self.kalman_processor.get_prediction("right")
                if right_kalman_pred is not None:
                    right_kalman_pos = (float(right_kalman_pred[0]), float(right_kalman_pred[1]))
                    frame_data["right"]["kalman_prediction"] = {
                        "x": right_kalman_pos[0],
                        "y": right_kalman_pos[1],
                        "vx": float(right_kalman_pred[2]),
                        "vy": float(right_kalman_pred[3])
                    }
                
                # Calculate fused coordinates
                coords_to_fuse = []
                if right_hsv_center:
                    coords_to_fuse.append(right_hsv_center)
                if right_hough_center:
                    coords_to_fuse.append(right_hough_center)
                if right_kalman_pred:
                    coords_to_fuse.append((right_kalman_pred[0], right_kalman_pred[1]))
                
                if coords_to_fuse:
                    fused_coords = fuse_coordinates(coords_to_fuse)
                    if fused_coords:
                        frame_data["right"]["fused_center"] = {
                            "x": float(fused_coords[0]),
                            "y": float(fused_coords[1])
                        }
                        logging.debug(f"Right fused coordinates: {fused_coords}")
            
            # Use data saver service to save the frame data
            return self.data_saver.save_frame_to_xml(frame_number, frame_data, None)
            
        except Exception as e:
            logging.error(f"Error saving frame tracking data: {e}")
            return None
    
    def initialize_xml_tracking(self, folder_name):
        """
        Initialize XML tracking with the given folder name.
        
        Args:
            folder_name: Name of the folder for tracking data
        
        Returns:
            bool: Success or failure
        """
        try:
            # Initialize data saver if not already done
            if not hasattr(self, 'data_saver') or self.data_saver is None:
                self.data_saver = DataSaver()
            
            # Create output folder path
            output_path = os.path.join(os.getcwd(), "tracking_data", folder_name)
            os.makedirs(output_path, exist_ok=True)
            
            # Initialize XML tracking in the data saver
            result = self.data_saver.initialize_xml_tracking(folder_name)
            
            if result:
                logging.info(f"XML tracking initialized for folder: {folder_name}")
            else:
                logging.warning(f"Failed to initialize XML tracking for folder: {folder_name}")
                
            return result
        except Exception as e:
            logging.error(f"Error initializing XML tracking: {e}")
            return False
            
    def append_frame_xml(self, frame_number, frame_name=None):
        """
        Append the current frame's tracking data to XML file.
        
        Args:
            frame_number: Current frame number
            frame_name: Optional frame filename
        
        Returns:
            Boolean success indicator
        """
        try:
            # Get the current frame data
            frame_data = self.get_frame_data_dict(frame_number)
            
            # Use the data saver to add to XML
            result = self.data_saver.save_frame_to_xml(frame_number, frame_data, frame_name)
            
            if result:
                logging.debug(f"Frame {frame_number} tracking data saved to XML")
            else:
                logging.warning(f"Failed to save frame {frame_number} to XML")
                
            return result
                
        except Exception as e:
            logging.error(f"Error saving frame to XML: {e}")
            return False
    
    def save_xml_tracking_data(self, folder_path=None):
        """
        Save the complete XML tracking data to a file.
        For the incremental logging approach, this creates a full snapshot
        of the in-memory tracking data.
        
        Args:
            folder_path: Optional path to the output folder
            
        Returns:
            Path to the saved file or None if failed
        """
        try:
            # Have the data saver save a snapshot of the current data
            xml_path = self.data_saver.save_xml_tracking_data(folder_path)
            
            if xml_path:
                logging.info(f"XML tracking data snapshot saved to {xml_path}")
            
            return xml_path
            
        except Exception as e:
            logging.error(f"Error saving XML tracking data: {e}")
            return None
    
    def get_frame_data_dict(self, frame_number):
        """
        Create a dictionary with frame tracking data.
        
        Args:
            frame_number: Current frame number
            
        Returns:
            Dictionary with frame data
        """
        data = {
            "tracking_active": self.is_enabled,
            "left": {},
            "right": {}
        }
        
        # Initialize variables to avoid UnboundLocalError
        left_hsv_center = None
        right_hsv_center = None
        
        # Add left camera data - safely access circles
        if self.model.left_circles and len(self.model.left_circles) > 0:
            best_left = self.model.left_circles[0]
            data["left"]["hough_center"] = {
                "x": float(best_left[0]),
                "y": float(best_left[1]),
                "radius": float(best_left[2])
            }
        
        # Add HSV center for left
        if hasattr(self, 'hsv_mask_generator') and self.hsv_mask_generator:
            if self.model.left_mask is not None:
                left_hsv_center = self.roi_computer.compute_mask_centroid(self.model.left_mask)
            if left_hsv_center:
                data["left"]["hsv_center"] = {
                    "x": left_hsv_center[0],
                    "y": left_hsv_center[1]
                }
        
        # Add left Kalman prediction
        if self.model.left_prediction is not None:
            data["left"]["kalman_prediction"] = {
                "x": self.model.left_prediction[0],
                "y": self.model.left_prediction[1],
                "vx": self.model.left_prediction[2],
                "vy": self.model.left_prediction[3]
            }
        
        # Add right camera data - safely access circles
        if self.model.right_circles and len(self.model.right_circles) > 0:
            best_right = self.model.right_circles[0]
            data["right"]["hough_center"] = {
                "x": float(best_right[0]),
                "y": float(best_right[1]),
                "radius": float(best_right[2])
            }
        
        # Add HSV center for right
        if hasattr(self, 'hsv_mask_generator') and self.hsv_mask_generator:
            if self.model.right_mask is not None:
                right_hsv_center = self.roi_computer.compute_mask_centroid(self.model.right_mask)
            if right_hsv_center:
                data["right"]["hsv_center"] = {
                    "x": right_hsv_center[0],
                    "y": right_hsv_center[1]
                }
        
        # Add right Kalman prediction
        if self.model.right_prediction is not None:
            data["right"]["kalman_prediction"] = {
                "x": self.model.right_prediction[0],
                "y": self.model.right_prediction[1],
                "vx": self.model.right_prediction[2],
                "vy": self.model.right_prediction[3]
            }
        
        # Add latest 3D world point if available
        world_point = None
        if hasattr(self.model, 'get_latest_3d_point'):
            world_point = self.model.get_latest_3d_point()
        elif hasattr(self, '_world_coordinate_history') and self._world_coordinate_history:
            # Use our internal world coordinate history
            world_point = self._world_coordinate_history[-1]
            
        if world_point:
            data["world"] = world_point
            
        return data
    
    def save_frame_to_json(self, frame_number, folder_path=None):
        """
        Save the current frame's tracking data to JSON file.
        
        Args:
            frame_number: Current frame number
            folder_path: Optional path to the output folder
            
        Returns:
            Path to the saved file or None if failed
        """
        try:
            # Get the current frame data
            frame_data = self.get_frame_data_dict(frame_number)
            
            # Use the data saver to save as JSON
            result = self.data_saver.save_json_frame(frame_number, frame_data, folder_path)
            
            if result:
                logging.debug(f"Frame {frame_number} tracking data saved as JSON to {result}")
            else:
                logging.warning(f"Failed to save frame {frame_number} as JSON")
                
            return result
            
        except Exception as e:
            logging.error(f"Error saving frame to JSON: {e}")
            return None
    
    def process_frame(self, frame_index, frame=None):
        """
        Process a frame and log tracking data to XML.
        
        Args:
            frame_index: Index of the current frame
            frame: Frame object (optional)
        """
        # Safely check if tracking is enabled and images are available
        is_tracking_enabled = self.is_enabled
        has_left_image = hasattr(self.model, 'left_image') and self.model.left_image is not None
        has_right_image = hasattr(self.model, 'right_image') and self.model.right_image is not None
        
        # Process images if tracking is enabled and we have images
        if is_tracking_enabled and (has_left_image or has_right_image):
            # Process images using existing instances - don't create new ones
            self._process_images()
                
            # Log to XML if tracking data exists - throttle I/O to reduce overhead
            # Only log every 5 frames to reduce I/O overhead
            should_log_xml = frame_index % 5 == 0
            
            if should_log_xml and hasattr(self, 'data_saver') and self.data_saver is not None:
                # Get frame data dictionary
                frame_data = self.get_frame_data_dict(frame_index)
                
                # Generate frame name
                frame_name = f"frame_{frame_index:06d}.png"
                
                # Log to XML
                result = self.append_frame_xml(frame_index, frame_name)
                if result:
                    logging.debug(f"XML: frame {frame_index} appended")
                else:
                    logging.warning(f"Failed to append frame {frame_index} to XML")
        
        return True

    def update_camera_settings(self, camera_settings: Dict[str, Any]) -> None:
        """
        Update camera settings for 3D triangulation.
        
        Args:
            camera_settings: Camera configuration parameters
        """
        self.camera_settings = camera_settings.copy()
        
        # Update triangulation service with new settings
        if hasattr(self, 'triangulator') and self.triangulator is not None:
            self.triangulator.set_camera(self.camera_settings)
            logging.info("Camera settings updated for triangulation")
            
        # Save settings to configuration
        self.config_manager.set_camera_settings(camera_settings)

    def set_hough_circle_settings(self, hough_settings):
        """
        Update Hough Circle settings for circle detection.
        
        Args:
            hough_settings (dict): Dictionary containing Hough Circle parameters
        """
        # Store internally
        self._hough_settings = hough_settings.copy()
        
        # Update configuration
        self.config_manager.set_hough_circle_settings(hough_settings)
        
        # Update detector with new settings
        if hasattr(self, 'circle_detector') and self.circle_detector is not None:
            self.circle_detector.update_settings(hough_settings)
            logging.info(f"Hough circle settings updated: {hough_settings}")
        
        # Reprocess images if enabled
        if self.is_enabled and (self.model.left_image is not None or self.model.right_image is not None):
            self._process_images() 

    def get_current_hough_settings(self):
        """
        Get the current Hough Circle settings used by the detector.
        This is useful for debugging to verify what parameters are actually being used.
        
        Returns:
            dict: Current active Hough Circle settings
        """
        # Return a copy of the current settings to avoid accidental modification
        if hasattr(self, 'circle_detector') and self.circle_detector is not None:
            # Get settings directly from the detector instance for the most accurate values
            if hasattr(self.circle_detector, 'hough_settings'):
                return self.circle_detector.hough_settings.copy()
        
        # Fall back to internal settings
        return self._hough_settings.copy() if hasattr(self, '_hough_settings') else {}

    def get_detection_settings_summary(self):
        """
        Get a summary of all detection settings for debugging purposes.
        
        Returns:
            dict: Dictionary containing all detection settings
        """
        summary = {
            "hsv": self.get_hsv_values(),
            "roi": self.get_roi_settings(),
            "hough": self.get_current_hough_settings(),
            "detection_rate": self.get_detection_rate(),
            "is_enabled": self.is_enabled
        }
        
        # Add circle detector info if available
        if hasattr(self, 'circle_detector') and self.circle_detector is not None:
            if hasattr(self.circle_detector, 'adaptive'):
                summary["hough"]["adaptive_enabled"] = self.circle_detector.adaptive
        
        # Add additional detection info
        if hasattr(self.model, 'detection_stats'):
            summary["detection_stats"] = self.model.detection_stats
        
        # Add circle counts
        left_circles = getattr(self.model, 'left_circles', None)
        right_circles = getattr(self.model, 'right_circles', None)
        
        summary["circle_counts"] = {
            "left": len(left_circles) if left_circles is not None else 0,
            "right": len(right_circles) if right_circles is not None else 0
        }
        
        return summary 

    def _detect_circles_in_cropped_images(self, left_image, right_image, left_mask, right_mask, left_roi, right_roi):
        """
        Detect circles in cropped images within ROIs.
        This ensures Hough circles are only detected within the ROI region.
        
        Args:
            left_image (numpy.ndarray): Cropped left input image
            right_image (numpy.ndarray): Cropped right input image
            left_mask (numpy.ndarray): Cropped left binary mask
            right_mask (numpy.ndarray): Cropped right binary mask
            left_roi (dict): Left ROI dictionary
            right_roi (dict): Right ROI dictionary
            
        Returns:
            tuple: (left_circles, right_circles) - Lists of detected circles
        """
        # Use existing CircleDetector instance
        current_hough_settings = self._hough_settings.copy()
        
        # If adaptive is enabled in settings
        adaptive_enabled = current_hough_settings.get('adaptive', False)
        
        # Ensure min_radius is small enough for tennis ball detection
        if adaptive_enabled:
            if 'min_radius' in current_hough_settings:
                current_hough_settings['min_radius'] = max(HOUGH.MIN_RADIUS_CROPPED, min(current_hough_settings['min_radius'], HOUGH.MAX_RADIUS_CROPPED))
            
            if 'param2' in current_hough_settings:
                current_hough_settings['param2'] = min(current_hough_settings['param2'], HOUGH.MAX_PARAM2_CROPPED)
        
        # Update detector with modified settings if needed
        if current_hough_settings != self._hough_settings:
            self.circle_detector.update_settings(current_hough_settings)
            logging.debug(f"Updated circle detector settings for cropped images: {current_hough_settings}")
            # Store the updated settings
            self._hough_settings = current_hough_settings.copy()
        
        # Detect circles in cropped ROI images directly (no need to pass ROI parameter)
        # We're already working with cropped images
        left_result = self.circle_detector.detect_circles(
            img=left_image,
            mask=left_mask,
            roi=None,  # No need for ROI as image is already cropped
            side="left"
        )
        
        right_result = self.circle_detector.detect_circles(
            img=right_image,
            mask=right_mask,
            roi=None,  # No need for ROI as image is already cropped
            side="right"
        )
        
        # Create new circle lists with coordinates adjusted back to full image coordinates
        left_circles = left_result['circles']
        right_circles = right_result['circles']
        
        # Ensure we have lists, not None values
        left_circles = [] if left_circles is None else left_circles
        right_circles = [] if right_circles is None else right_circles
        
        # Convert coordinates from ROI-relative to full-image coordinates
        if left_circles and left_roi is not None:
            left_x = left_roi.get('x', 0)
            left_y = left_roi.get('y', 0)
            left_circles = [(x + left_x, y + left_y, r) for x, y, r in left_circles]
        
        if right_circles and right_roi is not None:
            right_x = right_roi.get('x', 0)
            right_y = right_roi.get('y', 0)
            right_circles = [(x + right_x, y + right_y, r) for x, y, r in right_circles]
        
        return left_circles, right_circles

    def update_kalman_settings(self, kalman_settings: Dict[str, Any]) -> None:
        """
        Update Kalman filter settings for ball tracking.
        
        Args:
            kalman_settings: Dictionary containing Kalman filter parameters
        """
        # Update internal settings
        self.kalman_settings.update(kalman_settings)
        
        # Update Kalman processor with new settings
        if hasattr(self, 'kalman_processor') and self.kalman_processor is not None:
            self.kalman_processor.update_params(kalman_settings)
            logging.info(f"Kalman filter settings updated")
        
        # Update configuration
        self.config_manager.set_kalman_settings(kalman_settings)
        
        # Reprocess current frame if tracking is enabled
        if self.is_enabled and (self.model.left_image is not None or self.model.right_image is not None):
            self._process_images()

    def _process_hsv_masks(self, left_image, right_image):
        # Process HSV masks
        left_mask = self._apply_hsv_mask(left_image, self.hsv_values)
        right_mask = self._apply_hsv_mask(right_image, self.hsv_values)

        # Set masks to model
        self.model.left_mask = left_mask
        self.model.right_mask = right_mask

        # Check for minimum pixel count
        min_pixel_threshold = TRACKING.MIN_PIXEL_THRESHOLD
        left_pixel_count = np.count_nonzero(left_mask)
        right_pixel_count = np.count_nonzero(right_mask)

        if left_pixel_count < min_pixel_threshold and right_pixel_count < min_pixel_threshold:
            logging.warning(f"HSV mask pixel count too low: left={left_pixel_count}, right={right_pixel_count}. Skipping frame.")
            return False

        return True

    def _draw_roi(self, left_frame, right_frame, show_original_roi=False):
        from src.views.visualization import VisualizerFactory
        
        # Draw ROI on both frames if enabled
        roi_settings = self.roi_settings or {}
        if not roi_settings.get('enabled', False):
            return
        
        # Create visualizers
        left_visualizer = VisualizerFactory.create(backend="opencv")
        right_visualizer = VisualizerFactory.create(backend="opencv")
        
        if hasattr(self, 'left_roi') and self.left_roi:
            x, y = self.left_roi.get('x', 0), self.left_roi.get('y', 0)
            w, h = self.left_roi.get('width', 0), self.left_roi.get('height', 0)
            
            # Draw rectangle for ROI
            left_frame = left_visualizer.draw_rectangle(
                left_frame, 
                (x, y), 
                (x + w, y + h), 
                color=COLOR.GREEN, 
                thickness=TRACKING.ROI_THICKNESS
            )
            
            # Draw center point
            center_x, center_y = x + w // 2, y + h // 2
            left_frame = left_visualizer.draw_point(
                left_frame, 
                (center_x, center_y), 
                radius=ROI.CENTER_MARKER_SIZE, 
                color=COLOR.RED, 
                thickness=-1  # Filled circle
            )
            logging.debug(f"Drew left ROI: {self.left_roi}")
        
        if hasattr(self, 'right_roi') and self.right_roi:
            x, y = self.right_roi.get('x', 0), self.right_roi.get('y', 0)
            w, h = self.right_roi.get('width', 0), self.right_roi.get('height', 0)
            
            # Draw rectangle for ROI
            right_frame = right_visualizer.draw_rectangle(
                right_frame, 
                (x, y), 
                (x + w, y + h), 
                color=COLOR.GREEN, 
                thickness=TRACKING.ROI_THICKNESS
            )
            
            # Draw center point
            center_x, center_y = x + w // 2, y + h // 2
            right_frame = right_visualizer.draw_point(
                right_frame, 
                (center_x, center_y), 
                radius=ROI.CENTER_MARKER_SIZE, 
                color=COLOR.RED, 
                thickness=-1  # Filled circle
            )
            logging.debug(f"Drew right ROI: {self.right_roi}")
    
    def _draw_detected_circles(self, left_frame, right_frame):
        from src.views.visualization import VisualizerFactory
        
        if not hasattr(self.model, 'left_circles') or not hasattr(self.model, 'right_circles'):
            return
        
        # Create visualizers
        left_visualizer = VisualizerFactory.create(backend="opencv")
        right_visualizer = VisualizerFactory.create(backend="opencv")
            
        left_circles = self.model.left_circles or []
        if left_circles:
            left_frame = left_visualizer.draw_circles(
                left_frame, 
                left_circles, 
                color=COLOR.YELLOW, 
                thickness=TRACKING.CIRCLE_THICKNESS,
                label_circles=False
            )
            logging.debug(f"Drew {len(left_circles)} circles on left image")
        
        right_circles = self.model.right_circles or []
        if right_circles:
            right_frame = right_visualizer.draw_circles(
                right_frame, 
                right_circles, 
                color=COLOR.YELLOW, 
                thickness=TRACKING.CIRCLE_THICKNESS,
                label_circles=False
            )
            logging.debug(f"Drew {len(right_circles)} circles on right image")
    
    def _draw_trajectory(self, left_frame, right_frame):
        from src.views.visualization import VisualizerFactory
        
        if not hasattr(self, 'kalman_processor') or self.kalman_processor is None:
            return
        
        # Create visualizers
        left_visualizer = VisualizerFactory.create(backend="opencv")
        right_visualizer = VisualizerFactory.create(backend="opencv")
            
        # Draw path on the left image
        left_history = self.kalman_processor.get_position_history("left")
        if left_history and len(left_history) > 1:
            # Draw trajectory
            left_frame = left_visualizer.draw_trajectory(
                left_frame,
                left_history,
                color=TRACKING.TRAJECTORY_COLOR,
                thickness=TRACKING.TRAJECTORY_THICKNESS,
                max_points=20
            )
            
            # Draw the predicted point if available
            if hasattr(self.model, 'left_prediction') and self.model.left_prediction is not None:
                x, y = self.model.left_prediction
                left_frame = left_visualizer.draw_point(
                    left_frame,
                    (int(x), int(y)),
                    color=COLOR.ORANGE,
                    radius=TRACKING.UNCERTAINTY_RADIUS // 4,
                    thickness=-1  # Filled circle
                )
        
        logging.debug(f"Drew left trajectory with {len(left_history) if left_history else 0} points")
        
        # Draw path on the right image
        right_history = self.kalman_processor.get_position_history("right")
        if right_history and len(right_history) > 1:
            # Draw trajectory
            right_frame = right_visualizer.draw_trajectory(
                right_frame,
                right_history,
                color=TRACKING.TRAJECTORY_COLOR,
                thickness=TRACKING.TRAJECTORY_THICKNESS,
                max_points=20
            )
            
            # Draw the predicted point if available
            if hasattr(self.model, 'right_prediction') and self.model.right_prediction is not None:
                x, y = self.model.right_prediction
                right_frame = right_visualizer.draw_point(
                    right_frame,
                    (int(x), int(y)),
                    color=COLOR.ORANGE,
                    radius=TRACKING.UNCERTAINTY_RADIUS // 4,
                    thickness=-1  # Filled circle
                )
        
        logging.debug(f"Drew right trajectory with {len(right_history) if right_history else 0} points")