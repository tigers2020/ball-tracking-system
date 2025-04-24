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

from src.models.tracking_data_model import TrackingDataModel
from src.services.hsv_mask_generator import HSVMaskGenerator
from src.services.roi_computer import ROIComputer
from src.services.circle_detector import CircleDetector
from src.services.kalman_processor import KalmanProcessor
from src.services.data_saver import DataSaver
from src.services.triangulation_service import TriangulationService
from src.utils.config_manager import ConfigManager
from src.utils.coord_utils import fuse_coordinates


class BallTrackingController(QObject):
    """
    Controller class for ball tracking functionality.
    Handles orchestration of various services for ball detection and tracking.
    """
    
    # Signals
    mask_updated = Signal(np.ndarray, np.ndarray, dict)  # left_mask, right_mask, hsv_settings
    roi_updated = Signal(dict, dict)  # left_roi, right_roi
    detection_updated = Signal(float, tuple, tuple)  # detection_rate, left_coords, right_coords
    circles_processed = Signal(np.ndarray, np.ndarray)  # left_circle_image, right_circle_image
    tracking_updated = Signal(float, float, float)  # x, y, z
    
    def __init__(self, model: Any, config_manager: ConfigManager):
        """
        Initialize BallTrackingController.
        
        Args:
            model: Data model (TrackingDataModel or StereoImageModel)
            config_manager: Configuration manager
        """
        super().__init__()
        
        self.model = model
        self.config_manager = config_manager
        
        # Initialize internal state variables for compatibility
        self._enabled = False
        self._detection_counter = 0
        self._frame_counter = 0
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
        self.circle_detector = CircleDetector(hough_settings)
        
        # Get Kalman settings and initialize processor
        self.kalman_settings = self.config_manager.get_kalman_settings()
        self.kalman_processor = KalmanProcessor(self.kalman_settings)
        
        # Initialize triangulation service with camera settings
        self.camera_settings = self.config_manager.get_camera_settings()
        self.triangulator = TriangulationService(self.camera_settings)
        
        self.data_saver = DataSaver()
        
        # Register finalize_xml with atexit to ensure XML is properly closed at exit
        atexit.register(self.data_saver.finalize_xml)
        
        # Timestamp tracking for Kalman filter dt calculation
        self.last_update_time = {"left": None, "right": None}
        
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
        if hasattr(self, 'tracking_enabled_changed'):
            self.tracking_enabled_changed.emit(enabled)
        
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
        Process images with the current ball tracking settings.
        This includes:
        1. Apply HSV threshold to create masks
        2. Compute ROIs
        3. Detect circles in ROIs
        4. Process predictions through Kalman filter (if enabled)
        5. Draw visualizations (ROI, circles, predictions, trajectories)
        """
        try:
            # Import visualization modules at the beginning
            from src.views.visualization.hough_visualizer import draw_circles
            from src.views.visualization.kalman_visualizer import draw_prediction, draw_trajectory
            from src.views.visualization.roi_visualizer import draw_roi
            
            # Skip processing if model or images aren't available
            if not hasattr(self.model, 'left_image') or not hasattr(self.model, 'right_image'):
                logging.warning("Model missing image attributes, skipping image processing")
                return
                
            # Extract properties
            left_image = self.model.left_image
            right_image = self.model.right_image
            
            # Skip processing if either image is None
            if left_image is None or right_image is None:
                return
                
            # Get current values
            hsv_values = self.get_hsv_values()
            roi_settings = self.get_roi_settings()
            
            # Apply HSV threshold to create masks
            left_mask, right_mask = self._apply_hsv_threshold(left_image, right_image, hsv_values)
            
            # Set the masks to the model
            self.model.left_mask = left_mask
            self.model.right_mask = right_mask
            
            # Store original masks for internal use
            self.left_mask = left_mask.copy()
            self.right_mask = right_mask.copy()
            
            # Calculate ROIs based on settings (even if disabled, for visualization purposes)
            self.left_roi, self.right_roi = self._compute_rois(left_image, right_image, roi_settings)
            
            # Set the ROIs to the model
            self.model.left_roi = self.left_roi
            self.model.right_roi = self.right_roi
            
            # Apply ROI masks only if ROI is enabled for detection
            if roi_settings.get('enabled', False):
                self.left_mask = self._apply_roi_mask(self.left_mask, self.left_roi)
                self.right_mask = self._apply_roi_mask(self.right_mask, self.right_roi)
            
            # Detect circles
            left_circles, right_circles = self._detect_circles(left_image, right_image, left_mask, right_mask, roi_settings)
            
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
            
            # 1. Always draw ROI on images (regardless of whether ROI is enabled)
            if self.left_roi:
                # Use thicker lines (4) for better visibility
                left_viz_image = draw_roi(
                    left_viz_image, 
                    self.left_roi, 
                    color=(0, 255, 0),  # Green color
                    thickness=4,
                    show_center=True,
                    center_color=(0, 0, 255)  # Red center
                )
                logging.debug(f"Drew left ROI: {self.left_roi}")
            
            if self.right_roi:
                right_viz_image = draw_roi(
                    right_viz_image, 
                    self.right_roi, 
                    color=(0, 255, 0),  # Green color
                    thickness=4,
                    show_center=True,
                    center_color=(0, 0, 255)  # Red center
                )
                logging.debug(f"Drew right ROI: {self.right_roi}")
            
            # 2. Draw circles on images if detected
            if left_circles:
                # Use thicker lines for better visibility
                left_viz_image = draw_circles(
                    left_viz_image, 
                    left_circles, 
                    main_color=(0, 255, 0),  # Green for main circle
                    thickness=4,
                    label_circles=True  # Add numbered labels for clearer identification
                )
                logging.debug(f"Drew {len(left_circles)} circles on left image")
            
            if right_circles:
                right_viz_image = draw_circles(
                    right_viz_image, 
                    right_circles, 
                    main_color=(0, 255, 0),  # Green for main circle
                    thickness=4,
                    label_circles=True
                )
                logging.debug(f"Drew {len(right_circles)} circles on right image")
            
            # 3. Draw Kalman prediction and trajectory on images if available
            if left_prediction:
                # Extract current position from circles and predicted position from Kalman
                current_pos = (int(left_circles[0][0]), int(left_circles[0][1])) if left_circles else None
                pred_pos = (int(left_prediction[0]), int(left_prediction[1]))
                
                # Draw prediction arrow with thicker line
                left_viz_image = draw_prediction(
                    left_viz_image, 
                    current_pos, 
                    pred_pos, 
                    arrow_color=(0, 255, 255),  # Yellow-green arrow
                    thickness=4,
                    draw_uncertainty=True,  # 예측 불확실성 표시
                    uncertainty_radius=20   # 더 큰 불확실성 원
                )
                
                # Draw trajectory if available
                left_history = self.kalman_processor.get_position_history("left")
                if left_history and len(left_history) > 1:
                    left_viz_image = draw_trajectory(
                        left_viz_image, 
                        left_history, 
                        color=(255, 255, 0),  # Yellow trajectory
                        thickness=5,          # 더 두꺼운 선으로 변경
                        max_points=20         # 더 많은 히스토리 포인트 표시
                    )
                    logging.debug(f"Drew left trajectory with {len(left_history)} points")
            
            if right_prediction:
                # Extract current position from circles and predicted position from Kalman
                current_pos = (int(right_circles[0][0]), int(right_circles[0][1])) if right_circles else None
                pred_pos = (int(right_prediction[0]), int(right_prediction[1]))
                
                # Draw prediction arrow with thicker line
                right_viz_image = draw_prediction(
                    right_viz_image, 
                    current_pos, 
                    pred_pos, 
                    arrow_color=(0, 255, 255),  # Yellow-green arrow
                    thickness=4,
                    draw_uncertainty=True,  # 예측 불확실성 표시
                    uncertainty_radius=20   # 더 큰 불확실성 원
                )
                
                # Draw trajectory if available
                right_history = self.kalman_processor.get_position_history("right")
                if right_history and len(right_history) > 1:
                    right_viz_image = draw_trajectory(
                        right_viz_image, 
                        right_history, 
                        color=(255, 255, 0),  # Yellow trajectory
                        thickness=5,          # 더 두꺼운 선으로 변경
                        max_points=20         # 더 많은 히스토리 포인트 표시
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
            
            # Update detection information to be displayed in the UI
            self._update_detection_signal()
            
            # Emit signal for image processing complete
            if hasattr(self, 'image_processed') and self.image_processed is not None:
                self.image_processed.emit()
            
        except Exception as e:
            logging.error(f"Error processing images: {str(e)}")
            import traceback
            logging.error(f"Error details: {traceback.format_exc()}")
    
    def _apply_hsv_threshold(self, left_image, right_image, hsv_values):
        """
        Apply HSV threshold to create binary masks from left and right images.
        
        Args:
            left_image (numpy.ndarray): Left input image
            right_image (numpy.ndarray): Right input image
            hsv_values (dict): Dictionary containing HSV min/max values
            
        Returns:
            tuple: (left_mask, right_mask) - Binary masks for both images
        """
        # Process left image
        left_hsv = cv2.cvtColor(left_image, cv2.COLOR_BGR2HSV)
        left_mask = cv2.inRange(left_hsv, 
                              (hsv_values['h_min'], hsv_values['s_min'], hsv_values['v_min']),
                              (hsv_values['h_max'], hsv_values['s_max'], hsv_values['v_max']))
        
        # Process right image
        right_hsv = cv2.cvtColor(right_image, cv2.COLOR_BGR2HSV)
        right_mask = cv2.inRange(right_hsv,
                               (hsv_values['h_min'], hsv_values['s_min'], hsv_values['v_min']),
                               (hsv_values['h_max'], hsv_values['s_max'], hsv_values['v_max']))
        
        return left_mask, right_mask
    
    def _compute_rois(self, left_image, right_image, roi_settings):
        """
        Compute ROIs based on the given images and ROI settings.
        
        Args:
            left_image (numpy.ndarray): Left input image
            right_image (numpy.ndarray): Right input image
            roi_settings (dict): Dictionary containing ROI settings
            
        Returns:
            tuple: (left_roi, right_roi) - ROI dictionaries for both images
        """
        left_roi = self.roi_computer.compute_roi(self.left_mask, left_image)
        right_roi = self.roi_computer.compute_roi(self.right_mask, right_image)
        return left_roi, right_roi
    
    def _apply_roi_mask(self, mask, roi):
        """
        Apply a ROI mask to an existing mask.
        
        Args:
            mask (numpy.ndarray): Input mask
            roi (dict): ROI dictionary
            
        Returns:
            numpy.ndarray: Mask with ROI applied
        """
        if roi:
            # 직접 필요한 값만 추출
            x = roi.get("x", 0)
            y = roi.get("y", 0)
            w = roi.get("width", 100)
            h = roi.get("height", 100)
            return cv2.rectangle(mask, (x, y), (x + w, y + h), (255), -1)
        else:
            return mask
    
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
        left_circles = self.circle_detector.detect_circles(
            img=left_image, 
            mask=left_mask, 
            roi=self.left_roi if roi_settings.get("enabled", False) else None
        )
        right_circles = self.circle_detector.detect_circles(
            img=right_image, 
            mask=right_mask, 
            roi=self.right_roi if roi_settings.get("enabled", False) else None
        )
        return left_circles['circles'], right_circles['circles']
    
    def _process_predictions(self, left_circles, right_circles):
        """
        Process detected circles through Kalman filter.
        
        Args:
            left_circles (list): List of detected left circles
            right_circles (list): List of detected right circles
            
        Returns:
            tuple: (left_prediction, right_prediction)
        """
        left_prediction = None
        right_prediction = None
        
        if left_circles:
            x, y, r = left_circles[0]
            left_prediction = self.kalman_processor.update("left", x, y)
        
            if right_circles:
                x, y, r = right_circles[0]
            right_prediction = self.kalman_processor.update("right", x, y)
        
        return left_prediction, right_prediction
    
    def _get_best_circle(self, circles):
        """
        Get the best circle from a list of detected circles.
        
        Args:
            circles: List of detected circles
            
        Returns:
            tuple: Best circle (x, y, r) or None if no circles are detected
        """
        if circles:
            return circles[0]
        else:
            return None
    
    def _fuse_coordinates(self):
        """
        Fuse coordinates from both left and right images.
        
        Returns:
            tuple: Fused coordinates (uL, vL, uR, vR) or None if no circles are detected
        """
        if self.model.left_circles and self.model.right_circles:
            left_circle = self.model.left_circles[0]
            right_circle = self.model.right_circles[0]
            return (left_circle[0], left_circle[1], right_circle[0], right_circle[1])
        else:
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
    
    def _update_detection_signal(self):
        """Emit detection update signal with current state."""
        # Get detection rate with compatibility check
        if hasattr(self.model, 'get_detection_rate'):
            detection_rate = self.model.get_detection_rate()
        else:
            # Default to calculated detection rate
            detection_rate = self._detection_counter / self._frame_counter if self._frame_counter > 0 else 0.0
            
        # Prepare coordinates for signal
        left_coords = None
        right_coords = None
        
        # Get coordinates from the detected circles
        if hasattr(self, 'left_circles') and self.left_circles and len(self.left_circles) > 0:
            # Extract (x, y, r) from the first detected circle
            left_circle = self.left_circles[0]
            left_coords = (int(left_circle[0]), int(left_circle[1]), int(left_circle[2]))
            logging.debug(f"Left circle coordinates: {left_coords}")
            
        if hasattr(self, 'right_circles') and self.right_circles and len(self.right_circles) > 0:
            # Extract (x, y, r) from the first detected circle
            right_circle = self.right_circles[0]
            right_coords = (int(right_circle[0]), int(right_circle[1]), int(right_circle[2]))
            logging.debug(f"Right circle coordinates: {right_coords}")
            
        # Emit the signal with detection information
        logging.debug(f"Emitting detection_updated signal: rate={detection_rate:.2f}, left={left_coords}, right={right_coords}")
        self.detection_updated.emit(detection_rate, left_coords, right_coords)
    
    def detect_circles_in_roi(self):
        """
        Manually detect circles in the current ROIs and update the model.
        This is a public wrapper for the internal _detect_circles method.
        
        Returns:
            tuple: (left_processed_image, right_processed_image)
        """
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
            
            # Create circle visualizations
            left_viz = None
            right_viz = None
            
            if self.model.left_image is not None and self.model.left_roi is not None:
                left_viz = self.model.left_image.copy()
                
                # Draw ROI if available
                if self.model.left_roi:
                    try:
                        x = int(self.model.left_roi.get("x", 0))
                        y = int(self.model.left_roi.get("y", 0))
                        w = int(self.model.left_roi.get("width", 100))
                        h = int(self.model.left_roi.get("height", 100))
                        cv2.rectangle(left_viz, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    except (ValueError, TypeError, KeyError) as e:
                        logging.error(f"Error drawing left ROI in visualization: {e}")
                
                # Draw circles if available
                if self.model.left_circles:
                    for circle in self.model.left_circles:
                        try:
                            x, y, r = circle
                            cv2.circle(left_viz, (int(x), int(y)), int(r), (0, 0, 255), 2)
                        except (ValueError, TypeError, IndexError) as e:
                            logging.error(f"Error drawing left circle in visualization: {e}")
            
            if self.model.right_image is not None and self.model.right_roi is not None:
                right_viz = self.model.right_image.copy()
                
                # Draw ROI if available
                if self.model.right_roi:
                    try:
                        x = int(self.model.right_roi.get("x", 0))
                        y = int(self.model.right_roi.get("y", 0))
                        w = int(self.model.right_roi.get("width", 100))
                        h = int(self.model.right_roi.get("height", 100))
                        cv2.rectangle(right_viz, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    except (ValueError, TypeError, KeyError) as e:
                        logging.error(f"Error drawing right ROI in visualization: {e}")
                
                # Draw circles if available
                if self.model.right_circles:
                    for circle in self.model.right_circles:
                        try:
                            x, y, r = circle
                            cv2.circle(right_viz, (int(x), int(y)), int(r), (0, 0, 255), 2)
                        except (ValueError, TypeError, IndexError) as e:
                            logging.error(f"Error drawing right circle in visualization: {e}")
            
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
        Reset all tracking data and filters.
        """
        try:
            # Clean up data saver queue
            self.data_saver.cleanup()
            
            # Reset Kalman filters
            self.kalman_processor.reset()
            
            # Reset model data if method exists
            if hasattr(self.model, 'reset'):
                self.model.reset()
            else:
                # Basic reset for StereoImageModel
                if hasattr(self.model, 'left_mask'):
                    self.model.left_mask = None
                if hasattr(self.model, 'right_mask'):
                    self.model.right_mask = None
                if hasattr(self.model, 'left_roi'):
                    self.model.left_roi = None
                if hasattr(self.model, 'right_roi'):
                    self.model.right_roi = None
                # Clear cropped images
                self.model.cropped_images = {
                    "left": None,
                    "right": None
                }
            
            # Reset timestamp tracking
            self.last_update_time = {"left": None, "right": None}
            
            logging.info("Ball tracking reset complete")
            
        except Exception as e:
            logging.error(f"Error resetting tracking: {e}")
            
        # Return stats if available, or a default value
        if hasattr(self.model, 'detection_stats'):
            return self.model.detection_stats
        else:
            # Create default stats
            return {
                "is_tracking": False,
                "frames_processed": 0,
                "frames_detected": 0,
                "detection_rate": 0.0,
                "lost_frames": 0
            }

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
            # Always process images when this method is called
            self._process_images()
                
            # Log to XML if tracking data exists
            if hasattr(self, 'data_saver') and self.data_saver is not None:
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