#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tennis ball detector module.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
import time
import logging


class BallDetector:
    """
    Tennis ball detector using color-based segmentation and blob detection.
    """
    
    def __init__(
        self,
        hsv_lower: Tuple[int, int, int] = (25, 70, 70),
        hsv_upper: Tuple[int, int, int] = (65, 255, 255),
        min_radius: int = 5,
        max_radius: int = 50,
        min_area: int = 30,
        max_area: int = 3000,
        blur_size: int = 5,
        detection_threshold: float = 0.5,
        history_len: int = 5,
    ):
        """
        Initialize the ball detector.
        
        Args:
            hsv_lower: Lower HSV threshold for tennis ball color
            hsv_upper: Upper HSV threshold for tennis ball color
            min_radius: Minimum radius of ball in pixels
            max_radius: Maximum radius of ball in pixels
            min_area: Minimum area of ball contour in pixels
            max_area: Maximum area of ball contour in pixels
            blur_size: Size of Gaussian blur kernel
            detection_threshold: Confidence threshold for detection
            history_len: Length of detection history to keep
        """
        self.hsv_lower = hsv_lower
        self.hsv_upper = hsv_upper
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_area = min_area
        self.max_area = max_area
        self.blur_size = blur_size
        self.detection_threshold = detection_threshold
        
        # For background subtraction
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, 
            varThreshold=50, 
            detectShadows=False
        )
        
        # Store detection history for smoother tracking
        self.detection_history = []
        self.history_len = history_len
        
        # Time of last detection for calculating velocity
        self.last_detection_time = None
        self.last_position = None
        self.velocity = None
        
        # For adaptive HSV range
        self.adaptive_mode = False
        self.hsv_ranges = []
        
        # ROI tracking
        self.use_roi = False
        self.roi = None
        self.roi_margin = 50  # 픽셀 단위의 ROI 여유 공간
        self.roi_min_size = 100  # 최소 ROI 크기
        self.roi_max_size = 300  # 최대 ROI 크기
        self.roi_expansion_rate = 1.5  # 검출 실패 시 확장 비율
        self.roi_contraction_rate = 0.9  # 검출 성공 시 축소 비율
        self.roi_failure_count = 0
        self.roi_max_failures = 5  # 연속 실패 최대 횟수
        
        # Blob detector parameters
        self.params = cv2.SimpleBlobDetector_Params()
        self.params.filterByArea = True
        self.params.minArea = min_area
        self.params.maxArea = max_area
        self.params.filterByCircularity = True
        self.params.minCircularity = 0.6
        self.params.filterByConvexity = True
        self.params.minConvexity = 0.7
        self.params.filterByInertia = True
        self.params.minInertiaRatio = 0.5
        
        # Create blob detector
        self.blob_detector = cv2.SimpleBlobDetector_create(self.params)
        
    def add_hsv_range(self, hsv_lower: Tuple[int, int, int], hsv_upper: Tuple[int, int, int]):
        """
        Add a HSV color range for detection.
        
        Args:
            hsv_lower: Lower HSV threshold
            hsv_upper: Upper HSV threshold
        """
        self.hsv_ranges.append((hsv_lower, hsv_upper))
        
    def enable_adaptive_mode(self, enable: bool = True):
        """
        Enable or disable adaptive HSV range mode.
        
        Args:
            enable: True to enable adaptive mode, False to disable
        """
        self.adaptive_mode = enable
        
    def enable_roi_tracking(self, enable: bool = True, initial_roi: Tuple[int, int, int, int] = None):
        """
        ROI 기반 트래킹을 활성화/비활성화합니다.
        
        Args:
            enable: ROI 트래킹 활성화 여부
            initial_roi: 초기 ROI (x, y, width, height)
        """
        self.use_roi = enable
        if enable and initial_roi:
            self.roi = initial_roi
            self.roi_failure_count = 0

    def update_roi(self, center: Tuple[int, int], radius: int, frame_shape: Tuple[int, int]):
        """
        볼 위치와 속도에 기반하여 ROI를 동적으로 업데이트합니다.
        
        Args:
            center: 볼의 중심 좌표 (x, y)
            radius: 볼의 반경
            frame_shape: 프레임 크기 (height, width)
        """
        if not self.use_roi or center is None:
            return
            
        # 프레임 크기 가져오기
        height, width = frame_shape
        
        # 속도 기반 ROI 예측
        pred_x, pred_y = center
        pred_size = max(radius * 4, self.roi_min_size)
        
        # 속도 정보가 있으면 예측 위치 조정
        if self.velocity is not None:
            # 속도에 따라 ROI 위치 예측 (0.1초 후 위치 예측)
            vx, vy = self.velocity
            pred_x += int(vx * 0.1)
            pred_y += int(vy * 0.1)
            
            # 속도 크기에 따라 ROI 크기 조정
            velocity_magnitude = np.sqrt(vx**2 + vy**2)
            if velocity_magnitude > 100:  # 빠르게 움직이는 경우
                pred_size = min(pred_size * 1.5, self.roi_max_size)
        
        # ROI 여유 공간 추가
        half_size = pred_size // 2 + self.roi_margin
        
        # ROI 계산
        roi_x = max(0, pred_x - half_size)
        roi_y = max(0, pred_y - half_size)
        roi_w = min(width - roi_x, pred_size + self.roi_margin * 2)
        roi_h = min(height - roi_y, pred_size + self.roi_margin * 2)
        
        # ROI 업데이트
        self.roi = (int(roi_x), int(roi_y), int(roi_w), int(roi_h))
        self.roi_failure_count = 0

    def expand_roi(self, frame_shape: Tuple[int, int]):
        """
        연속 검출 실패 시 ROI를 확장합니다.
        
        Args:
            frame_shape: 프레임 크기 (height, width)
        """
        if not self.use_roi or self.roi is None:
            return
            
        # 프레임 크기 가져오기
        height, width = frame_shape
        
        # 현재 ROI 정보
        x, y, w, h = self.roi
        
        # ROI 중심점 계산
        center_x = x + w // 2
        center_y = y + h // 2
        
        # 확장된 크기 계산 (최대 크기 제한)
        new_w = min(int(w * self.roi_expansion_rate), width)
        new_h = min(int(h * self.roi_expansion_rate), height)
        
        # 새 ROI 위치 계산
        new_x = max(0, center_x - new_w // 2)
        new_y = max(0, center_y - new_h // 2)
        
        # 프레임 경계 확인
        if new_x + new_w > width:
            new_x = width - new_w
        if new_y + new_h > height:
            new_y = height - new_h
            
        # ROI 업데이트
        self.roi = (int(new_x), int(new_y), int(new_w), int(new_h))
        
        # 연속 실패 횟수가 최대치에 도달하면 ROI 비활성화
        self.roi_failure_count += 1
        if self.roi_failure_count >= self.roi_max_failures:
            # 전체 프레임으로 ROI 확장
            self.roi = (0, 0, width, height)

    def reset_roi(self, frame_shape):
        """ROI를 화면 중앙으로 재설정"""
        h, w = frame_shape
        center_x, center_y = w // 2, h // 2
        roi_size = self.roi_min_size * 2
        self.roi = (center_x - roi_size//2, center_y - roi_size//2, roi_size, roi_size)
        self.roi_failure_count = 0

    def detect(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[int], float]:
        """
        Detect tennis ball in the frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            Tuple containing:
                - Center coordinates (x, y) or None if not detected
                - Radius of the ball or None if not detected
                - Confidence score (0.0 to 1.0)
        """
        # 프레임 크기 가져오기
        frame_height, frame_width = frame.shape[:2]
        
        # ROI 적용 (활성화된 경우)
        roi_applied = False
        if self.use_roi and self.roi is not None:
            x, y, w, h = self.roi
            # ROI가 유효한지 확인
            if w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= frame_width and y + h <= frame_height:
                roi_frame = frame[y:y+h, x:x+w]
                roi_applied = True
            else:
                roi_frame = frame
                self.roi = (0, 0, frame_width, frame_height)
        else:
            roi_frame = frame
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(roi_frame, (self.blur_size, self.blur_size), 0)
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Create masks for the tennis ball color
        if self.adaptive_mode and self.hsv_ranges:
            # Use multiple HSV ranges if in adaptive mode
            mask = None
            for lower, upper in self.hsv_ranges:
                range_mask = cv2.inRange(hsv, lower, upper)
                if mask is None:
                    mask = range_mask
                else:
                    mask = cv2.bitwise_or(mask, range_mask)
        else:
            # Use default HSV range for yellow
            yellow_mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
            
            # Add additional masks for red (handling hue wrap-around)
            red_lower1 = (0, 70, 70)
            red_upper1 = (10, 255, 255)
            red_lower2 = (170, 70, 70)
            red_upper2 = (180, 255, 255)
            
            red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
            red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
            
            # Combine all masks
            mask = cv2.bitwise_or(yellow_mask, cv2.bitwise_or(red_mask1, red_mask2))
        
        # Apply morphological operations to remove noise (증가된 반복 횟수)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # 모폴로지 연산 2회 수행
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Combine with background subtraction for moving objects
        fg_mask = self.bg_subtractor.apply(roi_frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Combine masks
        combined_mask = cv2.bitwise_and(mask, fg_mask)
        
        # Use blob detector for robust detection
        inverted_mask = cv2.bitwise_not(combined_mask)
        
        # Try Hough circles detection with reduced threshold
        circles = cv2.HoughCircles(
            inverted_mask, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2,
            minDist=10,
            param1=50,
            param2=25,  # 36에서 25로 낮춤
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        
        best_center = None
        best_radius = None
        best_confidence = 0.0
        
        # Process Hough circles if found
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                center = (int(circle[0]), int(circle[1]))
                radius = int(circle[2])
                
                # Check if circle is in motion (using background subtraction)
                motion_mask = cv2.circle(np.zeros_like(fg_mask), center, radius, 255, -1)
                motion_score = cv2.countNonZero(cv2.bitwise_and(fg_mask, motion_mask)) / max(cv2.countNonZero(motion_mask), 1)
                
                # Calculate confidence score
                confidence = 0.5 + 0.3 * (1.0 - abs(radius - 15) / 30) + 0.2 * motion_score
                
                if confidence > best_confidence:
                    best_center = center
                    best_radius = radius
                    best_confidence = confidence
        
        # Process keypoints from blob detector as fallback
        keypoints = self.blob_detector.detect(inverted_mask)
        for keypoint in keypoints:
            center = (int(keypoint.pt[0]), int(keypoint.pt[1]))
            radius = int(keypoint.size / 2)
            
            if radius < self.min_radius or radius > self.max_radius:
                continue
                
            # Check if point is in motion
            motion_mask = cv2.circle(np.zeros_like(fg_mask), center, radius, 255, -1)
            motion_score = cv2.countNonZero(cv2.bitwise_and(fg_mask, motion_mask)) / max(cv2.countNonZero(motion_mask), 1)
            
            # Calculate confidence score
            confidence = 0.4 * keypoint.response + 0.3 * (1.0 - abs(radius - 15) / 30) + 0.3 * motion_score
            
            if confidence > best_confidence:
                best_center = center
                best_radius = radius
                best_confidence = confidence
        
        # ROI 좌표계에서 전체 프레임 좌표계로 변환
        if roi_applied and best_center is not None:
            best_center = (best_center[0] + self.roi[0], best_center[1] + self.roi[1])
                
        # Update detection history
        current_time = time.time()
        
        if best_center is not None and best_confidence >= self.detection_threshold:
            # Calculate velocity if possible
            if self.last_detection_time is not None and self.last_position is not None:
                dt = current_time - self.last_detection_time
                if dt > 0:
                    dx = (best_center[0] - self.last_position[0]) / dt
                    dy = (best_center[1] - self.last_position[1]) / dt
                    self.velocity = (dx, dy)
            
            # ROI 업데이트 (성공적인 검출)
            if self.use_roi:
                self.update_roi(best_center, best_radius, frame.shape[:2])
                    
            # Update history
            self.detection_history.append((best_center, best_radius, best_confidence))
            if len(self.detection_history) > self.history_len:
                self.detection_history.pop(0)
                
            # Update last position and time
            self.last_position = best_center
            self.last_detection_time = current_time
            
            return best_center, best_radius, best_confidence
        else:
            # ROI 확장 (검출 실패)
            if self.use_roi:
                self.roi_failure_count += 1
                if self.roi_failure_count >= self.roi_max_failures:
                    # 최대 실패 횟수에 도달하면 ROI를 화면 중앙으로 재설정
                    self.reset_roi(frame.shape[:2])
                    logging.warning("Max ROI failures reached. Resetting ROI to center.")
                else:
                    self.expand_roi(frame.shape[:2])
                
            # If no detection, try to predict based on velocity
            if (self.last_position is not None and self.velocity is not None and 
                current_time - self.last_detection_time < 0.5):  # Only predict for short gaps
                
                dt = current_time - self.last_detection_time
                predicted_x = int(self.last_position[0] + self.velocity[0] * dt)
                predicted_y = int(self.last_position[1] + self.velocity[1] * dt)
                
                # Check if prediction is within frame bounds
                h, w = frame.shape[:2]
                if 0 <= predicted_x < w and 0 <= predicted_y < h:
                    # Return prediction with lower confidence
                    predicted_center = (predicted_x, predicted_y)
                    if self.detection_history:
                        last_radius = self.detection_history[-1][1]
                        return predicted_center, last_radius, self.detection_threshold * 0.7
            
            return None, None, 0.0
    
    def draw_detection(
        self, 
        frame: np.ndarray, 
        center: Optional[Tuple[int, int]], 
        radius: Optional[int], 
        confidence: float
    ) -> np.ndarray:
        """
        Draw the detection result on the frame.
        
        Args:
            frame: Input image frame
            center: Center coordinates (x, y) or None if not detected
            radius: Radius of the ball or None if not detected
            confidence: Confidence score (0.0 to 1.0)
            
        Returns:
            Frame with detection visualization
        """
        result = frame.copy()
        
        # ROI 영역 표시 (활성화된 경우)
        if self.use_roi and self.roi is not None:
            x, y, w, h = self.roi
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                result, 
                f"ROI: {w}x{h}", 
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 0, 0), 
                2
            )
        
        if center is not None and radius is not None:
            # Draw the ball circle
            cv2.circle(result, center, radius, (0, 255, 0), 2)
            
            # Draw the center point
            cv2.circle(result, center, 2, (0, 0, 255), -1)
            
            # Draw confidence text
            text = f"Conf: {confidence:.2f}"
            cv2.putText(
                result, 
                text, 
                (center[0] - radius, center[1] - radius - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                2
            )
            
            # 속도 정보 표시
            if self.velocity is not None:
                vx, vy = self.velocity
                speed = np.sqrt(vx**2 + vy**2)
                cv2.putText(
                    result, 
                    f"Speed: {speed:.1f} px/s", 
                    (center[0] - radius, center[1] - radius - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    2
                )
            
            # Draw detection trail
            for i in range(1, len(self.detection_history)):
                prev_center = self.detection_history[i-1][0]
                curr_center = self.detection_history[i][0]
                # Use color gradient from red to green based on age
                color_factor = i / len(self.detection_history)
                color = (0, int(255 * color_factor), int(255 * (1 - color_factor)))
                cv2.line(result, prev_center, curr_center, color, 2)
                
        return result

    def calibrate_from_roi(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> None:
        """
        Calibrate HSV color range from a region of interest and set up ROI tracking.
        
        Args:
            frame: Input image frame
            roi: Region of interest as (x, y, width, height)
        """
        # Extract ROI
        x, y, w, h = roi
        roi_image = frame[y:y+h, x:x+w]
        
        # Convert to HSV
        hsv_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
        
        # Calculate mean HSV values
        mean_hsv = cv2.mean(hsv_roi)[:3]
        
        # Calculate HSV standard deviation
        hsv_std = np.std(hsv_roi.reshape(-1, 3), axis=0)
        
        # Set new HSV range with mean ± 2*std
        std_scale = 2.0
        self.hsv_lower = tuple(max(0, int(mean_hsv[i] - std_scale * hsv_std[i])) for i in range(3))
        self.hsv_upper = tuple(min(255, int(mean_hsv[i] + std_scale * hsv_std[i])) for i in range(3))
        
        # Add to HSV ranges list
        self.add_hsv_range(self.hsv_lower, self.hsv_upper)
        
        # 자동으로 ROI 트래킹 활성화
        self.enable_roi_tracking(True, roi)
        
        # 첫 검출을 위해 ROI를 약간 확장
        self.roi = (
            max(0, x - self.roi_margin), 
            max(0, y - self.roi_margin),
            min(frame.shape[1] - x, w + 2 * self.roi_margin),
            min(frame.shape[0] - y, h + 2 * self.roi_margin)
        )


class StereoBallDetector:
    """
    Stereo tennis ball detector for 3D position calculation.
    """
    
    def __init__(
        self,
        left_detector: BallDetector = None,
        right_detector: BallDetector = None,
        baseline: float = 0.1,  # meters
        focal_length: float = 500,  # pixels
        principal_point: Tuple[float, float] = (320, 240),  # pixels
    ):
        """
        Initialize stereo ball detector.
        
        Args:
            left_detector: BallDetector for left camera (created if None)
            right_detector: BallDetector for right camera (created if None)
            baseline: Distance between cameras in meters
            focal_length: Focal length in pixels
            principal_point: Principal point (cx, cy) in pixels
        """
        self.left_detector = left_detector if left_detector else BallDetector()
        self.right_detector = right_detector if right_detector else BallDetector()
        self.baseline = baseline
        self.focal_length = focal_length
        self.principal_point = principal_point
        
        # Last detections
        self.last_left_detection = None
        self.last_right_detection = None
        self.last_3d_position = None
        
    def detect(
        self, 
        left_frame: np.ndarray, 
        right_frame: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]], Optional[np.ndarray], float]:
        """
        Detect ball in stereo images and calculate 3D position.
        
        Args:
            left_frame: Left camera frame
            right_frame: Right camera frame
            
        Returns:
            Tuple containing:
                - Left image coordinates (x, y) or None
                - Right image coordinates (x, y) or None
                - 3D position [X, Y, Z] in meters or None
                - Confidence score (0.0 to 1.0)
        """
        # Detect ball in left and right frames
        left_center, left_radius, left_conf = self.left_detector.detect(left_frame)
        right_center, right_radius, right_conf = self.right_detector.detect(right_frame)
        
        # Calculate combined confidence
        confidence = min(left_conf, right_conf)
        
        # Keep track of the last valid detections
        if left_center is not None and left_conf > 0.3:
            self.last_left_detection = (left_center, left_radius, left_conf)
            
        if right_center is not None and right_conf > 0.3:
            self.last_right_detection = (right_center, right_radius, right_conf)
        
        # If we have valid detections in both frames
        if left_center is not None and right_center is not None:
            # Calculate disparity
            disparity = left_center[0] - right_center[0]
            
            # Avoid division by zero
            if abs(disparity) > 1:  # Threshold to avoid noise
                # Calculate depth (Z)
                Z = self.baseline * self.focal_length / disparity
                
                # Calculate X and Y
                X = (left_center[0] - self.principal_point[0]) * Z / self.focal_length
                Y = (left_center[1] - self.principal_point[1]) * Z / self.focal_length
                
                # Create 3D position vector
                position_3d = np.array([X, Y, Z])
                
                # Store last 3D position
                self.last_3d_position = position_3d
                
                return left_center, right_center, position_3d, confidence
            
        return left_center, right_center, None, confidence
    
    def draw_detection(
        self, 
        left_frame: np.ndarray, 
        right_frame: np.ndarray,
        left_center: Optional[Tuple[int, int]],
        right_center: Optional[Tuple[int, int]],
        position_3d: Optional[np.ndarray],
        confidence: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw detection results on stereo frames.
        
        Args:
            left_frame: Left camera frame
            right_frame: Right camera frame
            left_center: Left image coordinates (x, y) or None
            right_center: Right image coordinates (x, y) or None
            position_3d: 3D position [X, Y, Z] in meters or None
            confidence: Confidence score (0.0 to 1.0)
            
        Returns:
            Tuple containing:
                - Left frame with detection visualization
                - Right frame with detection visualization
        """
        # Draw left detection
        if left_center is not None and self.last_left_detection is not None:
            left_result = self.left_detector.draw_detection(
                left_frame, 
                left_center, 
                self.last_left_detection[1],  # radius
                confidence
            )
        else:
            left_result = left_frame.copy()
            
        # Draw right detection
        if right_center is not None and self.last_right_detection is not None:
            right_result = self.right_detector.draw_detection(
                right_frame, 
                right_center, 
                self.last_right_detection[1],  # radius
                confidence
            )
        else:
            right_result = right_frame.copy()
            
        # Display 3D position if available
        if position_3d is not None:
            text = f"3D: ({position_3d[0]:.2f}, {position_3d[1]:.2f}, {position_3d[2]:.2f})m"
            cv2.putText(
                left_result, 
                text, 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 255), 
                2
            )
            
            # Draw epipolar line on right frame
            if left_center is not None:
                cv2.line(
                    right_result,
                    (0, left_center[1]),
                    (right_frame.shape[1], left_center[1]),
                    (255, 0, 0),
                    1
                )
                
        return left_result, right_result
    
    def get_last_3d_position(self) -> Optional[np.ndarray]:
        """Get the last valid 3D position."""
        return self.last_3d_position.copy() if self.last_3d_position is not None else None
    
    def calibrate_from_roi(
        self, 
        left_frame: np.ndarray, 
        right_frame: np.ndarray, 
        left_roi: Tuple[int, int, int, int],
        right_roi: Tuple[int, int, int, int]
    ) -> None:
        """
        Calibrate ball detectors using ROIs from left and right frames.
        
        Args:
            left_frame: Left camera frame
            right_frame: Right camera frame
            left_roi: Region of interest in left frame
            right_roi: Region of interest in right frame
        """
        # Calibrate left and right detectors
        self.left_detector.calibrate_from_roi(left_frame, left_roi)
        self.right_detector.calibrate_from_roi(right_frame, right_roi)
        
        # Enable ROI tracking for both detectors
        self.left_detector.enable_roi_tracking(True, left_roi)
        self.right_detector.enable_roi_tracking(True, right_roi) 