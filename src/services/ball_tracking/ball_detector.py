#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tennis ball detector module.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
import time


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
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (self.blur_size, self.blur_size), 0)
        
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
            # Use default HSV range
            mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        
        # Apply morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Combine with background subtraction for moving objects
        fg_mask = self.bg_subtractor.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Combine masks
        combined_mask = cv2.bitwise_and(mask, fg_mask)
        
        # Use blob detector for robust detection
        inverted_mask = cv2.bitwise_not(combined_mask)
        keypoints = self.blob_detector.detect(inverted_mask)
        
        best_center = None
        best_radius = None
        best_confidence = 0.0
        
        # Process contours
        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Fit circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            # Check if circle size is reasonable
            if radius < self.min_radius or radius > self.max_radius:
                continue
            
            # Check if circle is in motion (using background subtraction)
            motion_mask = cv2.circle(np.zeros_like(fg_mask), center, radius, 255, -1)
            motion_score = cv2.countNonZero(cv2.bitwise_and(fg_mask, motion_mask)) / cv2.countNonZero(motion_mask)
            
            # Calculate confidence score
            confidence = 0.4 * circularity + 0.3 * (1.0 - abs(radius - 15) / 30) + 0.3 * motion_score
            
            if confidence > best_confidence:
                best_center = center
                best_radius = radius
                best_confidence = confidence
                
        # Process keypoints from blob detector
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
                    
            # Update history
            self.detection_history.append((best_center, best_radius, best_confidence))
            if len(self.detection_history) > self.history_len:
                self.detection_history.pop(0)
                
            # Update last position and time
            self.last_position = best_center
            self.last_detection_time = current_time
            
            return best_center, best_radius, best_confidence
        else:
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
        Calibrate HSV color range from a region of interest.
        
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
        Calibrate HSV color range from regions of interest in both frames.
        
        Args:
            left_frame: Left camera frame
            right_frame: Right camera frame
            left_roi: Left region of interest as (x, y, width, height)
            right_roi: Right region of interest as (x, y, width, height)
        """
        self.left_detector.calibrate_from_roi(left_frame, left_roi)
        self.right_detector.calibrate_from_roi(right_frame, right_roi)
        
        # Share calibration between detectors
        for hsv_range in self.left_detector.hsv_ranges:
            self.right_detector.add_hsv_range(*hsv_range)
            
        for hsv_range in self.right_detector.hsv_ranges:
            self.left_detector.add_hsv_range(*hsv_range)
            
        # Enable adaptive mode
        self.left_detector.enable_adaptive_mode(True)
        self.right_detector.enable_adaptive_mode(True) 