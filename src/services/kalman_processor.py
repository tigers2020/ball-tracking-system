#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kalman Filter Processor Module
This module contains the KalmanProcessor class for Kalman filtering ball coordinates.
"""

import logging
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
import cv2


class KalmanProcessor:
    """
    Service class for applying Kalman filtering to ball tracking data.
    """

    def __init__(self, settings: dict = None):
        """
        Initialize Kalman filter processor for both left and right camera feed.
        
        Args:
            settings: Dictionary containing Kalman filter settings
        """
        # Set default values
        settings = settings or {}
        
        # Extract settings with defaults
        self.dt = settings.get("dt", 0.1)
        self.process_noise = settings.get("process_noise", 0.02)
        self.measurement_noise = settings.get("measurement_noise", 0.1)
        self.reset_threshold = settings.get("reset_threshold", 100.0)
        self.velocity_decay_factor = settings.get("velocity_decay", 0.98)
        self.position_memory_factor = settings.get("position_memory", 0.7)
        self.min_updates_required = 5  # Keep this as a constant for now
        
        # Initialize Kalman filters for left and right cameras
        self.kalman_left = None
        self.kalman_right = None
        
        # Flags to check if filters are ready
        self.left_filter_ready = False
        self.right_filter_ready = False
        
        # Update counters
        self.left_update_count = 0
        self.right_update_count = 0
        
        # Last valid positions and states
        self.last_left_pos = None
        self.last_right_pos = None
        self.last_left_state = None
        self.last_right_state = None
        
        # Position history for trajectory visualization
        self.position_history = {
            "left": [],
            "right": []
        }
        self.max_history_length = 30  # Maximum number of positions to store in history
        
        # Initialize the filters
        self._init_kalman_filters()
        
        logging.info(f"Kalman processor initialized with dt={self.dt}, process_noise={self.process_noise}, "
                    f"measurement_noise={self.measurement_noise}, reset_threshold={self.reset_threshold}, "
                    f"velocity_decay={self.velocity_decay_factor}, position_memory={self.position_memory_factor}")

    def update_params(self, settings: dict) -> None:
        """
        Update Kalman filter parameters from settings dictionary.
        
        Args:
            settings: Dictionary containing updated Kalman filter settings
        """
        # Update settings if provided
        if "dt" in settings:
            self.dt = settings["dt"]
        if "process_noise" in settings:
            self.process_noise = settings["process_noise"]
        if "measurement_noise" in settings:
            self.measurement_noise = settings["measurement_noise"]
        if "reset_threshold" in settings:
            self.reset_threshold = settings["reset_threshold"]
        if "velocity_decay" in settings:
            self.velocity_decay_factor = settings["velocity_decay"]
        if "position_memory" in settings:
            self.position_memory_factor = settings["position_memory"]
        if "max_history_length" in settings:
            self.max_history_length = settings["max_history_length"]
            
        # Re-initialize the filters with updated parameters
        self._init_kalman_filters()
        
        logging.info(f"Kalman parameters updated: dt={self.dt}, process_noise={self.process_noise}, "
                   f"measurement_noise={self.measurement_noise}, reset_threshold={self.reset_threshold}, "
                   f"velocity_decay={self.velocity_decay_factor}, position_memory={self.position_memory_factor}")

    def _init_kalman_filters(self) -> None:
        """
        Initialize Kalman filter for 2D position and velocity tracking.
        State vector: [x, y, vx, vy]
        Measurement vector: [x, y]
        """
        # Left camera Kalman filter
        self.kalman_left = cv2.KalmanFilter(4, 2)
        self.kalman_left.transitionMatrix = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, self.velocity_decay_factor, 0],  # Apply velocity decay
            [0, 0, 0, self.velocity_decay_factor]   # Apply velocity decay
        ], np.float32)
        
        # Right camera Kalman filter
        self.kalman_right = cv2.KalmanFilter(4, 2)
        self.kalman_right.transitionMatrix = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, self.velocity_decay_factor, 0],  # Apply velocity decay
            [0, 0, 0, self.velocity_decay_factor]   # Apply velocity decay
        ], np.float32)
        
        # Set measurement matrix (H) for both filters
        # This maps the state vector [x, y, vx, vy] to the measurement vector [x, y]
        measurement_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        self.kalman_left.measurementMatrix = measurement_matrix
        self.kalman_right.measurementMatrix = measurement_matrix
        
        # Process noise covariance matrix (Q)
        # Higher values lead to more dynamic/responsive but less stable tracking
        process_noise_cov = np.eye(4, dtype=np.float32) * self.process_noise
        process_noise_cov[2:, 2:] *= 10  # Higher process noise for velocity components
        self.kalman_left.processNoiseCov = process_noise_cov
        self.kalman_right.processNoiseCov = process_noise_cov
        
        # Measurement noise covariance matrix (R)
        # Higher values mean less trust in measurements
        measurement_noise_cov = np.eye(2, dtype=np.float32) * self.measurement_noise
        self.kalman_left.measurementNoiseCov = measurement_noise_cov
        self.kalman_right.measurementNoiseCov = measurement_noise_cov
        
        # Initial posterior error covariance matrix (P)
        error_cov_post = np.eye(4, dtype=np.float32)
        error_cov_post[2:, 2:] *= 100  # Higher uncertainty for velocity components
        self.kalman_left.errorCovPost = error_cov_post.copy()
        self.kalman_right.errorCovPost = error_cov_post.copy()
        
        # Reset filter ready flags and counters
        self.left_filter_ready = False
        self.right_filter_ready = False
        self.left_update_count = 0
        self.right_update_count = 0
        
        # Clear position history
        self.position_history = {
            "left": [],
            "right": []
        }
        
        logging.debug("Kalman filters initialized with process noise and measurement noise matrices")

    def set_initial_state(self, camera: str, x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> None:
        """
        Set the initial state of the Kalman filter.
        
        Args:
            camera: Camera identifier ('left' or 'right')
            x: Initial x-coordinate
            y: Initial y-coordinate
            vx: Initial velocity in x-direction (default: 0.0)
            vy: Initial velocity in y-direction (default: 0.0)
        """
        try:
            if camera.lower() == 'left':
                kalman = self.kalman_left
                self.left_filter_ready = False
                self.left_update_count = 0
                self.last_left_pos = np.array([x, y], dtype=np.float32)
                # Add to position history
                if not self.position_history["left"]:
                    self.position_history["left"] = [(int(x), int(y))]
            elif camera.lower() == 'right':
                kalman = self.kalman_right
                self.right_filter_ready = False
                self.right_update_count = 0
                self.last_right_pos = np.array([x, y], dtype=np.float32)
                # Add to position history
                if not self.position_history["right"]:
                    self.position_history["right"] = [(int(x), int(y))]
            else:
                logging.error(f"Invalid camera identifier: {camera}")
                return
                
            # Set initial state
            kalman.statePost = np.array([x, y, vx, vy], dtype=np.float32).reshape(4, 1)
            
            logging.info(f"Set initial state for {camera} camera: pos=({x}, {y}), vel=({vx}, {vy})")
        except Exception as e:
            logging.error(f"Error setting initial Kalman state for {camera} camera: {e}")

    def update(self, camera: str, x: float, y: float, dt: float = None) -> Tuple[float, float, float, float]:
        """
        Update the Kalman filter with a new measurement and get updated state.
        
        Args:
            camera: Camera identifier ('left' or 'right')
            x: Measured x-coordinate
            y: Measured y-coordinate
            dt: Time since last update (seconds). If None, uses default dt
            
        Returns:
            Tuple of (predicted_x, predicted_y, velocity_x, velocity_y)
        """
        try:
            # Update transition matrix with actual dt if provided
            if dt is not None and dt > 0:
                # Only update if dt is a positive value
                current_dt = dt
                
                # Update the transition matrix with the current dt
                if camera.lower() == 'left':
                    self.kalman_left.transitionMatrix = np.array([
                        [1, 0, current_dt, 0],
                        [0, 1, 0, current_dt],
                        [0, 0, self.velocity_decay_factor, 0],
                        [0, 0, 0, self.velocity_decay_factor]
                    ], np.float32)
                    logging.debug(f"Updated left Kalman dt to {current_dt:.4f}")
                elif camera.lower() == 'right':
                    self.kalman_right.transitionMatrix = np.array([
                        [1, 0, current_dt, 0],
                        [0, 1, 0, current_dt],
                        [0, 0, self.velocity_decay_factor, 0],
                        [0, 0, 0, self.velocity_decay_factor]
                    ], np.float32)
                    logging.debug(f"Updated right Kalman dt to {current_dt:.4f}")
            
            # Select the appropriate Kalman filter
            if camera.lower() == 'left':
                kalman = self.kalman_left
                last_pos = self.last_left_pos
                update_count = self.left_update_count
            elif camera.lower() == 'right':
                kalman = self.kalman_right
                last_pos = self.last_right_pos
                update_count = self.right_update_count
            else:
                logging.error(f"Invalid camera identifier: {camera}")
                return (x, y, 0.0, 0.0)
            
            # Skip first prediction if initialization state
            if update_count == 0:
                kalman.statePost = np.array([x, y, 0, 0], dtype=np.float32).reshape(4, 1)
                
                # Add to position history
                self.position_history[camera.lower()].append((int(x), int(y)))
                
                # Update counter and store last position
                if camera.lower() == 'left':
                    self.left_update_count = 1
                    self.last_left_pos = np.array([x, y], dtype=np.float32)
                    self.last_left_state = kalman.statePost.copy()
                else:
                    self.right_update_count = 1
                    self.last_right_pos = np.array([x, y], dtype=np.float32)
                    self.last_right_state = kalman.statePost.copy()
                
                return (x, y, 0.0, 0.0)
                
            # Check if measurement is too far from last position (indicates tracking jump)
            if last_pos is not None:
                dist = np.sqrt((x - last_pos[0])**2 + (y - last_pos[1])**2)
                if dist > self.reset_threshold:
                    logging.warning(f"Measurement jump detected for {camera} camera: dist={dist:.2f} > threshold={self.reset_threshold}")
                    # Re-initialize the filter with current position
                    self.set_initial_state(camera, x, y)
                    return (x, y, 0.0, 0.0)
                
            # Perform prediction and update
            prediction = kalman.predict()
            
            # Create a measurement vector from actual coordinates
            measurement = np.array([x, y], dtype=np.float32).reshape(2, 1)
            
            # Perform the correction phase
            correction = kalman.correct(measurement)
            
            # Extract corrected state components
            corrected_state = correction.copy()
            corrected_x = corrected_state[0, 0]
            corrected_y = corrected_state[1, 0]
            velocity_x = corrected_state[2, 0]
            velocity_y = corrected_state[3, 0]
            
            # Add to position history
            position_tuple = (int(corrected_x), int(corrected_y))
            history_list = self.position_history[camera.lower()]
            
            # Only add if position is significantly different from last one
            if not history_list or np.sqrt((position_tuple[0] - history_list[-1][0])**2 + 
                                       (position_tuple[1] - history_list[-1][1])**2) > 1.0:
                history_list.append(position_tuple)
                # Trim history list if it exceeds maximum length
                if len(history_list) > self.max_history_length:
                    history_list = history_list[-self.max_history_length:]
                self.position_history[camera.lower()] = history_list
            
            # Update counter and check if filter is ready
            if camera.lower() == 'left':
                self.left_update_count += 1
                if self.left_update_count >= self.min_updates_required:
                    self.left_filter_ready = True
                
                # Store last position and state
                self.last_left_pos = np.array([corrected_x, corrected_y], dtype=np.float32)
                self.last_left_state = corrected_state.copy()
            elif camera.lower() == 'right':
                self.right_update_count += 1
                if self.right_update_count >= self.min_updates_required:
                    self.right_filter_ready = True
                
                # Store last position and state
                self.last_right_pos = np.array([corrected_x, corrected_y], dtype=np.float32)
                self.last_right_state = corrected_state.copy()
            
            logging.debug(f"Updated {camera} Kalman filter: pos=({corrected_x:.2f}, {corrected_y:.2f}), "
                        f"vel=({velocity_x:.2f}, {velocity_y:.2f}), count={update_count+1}")
            
            return (corrected_x, corrected_y, velocity_x, velocity_y)
            
        except Exception as e:
            logging.error(f"Error updating Kalman filter for {camera} camera: {e}")
            return (x, y, 0.0, 0.0)

    def get_position_history(self, camera: str) -> List[Tuple[int, int]]:
        """
        Get position history for the specified camera.
        
        Args:
            camera: Camera identifier ('left' or 'right')
            
        Returns:
            List of (x, y) position tuples or empty list if no history
        """
        if camera.lower() in self.position_history:
            return self.position_history[camera.lower()]
        return []

    def get_prediction(self, camera: str) -> Optional[Tuple[float, float, float, float]]:
        """
        Get the Kalman prediction for the next state without adding a new measurement.
        
        Args:
            camera: Camera identifier ('left' or 'right')
            
        Returns:
            Tuple of (predicted_x, predicted_y, velocity_x, velocity_y) or None if filter is not ready
        """
        try:
            # Check filter readiness
            if camera.lower() == 'left':
                if not self.left_filter_ready:
                    logging.debug(f"Left Kalman filter not ready yet ({self.left_update_count}/{self.min_updates_required} updates)")
                    return None
                kalman = self.kalman_left
                last_state = self.last_left_state
            elif camera.lower() == 'right':
                if not self.right_filter_ready:
                    logging.debug(f"Right Kalman filter not ready yet ({self.right_update_count}/{self.min_updates_required} updates)")
                    return None
                kalman = self.kalman_right
                last_state = self.last_right_state
            else:
                logging.error(f"Invalid camera identifier: {camera}")
                return None
            
            # 필터가 아직 없으면 None 반환
            if kalman is None:
                logging.debug(f"Kalman filter for {camera} camera is None")
                return None
                
            # statePre 직접 접근 - 예측 전 상태
            state = getattr(kalman, "statePre", None)
            if state is None:
                logging.debug(f"statePre not available for {camera} camera")
                return None
                
            # statePre는 cv2-Mat - numpy 배열로 변환
            state_arr = np.array(state, dtype=float).flatten()
            if state_arr.size < 4:  # pos(x,y) + vel(vx,vy)
                logging.debug(f"State array too small for {camera} camera: {state_arr.size}")
                return None
                
            # 처음 4개 요소만 튜플로 반환
            return tuple(state_arr[:4])
            
        except Exception as e:
            logging.error(f"Error getting Kalman prediction for {camera} camera: {e}")
            return None

    def reset_filters(self) -> None:
        """
        Reset both Kalman filters to initial state.
        """
        try:
            self._init_kalman_filters()
            self.last_left_pos = None
            self.last_right_pos = None
            self.last_left_state = None
            self.last_right_state = None
            # Clear position history
            self.position_history = {
                "left": [],
                "right": []
            }
            logging.info("Kalman filters have been reset")
        except Exception as e:
            logging.error(f"Error resetting Kalman filters: {e}")
            
    def reset(self) -> None:
        """
        Reset both Kalman filters to initial state.
        This is an alias for reset_filters() for compatibility.
        """
        self.reset_filters()

    def is_filter_ready(self, camera: str) -> bool:
        """
        Check if the Kalman filter for the specified camera is ready.
        
        Args:
            camera: Camera identifier ('left' or 'right')
            
        Returns:
            True if the filter is ready, False otherwise
        """
        if camera.lower() == 'left':
            return self.left_filter_ready
        elif camera.lower() == 'right':
            return self.right_filter_ready
        else:
            logging.error(f"Invalid camera identifier: {camera}")
            return False 