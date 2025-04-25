#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
3D Kalman Filter Service Module
This module contains the Kalman3DService class for Kalman filtering of 3D ball coordinates.
"""

import logging
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
import cv2


class Kalman3DService:
    """
    Service class for applying Kalman filtering to 3D ball tracking data.
    """

    def __init__(self, settings: dict = None):
        """
        Initialize 3D Kalman filter processor.
        
        Args:
            settings: Dictionary containing Kalman filter settings
        """
        # Set default values
        settings = settings or {}
        
        # Extract settings with defaults
        self.dt = settings.get("dt", 0.033)  # 30fps by default
        self.process_noise = settings.get("process_noise", 0.01)
        self.measurement_noise = settings.get("measurement_noise", 0.1)
        self.reset_threshold = settings.get("reset_threshold", 5.0)  # m/s jump threshold
        self.velocity_decay = settings.get("velocity_decay", 0.98)
        self.min_updates_required = settings.get("min_updates", 5)
        self.gravity = settings.get("gravity", 9.81)  # m/s^2
        self.use_physics_model = settings.get("use_physics_model", True)
        
        # State tracking
        self.kalman = None
        self.is_initialized = False
        self.update_count = 0
        self.last_state = None
        self.last_pos = None
        self.last_vel = None
        
        # Position and velocity history for visualization and analysis
        self.position_history = []
        self.velocity_history = []
        self.max_history_length = settings.get("max_history_length", 60)  # 2 seconds at 30fps
        
        # Initialize the Kalman filter
        self._init_kalman_filter()
        
        logging.info(f"3D Kalman processor initialized with dt={self.dt}, "
                    f"process_noise={self.process_noise}, "
                    f"measurement_noise={self.measurement_noise}, "
                    f"physics_model={'enabled' if self.use_physics_model else 'disabled'}")

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
            self.velocity_decay = settings["velocity_decay"]
        if "max_history_length" in settings:
            self.max_history_length = settings["max_history_length"]
        if "gravity" in settings:
            self.gravity = settings["gravity"]
        if "use_physics_model" in settings:
            self.use_physics_model = settings["use_physics_model"]
            
        # Re-initialize the filter with updated parameters
        self._init_kalman_filter()
        
        logging.info(f"3D Kalman parameters updated: dt={self.dt}, "
                   f"process_noise={self.process_noise}, "
                   f"measurement_noise={self.measurement_noise}, "
                   f"physics_model={'enabled' if self.use_physics_model else 'disabled'}")

    def _init_kalman_filter(self) -> None:
        """
        Initialize 3D Kalman filter for position and velocity tracking.
        State vector: [x, y, z, vx, vy, vz]
        Measurement vector: [x, y, z]
        """
        # Create 6D state Kalman filter (x, y, z, vx, vy, vz)
        self.kalman = cv2.KalmanFilter(6, 3)
        
        # Transition matrix (A) - with physics model if enabled
        if self.use_physics_model:
            # Apply constant velocity model with gravity in z-axis
            self.kalman.transitionMatrix = np.array([
                [1, 0, 0, self.dt, 0, 0],
                [0, 1, 0, 0, self.dt, 0],
                [0, 0, 1, 0, 0, self.dt],
                [0, 0, 0, self.velocity_decay, 0, 0],
                [0, 0, 0, 0, self.velocity_decay, 0],
                [0, 0, 0, 0, 0, self.velocity_decay]
            ], np.float32)
            
            # Add gravity effect to z-component (vz -= g*dt)
            # We don't modify the z directly because measurement will correct it
            self.gravity_vector = np.zeros((6, 1), np.float32)
            self.gravity_vector[5, 0] = -self.gravity * self.dt
            
        else:
            # Simple constant velocity model
            self.kalman.transitionMatrix = np.array([
                [1, 0, 0, self.dt, 0, 0],
                [0, 1, 0, 0, self.dt, 0],
                [0, 0, 1, 0, 0, self.dt],
                [0, 0, 0, self.velocity_decay, 0, 0],
                [0, 0, 0, 0, self.velocity_decay, 0],
                [0, 0, 0, 0, 0, self.velocity_decay]
            ], np.float32)
        
        # Measurement matrix (H) maps state to measurements [x, y, z]
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], np.float32)
        
        # Process noise covariance matrix (Q)
        # Higher values for velocity components (more dynamic)
        process_noise_cov = np.eye(6, dtype=np.float32) * self.process_noise
        process_noise_cov[3:, 3:] *= 10  # Higher for velocity components
        self.kalman.processNoiseCov = process_noise_cov
        
        # Measurement noise covariance matrix (R)
        # Can be adjusted based on triangulation confidence
        measurement_noise_cov = np.eye(3, dtype=np.float32) * self.measurement_noise
        self.kalman.measurementNoiseCov = measurement_noise_cov
        
        # Error covariance matrix (P)
        error_cov_post = np.eye(6, dtype=np.float32)
        error_cov_post[3:, 3:] *= 100  # Higher initial uncertainty for velocity
        self.kalman.errorCovPost = error_cov_post
        
        # Reset state tracking
        self.is_initialized = False
        self.update_count = 0
        self.last_state = None
        self.last_pos = None
        self.last_vel = None
        
        # Clear history
        self.position_history = []
        self.velocity_history = []
        
        logging.debug("3D Kalman filter initialized with appropriate noise matrices")

    def set_initial_state(self, position: np.ndarray, velocity: np.ndarray = None) -> None:
        """
        Set the initial state of the Kalman filter.
        
        Args:
            position: Initial 3D position [x, y, z]
            velocity: Initial 3D velocity [vx, vy, vz] (default: [0,0,0])
        """
        try:
            if velocity is None:
                velocity = np.zeros(3, dtype=np.float32)
            
            # Ensure position is a numpy array
            position = np.array(position, dtype=np.float32)
            velocity = np.array(velocity, dtype=np.float32)
            
            # Validate input dimensions
            if position.shape != (3,) or velocity.shape != (3,):
                logging.error(f"Invalid dimensions: position {position.shape}, velocity {velocity.shape}")
                return
                
            # Set initial state [x, y, z, vx, vy, vz]
            initial_state = np.concatenate([position, velocity]).reshape(6, 1)
            self.kalman.statePost = initial_state
            
            # Update tracking variables
            self.last_state = initial_state
            self.last_pos = position
            self.last_vel = velocity
            self.is_initialized = True
            self.update_count = 1
            
            # Add to history
            self.position_history = [position.copy()]
            self.velocity_history = [velocity.copy()]
            
            logging.info(f"Set initial 3D Kalman state: pos=({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}), "
                        f"vel=({velocity[0]:.2f}, {velocity[1]:.2f}, {velocity[2]:.2f})")
        except Exception as e:
            logging.error(f"Error setting initial 3D Kalman state: {e}")

    def predict(self, dt: float = None) -> Dict[str, np.ndarray]:
        """
        Predict the next state without measurement update.
        
        Args:
            dt: Time since last update (seconds). If None, uses default dt
            
        Returns:
            Dictionary with predicted position, velocity and state
        """
        if not self.is_initialized:
            logging.warning("Cannot predict: Kalman filter not initialized")
            return {
                "position": np.zeros(3),
                "velocity": np.zeros(3),
                "state": np.zeros(6)
            }
            
        # Update transition matrix with current dt if provided
        if dt is not None and dt > 0:
            current_dt = dt
            
            # Update transition matrix
            self.kalman.transitionMatrix[0, 3] = current_dt
            self.kalman.transitionMatrix[1, 4] = current_dt
            self.kalman.transitionMatrix[2, 5] = current_dt
            
            if self.use_physics_model:
                # Update gravity effect for new dt
                self.gravity_vector[5, 0] = -self.gravity * current_dt
                
            logging.debug(f"Updated Kalman dt to {current_dt:.4f}")
            
        # Apply gravity effect before prediction if using physics model
        if self.use_physics_model:
            # Apply external force (gravity)
            self.kalman.statePost = self.kalman.statePost + self.gravity_vector
            
        # Predict next state
        state_predicted = self.kalman.predict()
        
        # Extract position and velocity from state
        position = state_predicted[:3].flatten()
        velocity = state_predicted[3:].flatten()
        
        # Store predictions
        self.last_state = state_predicted
        self.last_pos = position
        self.last_vel = velocity
        
        return {
            "position": position,
            "velocity": velocity,
            "state": state_predicted.flatten()
        }

    def update(self, position: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Update the Kalman filter with a new position measurement.
        
        Args:
            position: Measured 3D position [x, y, z]
            
        Returns:
            Dictionary with updated position, velocity and state
        """
        if not self.is_initialized:
            # Initialize with the first measurement
            self.set_initial_state(position)
            return {
                "position": position,
                "velocity": np.zeros(3),
                "state": np.concatenate([position, np.zeros(3)])
            }
            
        try:
            # Convert to numpy array
            position = np.array(position, dtype=np.float32)
            
            # Check for valid measurement
            if not np.all(np.isfinite(position)):
                logging.warning(f"Invalid measurement with non-finite values: {position}")
                return {
                    "position": self.last_pos,
                    "velocity": self.last_vel,
                    "state": self.last_state.flatten() if self.last_state is not None else np.zeros(6)
                }
                
            # Check if measurement is too far from prediction (possible outlier)
            if self.last_pos is not None:
                distance = np.linalg.norm(position - self.last_pos)
                if distance > self.reset_threshold:
                    logging.warning(f"Measurement too far from prediction: {distance:.2f}m > {self.reset_threshold:.2f}m")
                    
                    # If multiple consecutive large jumps, reinitialize the filter
                    if self.update_count > self.min_updates_required:
                        logging.info("Continuing with existing filter despite distance")
                    else:
                        logging.info("Reinitializing filter with new position")
                        self.set_initial_state(position)
                        return {
                            "position": position,
                            "velocity": np.zeros(3),
                            "state": np.concatenate([position, np.zeros(3)])
                        }
            
            # Reshape measurement for OpenCV Kalman
            measurement = position.reshape(3, 1).astype(np.float32)
            
            # Perform the measurement update
            state_updated = self.kalman.correct(measurement)
            
            # Extract position and velocity
            updated_position = state_updated[:3].flatten()
            updated_velocity = state_updated[3:].flatten()
            
            # Store the updated state
            self.last_state = state_updated
            self.last_pos = updated_position
            self.last_vel = updated_velocity
            
            # Update counter and history
            self.update_count += 1
            
            # Add to history, maintaining max length
            self.position_history.append(updated_position.copy())
            self.velocity_history.append(updated_velocity.copy())
            
            if len(self.position_history) > self.max_history_length:
                self.position_history.pop(0)
                self.velocity_history.pop(0)
                
            logging.debug(f"Updated Kalman with measurement: meas=({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}), "
                         f"est=({updated_position[0]:.2f}, {updated_position[1]:.2f}, {updated_position[2]:.2f}), "
                         f"vel=({updated_velocity[0]:.2f}, {updated_velocity[1]:.2f}, {updated_velocity[2]:.2f})")
                
            return {
                "position": updated_position,
                "velocity": updated_velocity,
                "state": state_updated.flatten()
            }
            
        except Exception as e:
            logging.error(f"Error updating 3D Kalman: {e}")
            return {
                "position": self.last_pos if self.last_pos is not None else np.zeros(3),
                "velocity": self.last_vel if self.last_vel is not None else np.zeros(3),
                "state": self.last_state.flatten() if self.last_state is not None else np.zeros(6)
            }

    def get_position_history(self) -> List[np.ndarray]:
        """
        Get the history of estimated positions.
        
        Returns:
            List of 3D position arrays
        """
        return self.position_history.copy()

    def get_velocity_history(self) -> List[np.ndarray]:
        """
        Get the history of estimated velocities.
        
        Returns:
            List of 3D velocity arrays
        """
        return self.velocity_history.copy()

    def get_state(self) -> Dict[str, np.ndarray]:
        """
        Get the current state of the Kalman filter.
        
        Returns:
            Dictionary with current position, velocity and state
        """
        if not self.is_initialized or self.last_state is None:
            return {
                "position": np.zeros(3),
                "velocity": np.zeros(3),
                "state": np.zeros(6)
            }
            
        return {
            "position": self.last_pos,
            "velocity": self.last_vel,
            "state": self.last_state.flatten()
        }

    def is_ready(self) -> bool:
        """
        Check if the Kalman filter has enough updates to be reliable.
        
        Returns:
            True if the filter is initialized and has enough updates
        """
        return self.is_initialized and self.update_count >= self.min_updates_required

    def reset(self) -> None:
        """
        Reset the Kalman filter to initial state.
        """
        self._init_kalman_filter()
        logging.info("3D Kalman filter has been reset")

    def get_mahalanobis_distance(self, measurement: np.ndarray) -> float:
        """
        Calculate Mahalanobis distance between measurement and prediction.
        Used for outlier detection.
        
        Args:
            measurement: 3D position measurement [x, y, z]
            
        Returns:
            Mahalanobis distance (scalar)
        """
        if not self.is_initialized or self.last_state is None:
            return float('inf')
            
        # Get predicted position
        predicted_pos = self.last_pos
        
        # Calculate innovation (measurement - prediction)
        innovation = measurement - predicted_pos
        
        # Get measurement uncertainty (from error covariance)
        S = self.kalman.errorCovPre[:3, :3]
        
        # Calculate Mahalanobis distance
        try:
            S_inv = np.linalg.inv(S)
            distance = np.sqrt(innovation @ S_inv @ innovation.T)
            return float(distance)
        except np.linalg.LinAlgError:
            logging.warning("Could not calculate Mahalanobis distance: Singular matrix")
            return float('inf')

    def init_filter(self, initial_position=None):
        """
        Initialize the filter with an optional initial position.
        
        Args:
            initial_position: Initial 3D position [x, y, z] or None
            
        Returns:
            True if initialization successful
        """
        if initial_position is not None:
            self.set_initial_state(initial_position)
        else:
            # Initialize with zero state
            self.kalman.statePost = np.zeros((6, 1), dtype=np.float32)
            self.kalman.errorCovPost = np.eye(6, dtype=np.float32)
            self.last_pos = np.zeros(3)
            self.last_vel = np.zeros(3)
            self.last_state = np.zeros((6, 1))
            
        self.is_initialized = True
        return True 