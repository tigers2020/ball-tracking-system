#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
3D Kalman Filter for tennis ball tracking.
This module contains the Kalman3D class, which is used to track
the 3D position and velocity of a tennis ball.
"""

import numpy as np
from typing import Tuple, Optional


class Kalman3D:
    """
    3D Kalman Filter for tennis ball tracking.
    
    This class implements a Kalman filter to track the 3D position and
    velocity of a tennis ball. It accounts for gravity and can handle
    missing measurements.
    """
    
    def __init__(self, dt: float = 1/30.0, process_noise: float = 0.01, measurement_noise: float = 0.1):
        """
        Initialize the 3D Kalman filter.
        
        Args:
            dt: Time step between measurements (default: 1/30.0)
            process_noise: Process noise covariance factor (default: 0.01)
            measurement_noise: Measurement noise covariance factor (default: 0.1)
        """
        # Time step
        self.dt = dt
        
        # Gravity constant (m/s^2) - negative because gravity acts downward
        self.g = -9.81
        
        # State vector: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)
        
        # State transition matrix
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Control input matrix (for gravity)
        self.B = np.array([0, 0, 0.5 * dt**2, 0, 0, dt]).reshape(6, 1)
        
        # Control input (gravity)
        self.u = np.array([self.g])
        
        # Measurement matrix (we only measure position)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Process noise covariance
        q_pos = process_noise  # Position noise
        q_vel = process_noise * 10  # Velocity noise (higher)
        self.Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel])
        
        # Measurement noise covariance
        r = measurement_noise
        self.R = np.diag([r, r, r])
        
        # Error covariance matrix
        self.P = np.eye(6) * 1000  # Start with high uncertainty
        
        # Track initialization flag
        self.initialized = False
        
        # Track history
        self.history = []
        
        # Miss counter (for handling missing measurements)
        self.miss_counter = 0
        self.max_misses = 10  # Maximum number of consecutive missing measurements
        
        # Debug information
        self.debug_info = {}
        
    def reset(self):
        """
        Reset the Kalman filter.
        """
        self.state = np.zeros(6)
        self.P = np.eye(6) * 1000
        self.initialized = False
        self.history = []
        self.miss_counter = 0
        
    def init(self, pos: np.ndarray, vel: Optional[np.ndarray] = None):
        """
        Initialize the Kalman filter with a position and optionally a velocity.
        
        Args:
            pos: Initial position [x, y, z]
            vel: Initial velocity [vx, vy, vz] (optional)
        """
        self.state[:3] = pos
        
        if vel is not None:
            self.state[3:] = vel
        
        self.initialized = True
        self.history = [self.state.copy()]
        
    def predict(self):
        """
        Predict the next state using the motion model.
        """
        # Predict next state
        self.state = self.F @ self.state + self.B @ self.u
        
        # Update error covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Store prediction in debug info
        self.debug_info['predicted_state'] = self.state.copy()
        self.debug_info['predicted_covariance'] = self.P.copy()
        
    def update(self, measurement: Optional[np.ndarray] = None):
        """
        Update the filter with a new measurement.
        
        Args:
            measurement: 3D position measurement [x, y, z] or None if measurement is missing
        """
        if not self.initialized:
            if measurement is not None:
                self.init(measurement)
            return
        
        # First predict the next state
        self.predict()
        
        # If no measurement, increment miss counter and return
        if measurement is None:
            self.miss_counter += 1
            
            # If too many consecutive misses, mark as uninitialized
            if self.miss_counter > self.max_misses:
                self.initialized = False
                
            # Still add the predicted state to history
            self.history.append(self.state.copy())
            return
        
        # Reset miss counter if we have a valid measurement
        self.miss_counter = 0
        
        # Calculate Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state with measurement
        y = measurement - self.H @ self.state  # Measurement residual
        self.state = self.state + K @ y
        
        # Update error covariance
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P
        
        # Add current state to history
        self.history.append(self.state.copy())
        
        # Store update information in debug info
        self.debug_info['measurement'] = measurement
        self.debug_info['kalman_gain'] = K
        self.debug_info['residual'] = y
        self.debug_info['updated_state'] = self.state.copy()
        self.debug_info['updated_covariance'] = self.P.copy()
        
    def get_state(self) -> np.ndarray:
        """
        Get the current state vector.
        
        Returns:
            Current state vector [x, y, z, vx, vy, vz]
        """
        return self.state.copy()
    
    def get_position(self) -> np.ndarray:
        """
        Get the current position.
        
        Returns:
            Current position [x, y, z]
        """
        return self.state[:3].copy()
    
    def get_velocity(self) -> np.ndarray:
        """
        Get the current velocity.
        
        Returns:
            Current velocity [vx, vy, vz]
        """
        return self.state[3:].copy()
    
    def get_history(self) -> np.ndarray:
        """
        Get the history of states.
        
        Returns:
            Array of state vectors
        """
        return np.array(self.history)
    
    def set_dt(self, dt: float):
        """
        Set the time step.
        
        Args:
            dt: Time step in seconds
        """
        self.dt = dt
        
        # Update matrices that depend on dt
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt
        
        self.B[2] = 0.5 * dt**2
        self.B[5] = dt
        
    def set_gravity(self, g: float):
        """
        Set the gravity constant.
        
        Args:
            g: Gravity constant in m/s^2 (should be negative)
        """
        self.g = g
        self.u = np.array([self.g])
        
    def set_process_noise(self, q_pos: float, q_vel: float):
        """
        Set process noise parameters.
        
        Args:
            q_pos: Position process noise
            q_vel: Velocity process noise
        """
        self.Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel])
        
    def set_measurement_noise(self, r: float):
        """
        Set measurement noise parameter.
        
        Args:
            r: Measurement noise
        """
        self.R = np.diag([r, r, r])
        
    def predict_future_positions(self, n_steps: int) -> np.ndarray:
        """
        Predict future positions over n_steps.
        
        Args:
            n_steps: Number of time steps to predict into the future
            
        Returns:
            Array of predicted 3D positions
        """
        if not self.initialized:
            return np.array([])
        
        # Current state
        current_state = self.state.copy()
        
        # Storage for future positions
        future_positions = np.zeros((n_steps, 3))
        
        # Prediction loop
        for i in range(n_steps):
            # Predict next state
            current_state = self.F @ current_state + self.B @ self.u
            
            # Store position
            future_positions[i] = current_state[:3]
            
        return future_positions
    
    def smooth_trajectory(self, alpha: float = 0.5) -> np.ndarray:
        """
        Apply exponential smoothing to the trajectory history.
        
        Args:
            alpha: Smoothing factor (0 < alpha < 1)
            
        Returns:
            Smoothed trajectory
        """
        if len(self.history) < 2:
            return np.array(self.history)
        
        # Convert history to numpy array
        history_array = np.array(self.history)
        
        # Extract positions
        positions = history_array[:, :3]
        
        # Smooth positions using exponential smoothing
        smoothed = np.zeros_like(positions)
        smoothed[0] = positions[0]
        
        for i in range(1, len(positions)):
            smoothed[i] = alpha * positions[i] + (1 - alpha) * smoothed[i-1]
            
        return smoothed 