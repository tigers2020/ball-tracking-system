#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
3D Kalman filter for tennis ball tracking.
This module contains implementation of Kalman filter for tracking tennis ball in 3D space.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict


class BallKalmanFilter:
    """
    Kalman filter for tracking tennis ball in 3D space.
    
    This implementation uses a constant acceleration model for ball tracking.
    State vector: [x, y, z, vx, vy, vz, ax, ay, az]
    where (x, y, z) is the position, (vx, vy, vz) is the velocity, and (ax, ay, az) is the acceleration.
    """
    
    def __init__(
        self,
        process_noise_scale: float = 0.1,
        measurement_noise_scale: float = 0.1,
        dt: float = 1.0/30.0
    ):
        """
        Initialize Kalman filter for ball tracking.
        
        Args:
            process_noise_scale: Scale factor for process noise
            measurement_noise_scale: Scale factor for measurement noise
            dt: Time step between frames (default: 1/30 seconds for 30 fps)
        """
        # State dimension: 9 (x, y, z, vx, vy, vz, ax, ay, az)
        self.state_dim = 9
        
        # Measurement dimension: 3 (x, y, z)
        self.measurement_dim = 3
        
        # Time step
        self.dt = dt
        
        # Flag to check if filter is initialized
        self.initialized = False
        
        # State vector: [x, y, z, vx, vy, vz, ax, ay, az]
        self.x = np.zeros((self.state_dim, 1))
        
        # State covariance matrix
        self.P = np.eye(self.state_dim)
        
        # Define the state transition matrix (constant acceleration model)
        self.F = np.eye(self.state_dim)
        # Position update: x += vx*dt + 0.5*ax*dt^2, etc.
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt
        self.F[0, 6] = 0.5 * dt**2
        self.F[1, 7] = 0.5 * dt**2
        self.F[2, 8] = 0.5 * dt**2
        # Velocity update: vx += ax*dt, etc.
        self.F[3, 6] = dt
        self.F[4, 7] = dt
        self.F[5, 8] = dt
        
        # Measurement matrix (we only measure position)
        self.H = np.zeros((self.measurement_dim, self.state_dim))
        self.H[0, 0] = 1.0  # x
        self.H[1, 1] = 1.0  # y
        self.H[2, 2] = 1.0  # z
        
        # Process noise covariance matrix
        self.Q = np.eye(self.state_dim) * process_noise_scale
        
        # Measurement noise covariance matrix
        self.R = np.eye(self.measurement_dim) * measurement_noise_scale
        
        # Kalman gain
        self.K = np.zeros((self.state_dim, self.measurement_dim))
        
        # Identity matrix for covariance update
        self.I = np.eye(self.state_dim)
        
        # For tracking the ball trajectory
        self.trajectory = []
        
        # Store bounces
        self.bounces = []
        
    def reset(self):
        """Reset the Kalman filter state."""
        self.initialized = False
        self.x = np.zeros((self.state_dim, 1))
        self.P = np.eye(self.state_dim)
        self.trajectory = []
        self.bounces = []
        
    def init(self, measurement: np.ndarray):
        """
        Initialize the Kalman filter with the first measurement.
        
        Args:
            measurement: Initial position measurement [x, y, z]
        """
        # Initialize state with the measurement
        self.x[0:3, 0] = measurement
        
        # Initialize velocities and accelerations to zero
        self.x[3:, 0] = 0.0
        
        # Reset covariance matrix
        self.P = np.eye(self.state_dim)
        
        # Mark as initialized
        self.initialized = True
        
        # Add to trajectory
        self.trajectory.append(self.x[0:3, 0].copy())
        
    def predict(self):
        """
        Predict step of Kalman filter.
        This estimates the new state before incorporating a measurement.
        
        Returns:
            Predicted state vector [x, y, z, vx, vy, vz, ax, ay, az]
        """
        if not self.initialized:
            return self.x
        
        # Predict state
        self.x = self.F @ self.x
        
        # Include gravity effect on velocity (if needed)
        # self.x[5, 0] += 9.81 * self.dt  # Add gravity to vertical velocity (z-axis)
        
        # Update covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x
        
    def update(self, measurement: np.ndarray, confidence: float = 1.0):
        """
        Update step of Kalman filter.
        Incorporates a new measurement to refine the state estimate.
        
        Args:
            measurement: Position measurement [x, y, z]
            confidence: Confidence of the measurement (0.0 to 1.0)
                        Lower confidence will increase measurement noise
        
        Returns:
            Updated state vector [x, y, z, vx, vy, vz, ax, ay, az]
        """
        if not self.initialized:
            self.init(measurement)
            return self.x
        
        # Scale measurement noise by confidence (lower confidence = higher noise)
        R_scaled = self.R / max(confidence, 0.1)  # Prevent division by zero
        
        # Calculate Kalman gain
        PHT = self.P @ self.H.T
        self.K = PHT @ np.linalg.inv(self.H @ PHT + R_scaled)
        
        # Calculate measurement residual
        y = measurement.reshape(self.measurement_dim, 1) - self.H @ self.x
        
        # Update state estimate
        self.x = self.x + self.K @ y
        
        # Update error covariance
        self.P = (self.I - self.K @ self.H) @ self.P
        
        # Add to trajectory
        self.trajectory.append(self.x[0:3, 0].copy())
        
        # Check for bounce
        self._check_bounce()
        
        return self.x
    
    def _check_bounce(self, ground_height: float = 0.0, velocity_threshold: float = -2.0, bounce_height_threshold: float = 0.1):
        """
        Check if a bounce occurred.
        
        Args:
            ground_height: Height of the ground plane (default: 0.0)
            velocity_threshold: Vertical velocity threshold for bounce detection
            bounce_height_threshold: Height threshold above ground for bounce detection
        """
        # Need at least two points in trajectory
        if len(self.trajectory) < 2:
            return
        
        current_pos = self.x[0:3, 0]
        current_vel = self.x[3:6, 0]
        
        # Check if ball is close to ground and moving upward after being in downward trajectory
        if (abs(current_pos[2] - ground_height) < bounce_height_threshold and 
            current_vel[2] > 0 and 
            self.x[5, 0] > velocity_threshold):
            
            # Check if this bounce is significantly different from the last one
            if len(self.bounces) == 0 or np.linalg.norm(current_pos - self.bounces[-1]) > 0.5:
                self.bounces.append(current_pos.copy())
    
    def get_position(self) -> np.ndarray:
        """Get current estimated position."""
        return self.x[0:3, 0].copy()
    
    def get_velocity(self) -> np.ndarray:
        """Get current estimated velocity."""
        return self.x[3:6, 0].copy()
    
    def get_acceleration(self) -> np.ndarray:
        """Get current estimated acceleration."""
        return self.x[6:9, 0].copy()
    
    def get_state(self) -> np.ndarray:
        """Get complete state vector."""
        return self.x.copy()
    
    def get_trajectory(self) -> List[np.ndarray]:
        """Get the tracked trajectory."""
        return self.trajectory.copy()
    
    def get_bounces(self) -> List[np.ndarray]:
        """Get detected bounces."""
        return self.bounces.copy()


class BallTrajectoryPredictor:
    """
    Tennis ball trajectory predictor.
    Uses physics-based prediction to model ball flight accounting for gravity and drag.
    """
    
    def __init__(self, gravity: float = 9.81, drag_coefficient: float = 0.5, mass: float = 0.057):
        """
        Initialize trajectory predictor.
        
        Args:
            gravity: Gravitational acceleration (m/s^2)
            drag_coefficient: Air drag coefficient for tennis ball
            mass: Mass of tennis ball in kg (standard tennis ball: ~57g)
        """
        self.gravity = gravity
        self.drag_coefficient = drag_coefficient
        self.mass = mass
        self.radius = 0.0335  # Standard tennis ball radius: 3.35 cm
        
        # Air density at sea level (kg/m^3)
        self.air_density = 1.225
        
        # Calculated drag constant (air_density * drag_coefficient * area / 2 * mass)
        self.drag_constant = (self.air_density * self.drag_coefficient * 
                             (np.pi * self.radius**2) / (2 * self.mass))
    
    def predict_trajectory(
        self, 
        position: np.ndarray, 
        velocity: np.ndarray,
        time_points: int = 30,
        dt: float = 1/30
    ) -> List[np.ndarray]:
        """
        Predict ball trajectory for given initial conditions.
        
        Args:
            position: Initial position [x, y, z]
            velocity: Initial velocity [vx, vy, vz]
            time_points: Number of time points to predict
            dt: Time step in seconds
            
        Returns:
            List of predicted positions
        """
        trajectory = [position.copy()]
        pos = position.copy()
        vel = velocity.copy()
        
        for _ in range(time_points):
            # Calculate drag force
            speed = np.linalg.norm(vel)
            drag = self.drag_constant * speed**2
            
            # Calculate acceleration
            acc = np.zeros(3)
            if speed > 0:
                # Drag acts in opposite direction of velocity
                acc = -drag * vel / speed
            
            # Add gravity (assuming z is up)
            acc[2] -= self.gravity
            
            # Update velocity using acceleration
            vel = vel + acc * dt
            
            # Update position using velocity
            pos = pos + vel * dt
            
            # Bounce off the ground if needed (simple reflection model)
            if pos[2] < 0:
                pos[2] = -pos[2] * 0.8  # 0.8 = energy loss in bounce
                vel[2] = -vel[2] * 0.8
            
            trajectory.append(pos.copy())
        
        return trajectory
    
    def predict_bounce_location(
        self, 
        position: np.ndarray, 
        velocity: np.ndarray,
        ground_z: float = 0.0,
        max_iterations: int = 1000,
        dt: float = 1/100
    ) -> Optional[np.ndarray]:
        """
        Predict the location of the next bounce.
        
        Args:
            position: Current position [x, y, z]
            velocity: Current velocity [vx, vy, vz]
            ground_z: Height of the ground plane
            max_iterations: Maximum number of iterations
            dt: Time step for simulation
            
        Returns:
            Predicted bounce position or None if no bounce is predicted
        """
        pos = position.copy()
        vel = velocity.copy()
        
        # Skip if ball is moving upward and above ground
        if pos[2] > ground_z and vel[2] >= 0:
            return None
        
        for _ in range(max_iterations):
            # Calculate drag force
            speed = np.linalg.norm(vel)
            drag = self.drag_constant * speed**2
            
            # Calculate acceleration
            acc = np.zeros(3)
            if speed > 0:
                # Drag acts in opposite direction of velocity
                acc = -drag * vel / speed
            
            # Add gravity (assuming z is up)
            acc[2] -= self.gravity
            
            # Update velocity using acceleration
            vel = vel + acc * dt
            
            # Update position using velocity
            new_pos = pos + vel * dt
            
            # Check if crossed ground plane
            if pos[2] >= ground_z and new_pos[2] < ground_z:
                # Linear interpolation to find exact bounce location
                t = (ground_z - pos[2]) / (new_pos[2] - pos[2])
                bounce_pos = pos + t * (new_pos - pos)
                bounce_pos[2] = ground_z  # Ensure exactly on ground
                return bounce_pos
            
            pos = new_pos
            
            # If ball is moving upward and we've already checked below ground
            if pos[2] < ground_z and vel[2] > 0:
                return None
        
        return None 