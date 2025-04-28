#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bounce detector module.
This module contains the BounceDetector class, which is used to detect
bounces of the ball based on its trajectory.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class BounceDetector:
    """
    Bounce detector class.
    
    This class implements algorithms to detect bounces of a ball based
    on its 3D trajectory. It uses a combination of trajectory analysis,
    velocity changes, and height-based detection.
    """
    
    def __init__(self, court_height: float = 0.0, threshold_factor: float = 0.15):
        """
        Initialize the bounce detector.
        
        Args:
            court_height: Height of the court plane in world coordinates (default: 0.0)
            threshold_factor: Factor to determine the bounce threshold (default: 0.15)
        """
        # Court height (Z coordinate of the court plane)
        self.court_height = court_height
        
        # Bounce detection parameters
        self.velocity_change_threshold = 0.5  # Velocity change threshold for bounce detection
        self.height_threshold = 0.1  # Height threshold for bounce detection (m)
        self.threshold_factor = threshold_factor  # Factor to adjust bounce detection sensitivity
        self.min_bounce_interval = 0.2  # Minimum time between bounces (seconds)
        self.acceleration_threshold = 5.0  # Minimum vertical acceleration for bounce detection
        self.confidence_threshold = 0.6  # Minimum confidence for a bounce to be valid
        
        # Debug information
        self.debug_info = {}
        
    def set_court_height(self, height: float):
        """
        Set the court height.
        
        Args:
            height: Court height in world coordinates
        """
        self.court_height = height
        
    def detect_bounces(self, 
                      positions: np.ndarray, 
                      timestamps: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """
        Detect bounces in a trajectory.
        
        Args:
            positions: Array of 3D positions of shape (N, 3)
            timestamps: Array of timestamps corresponding to each position
            
        Returns:
            List of tuples (index, position) of detected bounces
        """
        if len(positions) < 3:
            return []
        
        # Calculate velocities
        velocities = self._calculate_velocities(positions, timestamps)
        
        # Calculate accelerations
        accelerations = self._calculate_accelerations(velocities, timestamps[1:])
        
        # Detect bounce candidates using different methods
        velocity_candidates = self._detect_velocity_changes(
            positions[1:-1],  # Exclude first and last positions
            velocities,
            accelerations,
            timestamps[1:-1]  # Timestamps corresponding to positions[1:-1]
        )
        
        height_candidates = self._detect_height_minima(
            positions[1:-1],
            timestamps[1:-1]
        )
        
        # Combine all candidates
        all_candidates = velocity_candidates + height_candidates
        
        # Calculate confidence scores for each candidate
        scored_candidates = self._score_candidates(
            all_candidates,
            positions,
            velocities,
            accelerations,
            timestamps
        )
        
        # Filter bounce candidates based on proximity to the court height
        # and minimum interval between bounces
        bounces = self._filter_candidates(scored_candidates, positions, timestamps)
        
        # Store debugger information
        self.debug_info = {
            'velocities': velocities,
            'accelerations': accelerations,
            'velocity_candidates': velocity_candidates,
            'height_candidates': height_candidates,
            'scored_candidates': scored_candidates,
            'bounces': bounces
        }
        
        return bounces
    
    def _calculate_velocities(self, positions: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """
        Calculate velocities from positions and timestamps.
        
        Args:
            positions: Array of 3D positions of shape (N, 3)
            timestamps: Array of timestamps
            
        Returns:
            Array of velocities of shape (N-1, 3)
        """
        velocities = np.zeros((len(positions) - 1, 3))
        for i in range(len(positions) - 1):
            dt = timestamps[i+1] - timestamps[i]
            if dt > 0:
                velocities[i] = (positions[i+1] - positions[i]) / dt
        
        return velocities
    
    def _calculate_accelerations(self, velocities: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """
        Calculate accelerations from velocities and timestamps.
        
        Args:
            velocities: Array of velocities of shape (N, 3)
            timestamps: Array of timestamps of shape (N+1)
            
        Returns:
            Array of accelerations of shape (N-1, 3)
        """
        accelerations = np.zeros((len(velocities) - 1, 3))
        for i in range(len(velocities) - 1):
            dt = timestamps[i+1] - timestamps[i]
            if dt > 0:
                accelerations[i] = (velocities[i+1] - velocities[i]) / dt
        
        return accelerations
    
    def _detect_velocity_changes(self, 
                               positions: np.ndarray,
                               velocities: np.ndarray, 
                               accelerations: np.ndarray,
                               timestamps: np.ndarray) -> List[Tuple[int, np.ndarray, float]]:
        """
        Detect significant changes in vertical velocity that might indicate bounces.
        
        Args:
            positions: Array of 3D positions of shape (N, 3)
            velocities: Array of velocities of shape (N+1, 3)
            accelerations: Array of accelerations of shape (N, 3)
            timestamps: Array of timestamps
            
        Returns:
            List of tuples (index, position, timestamp) of bounce candidates
        """
        bounce_candidates = []
        
        # We're interested in sign changes in the z-component of the velocity
        # Specifically, we're looking for transitions from negative to positive velocity
        # which indicates a bounce (ball moving downward, then upward)
        for i in range(1, len(velocities) - 1):
            # Check if the vertical velocity changes from negative to positive or is close to zero
            if (velocities[i-1][2] < -self.velocity_change_threshold and velocities[i][2] > 0) or \
               (abs(velocities[i][2]) < 0.1 and velocities[i-1][2] < -self.velocity_change_threshold):
                
                # Additional check: significant vertical acceleration
                if i-1 < len(accelerations) and accelerations[i-1][2] > self.acceleration_threshold:
                    # The bounce position is between positions[i-1] and positions[i]
                    # We approximate it by interpolation
                    idx = i - 1  # Index in the original positions array
                    bounce_candidates.append((idx, positions[i-1], timestamps[i-1], 'velocity'))
        
        return bounce_candidates
    
    def _detect_height_minima(self,
                             positions: np.ndarray,
                             timestamps: np.ndarray) -> List[Tuple[int, np.ndarray, float]]:
        """
        Detect local minima in height that might indicate bounces.
        
        Args:
            positions: Array of 3D positions of shape (N, 3)
            timestamps: Array of timestamps
            
        Returns:
            List of tuples (index, position, timestamp) of bounce candidates
        """
        bounce_candidates = []
        
        if len(positions) < 3:
            return bounce_candidates
            
        # Find local minima in height (z-coordinate)
        for i in range(1, len(positions) - 1):
            # Check if position[i] is a local minimum in height
            if (positions[i][2] < positions[i-1][2] and
                positions[i][2] < positions[i+1][2] and
                positions[i][2] < self.height_threshold + self.court_height):
                
                idx = i  # Index in the original positions array
                bounce_candidates.append((idx, positions[i], timestamps[i], 'height'))
                
        return bounce_candidates
    
    def _score_candidates(self,
                        candidates: List[Tuple[int, np.ndarray, float, str]],
                        positions: np.ndarray,
                        velocities: np.ndarray,
                        accelerations: np.ndarray,
                        timestamps: np.ndarray) -> List[Tuple[int, np.ndarray, float, float]]:
        """
        Calculate confidence scores for bounce candidates.
        
        Args:
            candidates: List of tuples (index, position, timestamp, method) of bounce candidates
            positions: Array of all 3D positions
            velocities: Array of all velocities
            accelerations: Array of all accelerations
            timestamps: Array of all timestamps
            
        Returns:
            List of tuples (index, position, timestamp, confidence) with calculated confidence scores
        """
        scored_candidates = []
        
        for idx, position, timestamp, method in candidates:
            # Initialize score components
            height_score = 0.0
            velocity_score = 0.0
            acceleration_score = 0.0
            
            # Calculate height score (higher when closer to court height)
            height_diff = abs(position[2] - self.court_height)
            if height_diff < self.height_threshold:
                height_score = 1.0 - (height_diff / self.height_threshold)
            
            # Calculate velocity score if we have valid indices
            if 0 <= idx < len(velocities) and idx+1 < len(velocities):
                prev_vel = velocities[idx][2]
                next_vel = velocities[idx+1][2]
                
                # Velocity should change from negative to positive
                if prev_vel < 0 and next_vel > 0:
                    velocity_magnitude = abs(next_vel - prev_vel)
                    velocity_score = min(1.0, velocity_magnitude / (2 * self.velocity_change_threshold))
            
            # Calculate acceleration score if we have valid indices
            if 0 <= idx < len(accelerations):
                acc = accelerations[idx][2]
                if acc > self.acceleration_threshold:
                    acceleration_score = min(1.0, acc / (2 * self.acceleration_threshold))
            
            # Combine scores with weights
            if method == 'velocity':
                # For velocity-based candidates, weight velocity change more heavily
                confidence = 0.3 * height_score + 0.5 * velocity_score + 0.2 * acceleration_score
            else:  # height
                # For height-based candidates, weight height more heavily
                confidence = 0.5 * height_score + 0.3 * velocity_score + 0.2 * acceleration_score
            
            scored_candidates.append((idx, position, timestamp, confidence))
        
        return scored_candidates
    
    def _filter_candidates(self, 
                        scored_candidates: List[Tuple[int, np.ndarray, float, float]], 
                        positions: np.ndarray,
                        timestamps: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """
        Filter bounce candidates based on confidence and minimum interval between bounces.
        
        Args:
            scored_candidates: List of tuples (index, position, timestamp, confidence)
            positions: Array of all 3D positions
            timestamps: Array of all timestamps
            
        Returns:
            List of tuples (index, position) of confirmed bounces
        """
        bounces = []
        
        if not scored_candidates:
            return []
        
        # Sort candidates by timestamp
        scored_candidates.sort(key=lambda x: x[2])
        
        # First filter by confidence threshold
        confident_candidates = [c for c in scored_candidates if c[3] >= self.confidence_threshold]
        
        last_bounce_time = -self.min_bounce_interval * 2
        
        for idx, position, timestamp, confidence in confident_candidates:
            # Check if enough time has passed since the last bounce
            if timestamp - last_bounce_time >= self.min_bounce_interval:
                # We have a valid bounce
                bounces.append((idx, position))
                last_bounce_time = timestamp
        
        return bounces
    
    def estimate_bounce_point(self, 
                            positions: np.ndarray, 
                            timestamps: np.ndarray, 
                            bounce_idx: int) -> Optional[Tuple[np.ndarray, float]]:
        """
        Estimate the exact bounce point by fitting a parabola to the trajectory.
        
        Args:
            positions: Array of 3D positions
            timestamps: Array of timestamps
            bounce_idx: Index of the bounce in the positions array
            
        Returns:
            Tuple of (bounce position, bounce time) or None if estimation fails
        """
        # We need at least 3 points around the bounce
        if bounce_idx <= 0 or bounce_idx >= len(positions) - 1:
            return None
        
        # Get points around the bounce
        # Use more points if available for better fitting
        start_idx = max(0, bounce_idx - 2)
        end_idx = min(len(positions), bounce_idx + 3)
        
        if end_idx - start_idx < 3:
            # Not enough points for fitting
            return None
        
        # Extract points and times
        points = positions[start_idx:end_idx]
        times = timestamps[start_idx:end_idx]
        
        # Shift times to start from 0 for numerical stability
        times_shifted = times - times[0]
        
        try:
            # Fit a parabola to the z-component: z = a*t^2 + b*t + c
            A = np.vstack([times_shifted**2, times_shifted, np.ones(len(times_shifted))]).T
            a, b, c = np.linalg.lstsq(A, points[:, 2], rcond=None)[0]
            
            # The minimum of the parabola occurs at t = -b/(2*a)
            # This is where the bounce happens
            if a > 0:  # Ensure it's a proper parabola opening upward
                bounce_time_shifted = -b / (2 * a)
                
                # Ensure the bounce time is within the time range
                if times_shifted[0] <= bounce_time_shifted <= times_shifted[-1]:
                    bounce_time = bounce_time_shifted + times[0]
                    
                    # Fit parabolas to x and y components for the same time range
                    A = np.vstack([times_shifted**2, times_shifted, np.ones(len(times_shifted))]).T
                    ax, bx, cx = np.linalg.lstsq(A, points[:, 0], rcond=None)[0]
                    ay, by, cy = np.linalg.lstsq(A, points[:, 1], rcond=None)[0]
                    
                    # Calculate the bounce position
                    x = ax * bounce_time_shifted**2 + bx * bounce_time_shifted + cx
                    y = ay * bounce_time_shifted**2 + by * bounce_time_shifted + cy
                    z = a * bounce_time_shifted**2 + b * bounce_time_shifted + c
                    
                    # Ensure z is close to court height
                    if abs(z - self.court_height) > 0.3:  # Arbitrary threshold
                        z = self.court_height  # Force the bounce to be on the court
                    
                    bounce_position = np.array([x, y, z])
                    return bounce_position, bounce_time
        
        except np.linalg.LinAlgError:
            # Fitting failed
            pass
        
        # If we get here, fitting failed or constraints weren't met
        # Return the position and time at the detected bounce index
        return positions[bounce_idx], timestamps[bounce_idx] 