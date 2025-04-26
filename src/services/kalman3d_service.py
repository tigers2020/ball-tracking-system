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
from src.utils.constants import ANALYSIS, STEREO


class Kalman3DService:
    """
    Service class for applying Kalman filtering to 3D ball tracking data.
    """

    def __init__(self, dt: float = 1/ANALYSIS.DEFAULT_FPS, 
                 process_noise: float = 2.5,
                 measurement_noise: float = 0.5,
                 reset_threshold: float = ANALYSIS.RESET_THRESHOLD,
                 min_updates_required: int = 3):
        """
        Initialize 3D Kalman filter for ball tracking.
        
        Args:
            dt: Time step between frames (seconds)
            process_noise: Process noise parameter (acceleration variance)
            measurement_noise: Measurement noise parameter
            reset_threshold: Distance threshold for filter reset (meters)
            min_updates_required: Minimum updates before allowing filter reset
        """
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.reset_threshold = reset_threshold
        self.min_updates_required = min_updates_required
        
        # 물리 모델 관련 변수 초기화
        self.use_physics_model = False
        self.gravity = ANALYSIS.GRAVITY
        self.velocity_decay = ANALYSIS.VELOCITY_DECAY
        self.max_history_length = ANALYSIS.MAX_HISTORY_LENGTH
        self.gravity_vector = np.zeros((6, 1), STEREO.DATA_TYPE)
        
        # State tracking variables
        self.is_initialized = False
        self.update_count = 0
        self.last_state = None
        self.last_pos = None
        self.last_vel = None
        self.position_history = []
        self.velocity_history = []
        
        # Initialize Kalman filter with 6 state variables (x, y, z, vx, vy, vz)
        # and 3 measurement variables (x, y, z)
        self.kalman = cv2.KalmanFilter(6, 3, 0, cv2.CV_32F)
        
        # 상태 전이 행렬 설정 (F)
        # x(k) = x(k-1) + vx(k-1)*dt
        # y(k) = y(k-1) + vy(k-1)*dt
        # z(k) = z(k-1) + vz(k-1)*dt
        # vx(k) = vx(k-1)
        # vy(k) = vy(k-1)
        # vz(k) = vz(k-1)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], np.float32)
        
        # 측정 행렬 설정 (H)
        # 위치만 측정 가능, 속도는 측정 불가능
        self.kalman.measurementMatrix = np.zeros((3, 6), np.float32)
        self.kalman.measurementMatrix[0, 0] = 1.0  # x
        self.kalman.measurementMatrix[1, 1] = 1.0  # y
        self.kalman.measurementMatrix[2, 2] = 1.0  # z
        
        # 프로세스 노이즈 공분산 설정 (Q)
        # 가속도에 의한 위치 및 속도 불확실성을 모델링
        # 테니스 공의 갑작스러운 속도 변화를 허용하기 위해 값을 조정
        
        # 가속도 분산 (더 높은 값은 더 빠른 적응 = 노이지한 추적, 더 낮은 값은 더 부드러운 추적 = 느린 적응)
        acceleration_variance = self.process_noise
        
        # 간소화된 continuous-time model을 사용한 프로세스 노이즈 행렬
        # Q_continuous를 이산화한 Q = G * G.T * a_var를 직접 사용
        dt2 = dt**2
        dt3 = dt**3
        
        # 프로세스 노이즈 행렬 구성
        # 참고: van Loan method와 같은 더 정확한 방법도 있으나, 
        # 이 단순화된 버전이 실제로 테니스 추적에 충분히 잘 작동함
        self.kalman.processNoiseCov = np.array([
            [dt3/3, 0, 0, dt2/2, 0, 0],
            [0, dt3/3, 0, 0, dt2/2, 0],
            [0, 0, dt3/3, 0, 0, dt2/2],
            [dt2/2, 0, 0, dt, 0, 0],
            [0, dt2/2, 0, 0, dt, 0],
            [0, 0, dt2/2, 0, 0, dt]
        ], np.float32) * acceleration_variance
        
        # 측정 노이즈 공분산 설정 (R)
        # 각 위치 좌표(x, y, z)의 측정 노이즈
        self.kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * self.measurement_noise
        
        # z축(높이)에 대한 노이즈는 추가로 증가 (일반적으로 높이 측정이 더 부정확)
        self.kalman.measurementNoiseCov[2, 2] *= 1.5
        
        # 사후 오차 공분산 초기화 (P_0)
        # 위치에 대한 낮은 불확실성, 속도에 대한 높은 불확실성으로 시작
        self.kalman.errorCovPost = np.diag([
            0.1, 0.1, 0.2,  # 위치 불확실성
            25.0, 25.0, 25.0  # 속도 불확실성
        ]).astype(np.float32)
        
        logging.debug(f"3D Kalman filter initialized with reset_threshold={self.reset_threshold}m and min_updates={self.min_updates_required}")

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
        # 고정된 reset_threshold를 유지하기 위해 설정에서 설정하지 않음
        # if "reset_threshold" in settings:
        #     self.reset_threshold = settings["reset_threshold"]
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
                   f"reset_threshold={self.reset_threshold}, "
                   f"physics_model={'enabled' if self.use_physics_model else 'disabled'}")

    def _init_kalman_filter(self) -> None:
        """
        Initialize 3D Kalman filter for position and velocity tracking.
        State vector: [x, y, z, vx, vy, vz]
        Measurement vector: [x, y, z]
        """
        # Create 6D state Kalman filter (x, y, z, vx, vy, vz)
        self.kalman = cv2.KalmanFilter(6, 3, 0, cv2.CV_32F)
        
        # Save critical parameters that should persist across reinitializations
        saved_reset_threshold = self.reset_threshold
        saved_min_updates = self.min_updates_required
        
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
        
        # 복원: 저장된 중요 매개변수 복원
        self.reset_threshold = saved_reset_threshold
        self.min_updates_required = saved_min_updates
        
        # Clear history
        self.position_history = []
        self.velocity_history = []
        
        logging.debug(f"3D Kalman filter initialized with reset_threshold={self.reset_threshold}m and min_updates={self.min_updates_required}")

    def set_initial_state(self, position: np.ndarray):
        """
        Initialize the Kalman filter state with a given position.
        
        Args:
            position: Initial 3D position [x, y, z]
        """
        self.is_initialized = True
        self.update_count = 1
        
        # 유효성 검사: 높이(z)값이 현실적인지 확인
        MAX_VALID_HEIGHT = ANALYSIS.MAX_VALID_HEIGHT  # 최대 유효 높이 (미터)
        if position[2] > MAX_VALID_HEIGHT:
            # 경고만 표시하고 원래 값 유지 (클램핑 제거)
            logging.warning(f"Initial height unusually high: {position[2]:.2f}m. Check camera calibration or triangulation.")
        elif position[2] < 0.0:
            logging.warning(f"Negative initial height: {position[2]:.2f}m. Setting to 0.0m.")
            position[2] = 0.0
        
        # 역사 초기화 또는 재설정
        self.position_history = [position.copy()]
        
        # 이전 상태 값이 있는 경우, 초기 속도 추정
        initial_velocity = np.zeros(3, dtype=np.float32)
        if self.last_pos is not None and self.position_history and len(self.position_history) > 1:
            # 위치 기록이 있으면 마지막 두 위치로부터 속도 추정
            try:
                # 이전 마지막 위치와 현재 위치 사이의 거리를 프레임 간격으로 나누어 속도 추정
                last_positions = self.position_history[-min(5, len(self.position_history)):]
                if len(last_positions) >= 2:
                    avg_displacement = np.zeros(3)
                    count = 0
                    
                    # 마지막 몇 프레임의 평균 변위 계산
                    for i in range(1, len(last_positions)):
                        displacement = last_positions[i] - last_positions[i-1]
                        distance = np.linalg.norm(displacement)
                        
                        # 합리적인 변위만 포함 (점프나 이상치 제외)
                        if distance < ANALYSIS.REASONABLE_DISPLACEMENT:  # 2m/frame 이하만 고려
                            avg_displacement += displacement
                            count += 1
                    
                    if count > 0:
                        # 초당 프레임 수로 나누어 미터/초 속도로 변환
                        initial_velocity = (avg_displacement / count) / self.dt
                        logging.info(f"Estimated initial velocity: {initial_velocity}")
                        
                        # 속도 크기 제한 (너무 빠른 초기 속도 방지)
                        speed = np.linalg.norm(initial_velocity)
                        if speed > ANALYSIS.MAX_BALL_SPEED:  # 30 m/s (108 km/h) 이상은 제한
                            logging.warning(f"Initial speed too high: {speed:.2f} m/s. Scaling down.")
                            initial_velocity = initial_velocity * (ANALYSIS.MAX_BALL_SPEED / speed)
            except Exception as e:
                logging.warning(f"Error estimating initial velocity: {e}")
                initial_velocity = np.zeros(3, dtype=np.float32)
        
        self.velocity_history = [initial_velocity.copy()]
        
        # Initialize state with position and velocity (position[0:3], velocity[3:6])
        state = np.zeros(6, dtype=np.float32)
        state[0:3] = position
        state[3:6] = initial_velocity
        
        # 상태 설정
        self.kalman.statePost = state.reshape(6, 1)
        
        # 초기 상태 불확실성 설정 - 위치는 낮은 불확실성, 속도는 높은 불확실성
        self.kalman.errorCovPost = np.diag([
            0.1, 0.1, 0.2,  # 낮은 위치 불확실성
            25.0, 25.0, 25.0  # 높은 속도 불확실성
        ]).astype(np.float32)
        
        # 측정 노이즈 및 프로세스 노이즈 업데이트 (재초기화시 필요할 수 있음)
        self.kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * self.measurement_noise
        
        # 위치만 측정되므로 측정 행렬은 첫 3개 상태만 연결
        self.kalman.measurementMatrix = np.zeros((3, 6), np.float32)
        self.kalman.measurementMatrix[0, 0] = 1.0
        self.kalman.measurementMatrix[1, 1] = 1.0
        self.kalman.measurementMatrix[2, 2] = 1.0
        
        # 현재 상태 저장
        self.last_state = self.kalman.statePost.copy()
        self.last_pos = position.copy()
        self.last_vel = initial_velocity.copy()

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

    def update(self, position: np.ndarray, confidence: float = 1.0) -> dict:
        """
        Update the Kalman filter with a new position measurement.
        
        Args:
            position: 3D position measurement [x, y, z]
            confidence: Confidence score of the measurement (0.0-1.0)
                        Lower values increase measurement noise
        
        Returns:
            dict: Dictionary containing filtered position and velocity
        """
        position = np.array(position, dtype=STEREO.DATA_TYPE)
        
        # Ensure position has correct shape
        if position.shape != (3,):
            position = position.flatten()[:3]
            
        # 로깅용으로 원래 측정값 저장
        original_position = position.copy()
        
        # Check for invalid or extremely large values
        if not np.all(np.isfinite(position)) or np.any(np.abs(position) > 1000):
            logging.warning(f"Invalid position detected: {position}. Using last valid position.")
            if self.last_pos is not None:
                position = self.last_pos.copy()
                confidence *= 0.1  # Significantly reduce confidence
            else:
                position = np.zeros(3, dtype=STEREO.DATA_TYPE)  # Use zeros as a fallback
                confidence *= 0.1  # Significantly reduce confidence
            
        # First measurement - initialize filter
        if not self.is_initialized:
            self.set_initial_state(position)
            return {
                "position": position,
                "velocity": np.zeros(3, dtype=STEREO.DATA_TYPE),
                "is_reliable": True,
                "measurement": position.copy(),
                "confidence": confidence
            }
            
        self.update_count += 1
            
        # 측정 거리가 임계값을 초과하는지 확인 (필터 재설정 조건)
        too_far = False
        distance = 0.0
        
        # 이전 위치가 있으면 거리 계산
        if self.last_pos is not None:
            distance = np.linalg.norm(position - self.last_pos)
            logging.debug(f"Distance from last position: {distance:.2f}m")
            
            # 충분한 업데이트 후에만 거리 기반 검사 적용
            if self.update_count > self.min_updates_required:
                # 전체 재설정 대신 거리 기반 신뢰도 조정
                if distance > self.reset_threshold:
                    too_far = True
                    logging.warning(f"Large distance detected: {distance:.2f}m > {self.reset_threshold:.2f}m threshold")
                    
                    # 거리에 따라 신뢰도 감소
                    # 거리가 threshold의 2배 이상이면 신뢰도를 0에 가깝게
                    distance_factor = min(distance / self.reset_threshold, 2.0)
                    confidence_scaling = max(0.05, 1.0 - (distance_factor - 1.0))
                    confidence *= confidence_scaling
                    
                    logging.info(f"Reduced confidence to {confidence:.2f} due to large displacement")
                
                # 신뢰도가 매우 낮으면 재초기화 고려
                if confidence < 0.05:  # 임계값을 0.1에서 0.05로 낮춤 (더 낮은 신뢰도까지 허용)
                    logging.debug("Very low confidence measurement, resetting filter")  # warning에서 debug로 변경
                    self.set_initial_state(position)
                    return {
                        "position": position,
                        "velocity": np.zeros(3, dtype=STEREO.DATA_TYPE),
                        "is_reliable": False,
                        "measurement": position.copy(),
                        "reset": True,
                        "confidence": confidence
                    }
        
        # Adjust measurement noise based on confidence
        if confidence < 1.0:
            # 기본 노이즈에 신뢰도 반비례 가중치 적용
            adjusted_noise = self.measurement_noise / confidence
            original_noise = self.kalman.measurementNoiseCov.copy()
            
            # 측정 노이즈 공분산 임시 조정
            self.kalman.measurementNoiseCov = np.eye(3, dtype=STEREO.DATA_TYPE) * adjusted_noise
            # Z축(높이)에 더 높은 노이즈 적용
            self.kalman.measurementNoiseCov[2, 2] *= 1.5
            
            logging.debug(f"Adjusted measurement noise to {adjusted_noise:.2f} (confidence: {confidence:.2f})")
        
        # Prediction step
        predicted_state = self.kalman.predict()
        
        try:
            # 측정 업데이트 수행 - 명시적으로 float32 타입 지정
            measurement = np.array(position, dtype=STEREO.DATA_TYPE).reshape(3, 1)
            corrected_state = self.kalman.correct(measurement)
            
            # 측정 노이즈 원래대로 복원 (신뢰도로 조정했던 경우)
            if confidence < 1.0:
                self.kalman.measurementNoiseCov = original_noise
            
            # Extract position and velocity from state
            filtered_position = corrected_state[:3].flatten()
            filtered_velocity = corrected_state[3:6].flatten()
        except cv2.error as e:
            logging.error(f"OpenCV error in Kalman update: {e}")
            
            # Fallback to predicted values when correction fails
            filtered_position = predicted_state[:3].flatten()
            filtered_velocity = predicted_state[3:6].flatten()
            
            # Reduce confidence further
            confidence *= 0.2
        
        # 현재 상태 저장 - 명시적으로 float32 타입 지정
        self.last_state = np.array(predicted_state.copy(), dtype=STEREO.DATA_TYPE)
        self.last_pos = filtered_position.copy()
        self.last_vel = filtered_velocity.copy()
        
        # 기록 저장 (시각화 및 분석용)
        self.position_history.append(filtered_position.copy())
        self.velocity_history.append(filtered_velocity.copy())
        
        # 역사 길이 제한
        max_history = ANALYSIS.MAX_HISTORY_LENGTH  # 2초 (60fps 기준)
        if len(self.position_history) > max_history:
            self.position_history.pop(0)
        if len(self.velocity_history) > max_history:
            self.velocity_history.pop(0)
            
        # 결과 반환
        return {
            "position": filtered_position,
            "velocity": filtered_velocity,
            "is_reliable": not too_far,
            "measurement": original_position,
            "confidence": confidence,
            "distance": distance
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