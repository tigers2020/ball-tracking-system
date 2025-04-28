#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Triangulation Service module.
This module provides a service for triangulating 3D positions from stereo camera images.
"""

import logging
import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any

from src.utils.constants import STEREO
from src.services.ball_tracking.triangulation import (
    triangulate_points,
    calculate_projection_matrices,
    triangulate_with_confidence
)

class TriangulationService:
    """Service for triangulating 3D positions from stereo camera images."""
    
    def __init__(self, camera_settings: Optional[Dict[str, Any]] = None):
        """
        Initialize the triangulation service.
        
        Args:
            camera_settings: Optional camera configuration dictionary
        """
        # Camera parameters
        self.baseline = 1.0  # meters
        self.focal_length = 1000.0  # pixels
        self.principal_point_x = 320.0  # pixels
        self.principal_point_y = 240.0  # pixels
        self.image_width = 640  # pixels
        self.image_height = 480  # pixels
        
        # Camera matrices
        self.camera_matrix_left = None
        self.camera_matrix_right = None
        self.dist_coeffs_left = None
        self.dist_coeffs_right = None
        
        # Projection matrices
        self.proj_matrix_left = None
        self.proj_matrix_right = None
        
        # Rotation and translation
        self.R = np.eye(3)  # Identity rotation matrix
        self.T = None
        
        # Set camera parameters if provided
        if camera_settings:
            self.set_camera(camera_settings)
    
    def set_camera(self, config):
        """
        Set camera parameters from config.
        
        Args:
            config: Camera configuration dictionary
        """
        if not config:
            logging.error("No camera configuration provided")
            return
            
        try:
            # 베이스라인 값을 설정 파일에서 가져옴
            if "baseline_m" in config:
                self.baseline = config["baseline_m"]
            else:
                logging.warning("Baseline distance not specified in config, using default")
                self.baseline = 0.5  # 기본값 (미터)
                
            # 초점거리 계산 - 센서 크기로 정규화 적용
            if "focal_length_mm" in config and "sensor_width_mm" in config and "image_width_px" in config:
                # 물리적 초점거리(mm)를 픽셀 단위로 변환 (normalized)
                self.focal_length = (config["focal_length_mm"] / config["sensor_width_mm"]) * config["image_width_px"]
            elif "focal_length_px" in config:
                # 직접 픽셀 단위로 지정된 경우
                self.focal_length = config["focal_length_px"]
            else:
                logging.warning("Focal length not properly specified in config, using default")
                self.focal_length = 1000.0  # 기본값 (픽셀)
            
            # 주점(Principal Point) 설정
            if "principal_point_x" in config and "principal_point_y" in config:
                self.principal_point_x = config["principal_point_x"]
                self.principal_point_y = config["principal_point_y"]
            else:
                # 이미지 중심을 주점으로 사용
                if "image_width_px" in config and "image_height_px" in config:
                    self.principal_point_x = config["image_width_px"] / 2
                    self.principal_point_y = config["image_height_px"] / 2
            
            # 이미지 크기 설정
            if "image_width_px" in config:
                self.image_width = config["image_width_px"]
            if "image_height_px" in config:
                self.image_height = config["image_height_px"]
            
            # 카메라 매트릭스 생성
            self.camera_matrix_left = np.array([
                [self.focal_length, 0, self.principal_point_x],
                [0, self.focal_length, self.principal_point_y],
                [0, 0, 1]
            ])
            
            # 왜곡 계수 (일반적으로 보정된 이미지를 사용하므로 0으로 설정)
            self.dist_coeffs_left = np.zeros(5)
            
            # 오른쪽 카메라는 왼쪽 카메라와 동일하게 설정 (X축으로만 이동)
            self.camera_matrix_right = self.camera_matrix_left.copy()
            self.dist_coeffs_right = self.dist_coeffs_left.copy()
            
            # 회전 행렬 (두 카메라는 평행하게 설정되어 있다고 가정)
            self.R = np.eye(3)
            
            # 이동 벡터 (X축으로 베이스라인만큼 이동)
            self.T = np.array([[self.baseline], [0], [0]])
            
            # 프로젝션 매트릭스 계산
            self.proj_matrix_left, self.proj_matrix_right = calculate_projection_matrices(
                self.camera_matrix_left, self.dist_coeffs_left,
                self.camera_matrix_right, self.dist_coeffs_right,
                self.R, self.T
            )
            
            logging.info(f"Triangulation parameters set: baseline={self.baseline:.3f}m, focal_length={self.focal_length:.1f}px")
            logging.info("Projection matrices calculated successfully")
            
        except KeyError as e:
            logging.error(f"Missing required camera parameter in config: {e}")
        except Exception as e:
            logging.error(f"Error setting camera parameters: {e}")
    
    def triangulate(self, uL, vL, uR, vR, confidence_left=1.0, confidence_right=1.0) -> Tuple[np.ndarray, float]:
        """
        Triangulate a 3D point from stereo image coordinates.
        
        Args:
            uL: X-coordinate in left image (pixels)
            vL: Y-coordinate in left image (pixels)
            uR: X-coordinate in right image (pixels)
            vR: Y-coordinate in right image (pixels)
            confidence_left: Confidence of left point detection (0.0-1.0)
            confidence_right: Confidence of right point detection (0.0-1.0)
            
        Returns:
            Tuple of (3D point in world coordinates [x, y, z] in meters, confidence value)
        """
        if not self.proj_matrix_left is not None or self.proj_matrix_right is not None:
            logging.error("Triangulation service not properly configured. Set camera parameters first.")
            return np.array([0.0, 0.0, 0.0]), 0.0
        
        try:
            # Check if inputs are valid
            if None in (uL, vL, uR, vR):
                logging.warning("Invalid input for triangulation: None values detected")
                return np.array([0.0, 0.0, 0.0]), 0.0
            
            # Create 2D point arrays
            point_left = np.array([float(uL), float(vL)], dtype=np.float32)
            point_right = np.array([float(uR), float(vR)], dtype=np.float32)
            
            # Triangulate with confidence
            point_3d, confidence = triangulate_with_confidence(
                point_left, point_right,
                self.proj_matrix_left, self.proj_matrix_right,
                confidence_left, confidence_right
            )
            
            # Log the triangulation result
            logging.debug(f"Triangulated: L({uL:.1f},{vL:.1f}), R({uR:.1f},{vR:.1f}) → 3D({point_3d[0]:.3f}, {point_3d[1]:.3f}, {point_3d[2]:.3f})m, conf={confidence:.2f}")
            
            return point_3d, confidence
            
        except Exception as e:
            logging.error(f"Error in triangulation: {e}")
            return np.array([0.0, 0.0, 0.0]), 0.0
    
    def is_configured(self) -> bool:
        """
        Check if the triangulation service is properly configured.
        
        Returns:
            True if projection matrices are set, False otherwise
        """
        return (self.proj_matrix_left is not None and 
                self.proj_matrix_right is not None) 