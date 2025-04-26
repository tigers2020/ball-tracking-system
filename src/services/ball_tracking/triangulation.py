#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Triangulation module for tennis ball tracking.
This module contains functions to calculate 3D positions
from 2D positions in stereo camera views.
"""

import numpy as np
import cv2
import logging
from typing import Tuple, Optional, List
from src.utils.constants import STEREO

# Use the constant from constants.py
MIN_VALID_DISPARITY = STEREO.MIN_DISPARITY * 4.0  # Adjusted value based on previous setting

def triangulate_points(
    point1: np.ndarray,
    point2: np.ndarray,
    proj_matrix1: np.ndarray,
    proj_matrix2: np.ndarray
) -> np.ndarray:
    """
    Triangulate a 3D point from two 2D points in different camera views.
    
    Args:
        point1: 2D point in first camera view [x, y]
        point2: 2D point in second camera view [x, y]
        proj_matrix1: Projection matrix for first camera (3x4)
        proj_matrix2: Projection matrix for second camera (3x4)
        
    Returns:
        3D point in world coordinates [x, y, z]
    """
    # 디스패리티 검증 
    disparity = point1[0] - point2[0]
    if abs(disparity) < MIN_VALID_DISPARITY:
        logging.warning(f"Disparity too small: {disparity:.2f}px < {MIN_VALID_DISPARITY}px. Triangulation may be unstable.")
        
    # Triangulate using OpenCV's triangulatePoints function
    # Reshape points to the format required by triangulatePoints
    points1 = np.array([point1], dtype=np.float32).T
    points2 = np.array([point2], dtype=np.float32).T
    
    # Triangulate
    homogeneous_points = cv2.triangulatePoints(proj_matrix1, proj_matrix2, points1, points2)
    
    # Convert from homogeneous coordinates to 3D coordinates
    homogeneous_points /= homogeneous_points[3]
    point_3d = homogeneous_points[:3].T[0]
    
    return point_3d


def triangulate_ball_points(
    points1: List[np.ndarray],
    points2: List[np.ndarray],
    proj_matrix1: np.ndarray,
    proj_matrix2: np.ndarray
) -> List[np.ndarray]:
    """
    Triangulate multiple 3D points from lists of 2D points in different camera views.
    
    Args:
        points1: List of 2D points in first camera view [[x1, y1], [x2, y2], ...]
        points2: List of 2D points in second camera view [[x1, y1], [x2, y2], ...]
        proj_matrix1: Projection matrix for first camera (3x4)
        proj_matrix2: Projection matrix for second camera (3x4)
        
    Returns:
        List of 3D points in world coordinates [[x1, y1, z1], [x2, y2, z2], ...]
    """
    if len(points1) != len(points2):
        raise ValueError("Input point lists must have the same length")
    
    if not points1:
        return []
    
    # Convert lists to numpy arrays
    points1_array = np.array(points1, dtype=np.float32).T
    points2_array = np.array(points2, dtype=np.float32).T
    
    # Triangulate
    homogeneous_points = cv2.triangulatePoints(proj_matrix1, proj_matrix2, points1_array, points2_array)
    
    # Convert from homogeneous coordinates to 3D coordinates
    homogeneous_points /= homogeneous_points[3]
    points_3d = homogeneous_points[:3].T
    
    return points_3d.tolist()


def triangulate_with_confidence(
    point1: np.ndarray,
    point2: np.ndarray,
    proj_matrix1: np.ndarray,
    proj_matrix2: np.ndarray,
    confidence1: float = 1.0,
    confidence2: float = 1.0
) -> Tuple[np.ndarray, float]:
    """
    Triangulate a 3D point from two 2D points with confidence values.
    
    Args:
        point1: 2D point in first camera view [x, y]
        point2: 2D point in second camera view [x, y]
        proj_matrix1: Projection matrix for first camera (3x4)
        proj_matrix2: Projection matrix for second camera (3x4)
        confidence1: Confidence value for point1 (0.0 to 1.0)
        confidence2: Confidence value for point2 (0.0 to 1.0)
        
    Returns:
        Tuple of (3D point in world coordinates [x, y, z], combined confidence)
    """
    # 디스패리티 검증 및 신뢰도 조정
    disparity = point1[0] - point2[0]
    disparity_valid = abs(disparity) >= MIN_VALID_DISPARITY
    
    if not disparity_valid:
        logging.warning(f"Disparity too small: {disparity:.2f}px < {MIN_VALID_DISPARITY}px. Reducing confidence.")
        # 디스패리티가 작을수록 신뢰도 감소
        disparity_factor = abs(disparity) / MIN_VALID_DISPARITY
        confidence1 *= disparity_factor
        confidence2 *= disparity_factor
    
    # Triangulate the 3D point
    point_3d = triangulate_points(point1, point2, proj_matrix1, proj_matrix2)
    
    # Calculate reprojection errors
    # Project the 3D point back to 2D
    point_3d_homogeneous = np.append(point_3d, 1.0)
    projected_point1 = proj_matrix1 @ point_3d_homogeneous
    projected_point1 = projected_point1[:2] / projected_point1[2]
    
    projected_point2 = proj_matrix2 @ point_3d_homogeneous
    projected_point2 = projected_point2[:2] / projected_point2[2]
    
    # Calculate reprojection errors
    error1 = np.linalg.norm(point1 - projected_point1)
    error2 = np.linalg.norm(point2 - projected_point2)
    
    # Calculate overall confidence based on reprojection errors and input confidences
    max_acceptable_error = 5.0  # pixels
    error_factor1 = max(0.0, 1.0 - error1 / max_acceptable_error)
    error_factor2 = max(0.0, 1.0 - error2 / max_acceptable_error)
    
    # Combine confidence values
    combined_confidence = confidence1 * confidence2 * error_factor1 * error_factor2
    
    return point_3d, combined_confidence


def calculate_projection_matrices(
    camera_matrix1: np.ndarray,
    dist_coeffs1: np.ndarray,
    camera_matrix2: np.ndarray,
    dist_coeffs2: np.ndarray,
    R: np.ndarray,
    T: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate projection matrices for stereo camera setup.
    
    Args:
        camera_matrix1: Camera matrix for first camera (3x3)
        dist_coeffs1: Distortion coefficients for first camera
        camera_matrix2: Camera matrix for second camera (3x3)
        dist_coeffs2: Distortion coefficients for second camera
        R: Rotation matrix from camera 1 to camera 2 (3x3)
        T: Translation vector from camera 1 to camera 2 (3x1)
        
    Returns:
        Tuple of (projection matrix for camera 1, projection matrix for camera 2)
    """
    # Projection matrix for first camera (world coordinates aligned with camera 1)
    proj_matrix1 = np.zeros((3, 4), dtype=np.float32)
    proj_matrix1[:3, :3] = camera_matrix1
    proj_matrix1 = camera_matrix1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    
    # Projection matrix for second camera
    RT = np.zeros((3, 4), dtype=np.float32)
    RT[:3, :3] = R
    RT[:3, 3] = T.flatten()
    proj_matrix2 = camera_matrix2 @ RT
    
    return proj_matrix1, proj_matrix2


def reproject_3d_point(
    point_3d: np.ndarray,
    proj_matrix: np.ndarray
) -> np.ndarray:
    """
    Reproject a 3D point back to a 2D point on an image.
    
    Args:
        point_3d: 3D point in world coordinates [x, y, z]
        proj_matrix: Projection matrix (3x4)
        
    Returns:
        2D point in image coordinates [x, y]
    """
    # Convert to homogeneous coordinates
    point_3d_homogeneous = np.append(point_3d, 1.0)
    
    # Project to image coordinates
    point_2d_homogeneous = proj_matrix @ point_3d_homogeneous
    
    # Convert from homogeneous coordinates
    point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]
    
    return point_2d


def refine_triangulation(
    point1: np.ndarray,
    point2: np.ndarray,
    proj_matrix1: np.ndarray,
    proj_matrix2: np.ndarray,
    initial_point_3d: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Refine a triangulated 3D point using nonlinear optimization.
    
    Args:
        point1: 2D point in first camera view [x, y]
        point2: 2D point in second camera view [x, y]
        proj_matrix1: Projection matrix for first camera (3x4)
        proj_matrix2: Projection matrix for second camera (3x4)
        initial_point_3d: Initial estimate of 3D point (optional)
        
    Returns:
        Refined 3D point in world coordinates [x, y, z]
    """
    # If no initial estimate is provided, use triangulate_points
    if initial_point_3d is None:
        initial_point_3d = triangulate_points(point1, point2, proj_matrix1, proj_matrix2)
    
    # Function to minimize - sum of squared reprojection errors
    def reprojection_error(point):
        p1 = reproject_3d_point(point, proj_matrix1)
        p2 = reproject_3d_point(point, proj_matrix2)
        error1 = np.sum((p1 - point1)**2)
        error2 = np.sum((p2 - point2)**2)
        return error1 + error2
    
    # Simple gradient descent optimization
    point = initial_point_3d.copy()
    step_size = 0.01
    num_iterations = 50
    
    for _ in range(num_iterations):
        # Calculate gradient numerically
        gradient = np.zeros(3)
        for i in range(3):
            delta = np.zeros(3)
            delta[i] = 0.0001
            gradient[i] = (reprojection_error(point + delta) - reprojection_error(point - delta)) / (2 * 0.0001)
        
        # Update point
        point -= step_size * gradient
    
    return point


def transform_to_world_coordinates(
    point_3d: np.ndarray,
    R_world: np.ndarray,
    T_world: np.ndarray
) -> np.ndarray:
    """
    Transform a 3D point from camera coordinates to world coordinates.
    
    Args:
        point_3d: 3D point in camera coordinates [x, y, z]
        R_world: Rotation matrix from camera to world (3x3)
        T_world: Translation vector from camera to world (3x1)
        
    Returns:
        3D point in world coordinates [x, y, z]
    """
    # Apply rotation and translation
    point_world = R_world @ point_3d + T_world.flatten()
    
    return point_world


def transform_points_to_world_coordinates(
    points_3d: np.ndarray,
    R_world: np.ndarray,
    T_world: np.ndarray
) -> np.ndarray:
    """
    Transform multiple 3D points from camera coordinates to world coordinates.
    
    Args:
        points_3d: Array of 3D points in camera coordinates
        R_world: Rotation matrix from camera to world (3x3)
        T_world: Translation vector from camera to world (3x1)
        
    Returns:
        Array of 3D points in world coordinates
    """
    # Reshape T_world to ensure correct broadcasting
    T_world_flat = T_world.flatten()
    
    # Apply rotation and translation to each point
    points_world = np.dot(points_3d, R_world.T) + T_world_flat
    
    return points_world


def calculate_reprojection_error(
    point_3d: np.ndarray,
    point_2d: np.ndarray,
    proj_matrix: np.ndarray
) -> float:
    """
    Calculate the reprojection error for a 3D point.
    
    Args:
        point_3d: 3D point in world coordinates [x, y, z]
        point_2d: Observed 2D point in image coordinates [x, y]
        proj_matrix: Projection matrix (3x4)
        
    Returns:
        Reprojection error in pixels
    """
    # Reproject 3D point to image
    projected_point = reproject_3d_point(point_3d, proj_matrix)
    
    # Calculate Euclidean distance
    error = np.linalg.norm(projected_point - point_2d)
    
    return error


def triangulate_with_focal_length(self, left_points, right_points):
    """Triangulate 3D points from 2D point correspondences using focal length."""
    # Convert focal length from mm to pixels
    if self.focal_length_mm and self.sensor_width_mm and self.image_width:
        focal_px = self.focal_length_mm / self.sensor_width_mm * self.image_width
        logging.debug(f"Using converted focal length: {self.focal_length_mm}mm → {focal_px}px")
    else:
        focal_px = self.focal_length_mm  # Use as-is if conversion params missing
        
    # Use focal_px in triangulation calculations...
    # Implementation to be completed based on specific requirements
    return None 