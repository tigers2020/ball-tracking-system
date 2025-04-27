#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Core Geometry Utility module.
This module provides essential functions for 3D triangulation, coordinate transformations,
and related geometry calculations used across the application.
"""

import numpy as np
import cv2
import logging
from typing import Tuple, List, Optional, Union, Dict, Any

logger = logging.getLogger(__name__)

# Constants
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

def create_rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """
    Create a 3D rotation matrix from Euler angles (ZYX convention).
    
    Args:
        rx: Rotation around X-axis (radians)
        ry: Rotation around Y-axis (radians)
        rz: Rotation around Z-axis (radians)
        
    Returns:
        3x3 rotation matrix (numpy array)
    """
    # Create rotation matrices for each axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ], dtype=np.float32)
    
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ], dtype=np.float32)
    
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Return Z-Y-X intrinsic rotation (most common in computer vision)
    return Rz @ Ry @ Rx

def triangulate_points(points_left: np.ndarray, 
                       points_right: np.ndarray,
                       proj_matrix_left: np.ndarray,
                       proj_matrix_right: np.ndarray) -> np.ndarray:
    """
    Triangulate multiple 3D points from stereo image points.
    
    Args:
        points_left: Points in left image (N,2)
        points_right: Points in right image (N,2)
        proj_matrix_left: 3x4 projection matrix for left camera
        proj_matrix_right: 3x4 projection matrix for right camera
        
    Returns:
        Array of 3D points (N,3)
    """
    if len(points_left) != len(points_right):
        raise ValueError("Input point lists must have the same length")
    
    if len(points_left) == 0:
        return np.array([])
    
    try:
        # Reshape points for triangulation: (2, N)
        points_left_reshaped = np.array(points_left, dtype=np.float32).T
        points_right_reshaped = np.array(points_right, dtype=np.float32).T
        
        # Triangulate
        points_4d = cv2.triangulatePoints(
            proj_matrix_left,
            proj_matrix_right,
            points_left_reshaped,
            points_right_reshaped
        )
        
        # Convert from homogeneous coordinates to 3D
        points_3d = (points_4d[:3] / points_4d[3]).T
        
        return points_3d
        
    except Exception as e:
        logger.error(f"Triangulation failed: {e}")
        return np.array([])

def triangulate_point(point_left: Tuple[float, float],
                     point_right: Tuple[float, float],
                     proj_matrix_left: np.ndarray,
                     proj_matrix_right: np.ndarray) -> Optional[np.ndarray]:
    """
    Triangulate a single 3D point from stereo image points.
    
    Args:
        point_left: Point in left image (x, y)
        point_right: Point in right image (x, y)
        proj_matrix_left: 3x4 projection matrix for left camera
        proj_matrix_right: 3x4 projection matrix for right camera
        
    Returns:
        3D point [x, y, z] or None if triangulation fails
    """
    try:
        # Convert to numpy arrays and reshape for triangulation
        point_left_np = np.array([point_left], dtype=np.float32)
        point_right_np = np.array([point_right], dtype=np.float32)
        
        # Use the triangulate_points function
        points_3d = triangulate_points(
            point_left_np, 
            point_right_np, 
            proj_matrix_left, 
            proj_matrix_right
        )
        
        if len(points_3d) == 0:
            return None
            
        return points_3d[0]
        
    except Exception as e:
        logger.error(f"Point triangulation failed: {e}")
        return None

def calculate_projection_matrices(camera_matrix: np.ndarray,
                                 R_left: np.ndarray, 
                                 T_left: np.ndarray,
                                 R_right: np.ndarray, 
                                 T_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate projection matrices for stereo cameras.
    
    Args:
        camera_matrix: Camera intrinsic matrix (3x3)
        R_left: Rotation matrix for left camera (3x3)
        T_left: Translation vector for left camera (3x1)
        R_right: Rotation matrix for right camera (3x3)
        T_right: Translation vector for right camera (3x1)
        
    Returns:
        Tuple of (left_projection_matrix, right_projection_matrix)
    """
    # Ensure T vectors are correctly shaped
    T_left = T_left.reshape(3, 1)
    T_right = T_right.reshape(3, 1)
    
    # Calculate projection matrices
    P_left = camera_matrix @ np.hstack((R_left, T_left))
    P_right = camera_matrix @ np.hstack((R_right, T_right))
    
    return P_left, P_right

def calculate_reprojection_error(point_3d: np.ndarray,
                                left_point: Tuple[float, float],
                                right_point: Tuple[float, float],
                                camera_matrix: np.ndarray,
                                R_left: np.ndarray,
                                T_left: np.ndarray,
                                R_right: np.ndarray,
                                T_right: np.ndarray) -> float:
    """
    Calculate reprojection error for a triangulated point.
    
    Args:
        point_3d: 3D point in world coordinates (X,Y,Z)
        left_point: Point in left image (x, y)
        right_point: Point in right image (x, y)
        camera_matrix: Camera intrinsic matrix (3x3)
        R_left: Rotation matrix for left camera (3x3)
        T_left: Translation vector for left camera (3x1)
        R_right: Rotation matrix for right camera (3x3)
        T_right: Translation vector for right camera (3x1)
        
    Returns:
        Average reprojection error in pixels
    """
    # Extract image coordinates
    uL, vL = left_point
    uR, vR = right_point
    
    # Ensure point_3d is properly shaped
    point_3d = np.array(point_3d).reshape(3, 1)
    
    # Project back to left camera image
    cam_pt_left = R_left @ (point_3d - T_left)
    
    # Ensure positive depth for left camera
    if cam_pt_left[2, 0] <= 0:
        return float('inf')
        
    # Project to left image
    z_left = cam_pt_left[2, 0]
    x_left = camera_matrix[0, 0] * cam_pt_left[0, 0] / z_left + camera_matrix[0, 2]
    y_left = camera_matrix[1, 1] * cam_pt_left[1, 0] / z_left + camera_matrix[1, 2]
    
    # Project to right camera image
    cam_pt_right = R_right @ (point_3d - T_right)
    
    # Ensure positive depth for right camera
    if cam_pt_right[2, 0] <= 0:
        return float('inf')
        
    # Project to right image
    z_right = cam_pt_right[2, 0]
    x_right = camera_matrix[0, 0] * cam_pt_right[0, 0] / z_right + camera_matrix[0, 2]
    y_right = camera_matrix[1, 1] * cam_pt_right[1, 0] / z_right + camera_matrix[1, 2]
    
    # Calculate reprojection errors
    err_left = np.sqrt((x_left - uL)**2 + (y_left - vL)**2)
    err_right = np.sqrt((x_right - uR)**2 + (y_right - vR)**2)
    
    # Return average error
    return (err_left + err_right) / 2.0

def transform_to_world(point_3d: np.ndarray,
                      rotation_matrix: np.ndarray,
                      translation_vector: np.ndarray) -> np.ndarray:
    """
    Transform a 3D point from camera coordinates to world coordinates.
    
    Args:
        point_3d: 3D point in camera coordinates [x, y, z]
        rotation_matrix: Rotation matrix from camera to world (3x3)
        translation_vector: Translation vector from camera to world (3x1)
        
    Returns:
        3D point in world coordinates [x, y, z]
    """
    # Ensure point_3d is a vector
    point_3d = np.array(point_3d).flatten()
    
    # Apply rotation and translation
    point_world = rotation_matrix.T @ point_3d + translation_vector.flatten()
    
    return point_world

def transform_batch_to_world(points_3d: np.ndarray,
                           rotation_matrix: np.ndarray,
                           translation_vector: np.ndarray) -> np.ndarray:
    """
    Transform multiple 3D points from camera coordinates to world coordinates.
    
    Args:
        points_3d: Array of 3D points in camera coordinates (Nx3)
        rotation_matrix: Rotation matrix from camera to world (3x3)
        translation_vector: Translation vector from camera to world (3x1)
        
    Returns:
        Array of 3D points in world coordinates (Nx3)
    """
    # Reshape translation_vector for proper broadcasting
    t_flat = translation_vector.flatten()
    
    # Apply rotation and translation to each point
    points_world = np.dot(points_3d, rotation_matrix) + t_flat
    
    return points_world

def ray_plane_intersection(camera_center: np.ndarray,
                          ray_direction: np.ndarray,
                          plane_point: np.ndarray = np.array([0, 0, 0]),
                          plane_normal: np.ndarray = np.array([0, 0, 1])) -> Optional[np.ndarray]:
    """
    Calculate the intersection of a ray with a plane.
    
    Args:
        camera_center: Camera center in world coordinates [X, Y, Z]
        ray_direction: Ray direction vector [dx, dy, dz]
        plane_point: A point on the plane [X, Y, Z], default is origin
        plane_normal: Normal vector to the plane [nx, ny, nz], default is [0,0,1] for Z=0 plane
        
    Returns:
        Intersection point in world coordinates [X, Y, Z] or None if no intersection
    """
    # Normalize vectors
    ray_direction = ray_direction / np.linalg.norm(ray_direction)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    
    # Check if ray is parallel to the plane
    denom = np.dot(ray_direction, plane_normal)
    if abs(denom) < 1e-6:
        return None  # Ray is parallel to the plane
    
    # Calculate distance along the ray to intersection
    t = np.dot(plane_point - camera_center, plane_normal) / denom
    
    # Ray points away from the plane
    if t < 0:
        return None
    
    # Calculate intersection point
    intersection = camera_center + t * ray_direction
    
    return intersection

def pixel_to_camera_ray(pixel_point: Tuple[float, float],
                       camera_matrix: np.ndarray) -> np.ndarray:
    """
    Convert a pixel coordinate to a ray in camera space.
    
    Args:
        pixel_point: Point in image (u, v)
        camera_matrix: Camera intrinsic matrix (3x3)
        
    Returns:
        Ray direction vector [dx, dy, dz] in camera space
    """
    u, v = pixel_point
    
    # Get camera intrinsics
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Calculate ray direction in camera space
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = 1.0
    
    # Normalize ray direction
    ray = np.array([x, y, z])
    ray = ray / np.linalg.norm(ray)
    
    return ray 