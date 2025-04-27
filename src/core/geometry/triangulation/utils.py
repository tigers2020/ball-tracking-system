#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Triangulation utility functions.

This module provides common utility functions for triangulation operations,
including DLT (Direct Linear Transform), SVD-based methods, and projection matrix calculations.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np
import cv2

logger = logging.getLogger(__name__)

def create_projection_matrix(camera_matrix: np.ndarray, 
                            rotation_vector: np.ndarray, 
                            translation_vector: np.ndarray) -> np.ndarray:
    """
    Create a projection matrix from camera intrinsics and extrinsics.
    
    Args:
        camera_matrix: 3x3 camera intrinsic matrix
        rotation_vector: Rodrigues rotation vector
        translation_vector: Translation vector
        
    Returns:
        3x4 projection matrix P = K[R|t]
    """
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Create RT matrix [R|t]
    rt_matrix = np.column_stack((rotation_matrix, translation_vector))
    
    # Create projection matrix P = K[R|t]
    projection_matrix = camera_matrix.dot(rt_matrix)
    
    return projection_matrix

def dlt_triangulation(points_2d: List[np.ndarray], 
                     projection_matrices: List[np.ndarray]) -> np.ndarray:
    """
    Triangulate a 3D point using Direct Linear Transform (DLT).
    
    Args:
        points_2d: List of 2D points [x, y] from multiple views
        projection_matrices: List of 3x4 projection matrices for each view
        
    Returns:
        3D point [X, Y, Z, W] in homogeneous coordinates
    """
    # Check for valid inputs
    if len(points_2d) != len(projection_matrices):
        raise ValueError("Number of points must match number of projection matrices")
    
    if len(points_2d) < 2:
        raise ValueError("At least two points required for triangulation")
    
    # Create the design matrix A
    A = np.zeros((2 * len(points_2d), 4))
    
    for i, (point, P) in enumerate(zip(points_2d, projection_matrices)):
        x, y = point
        A[2*i] = x * P[2] - P[0]
        A[2*i + 1] = y * P[2] - P[1]
    
    # Solve for the least squares solution using SVD
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    
    # Return homogeneous 3D point
    return X

def midpoint_triangulation(points_2d: List[np.ndarray], 
                          camera_centers: List[np.ndarray], 
                          camera_rays: List[np.ndarray]) -> np.ndarray:
    """
    Triangulate a 3D point using the midpoint method.
    
    This method finds the 3D point that is closest (in a least-squares sense)
    to all the rays from camera centers through the 2D image points.
    
    Args:
        points_2d: List of 2D points from multiple views
        camera_centers: List of camera center coordinates
        camera_rays: List of ray directions from camera centers through 2D points
        
    Returns:
        3D point [X, Y, Z]
    """
    n = len(points_2d)
    
    # Check for valid inputs
    if n != len(camera_centers) or n != len(camera_rays):
        raise ValueError("Number of points, camera centers, and rays must match")
    
    if n < 2:
        raise ValueError("At least two points required for triangulation")
    
    # Normalize ray directions
    directions = [ray / np.linalg.norm(ray) for ray in camera_rays]
    
    # Build the least squares system
    A = np.zeros((3 * n, 3))
    b = np.zeros(3 * n)
    
    for i, (center, direction) in enumerate(zip(camera_centers, directions)):
        # For each ray, we add the constraint that the point must be
        # closest to the line defined by center + t*direction
        eye_matrix = np.eye(3)
        A[3*i:3*i+3] = eye_matrix - np.outer(direction, direction)
        b[3*i:3*i+3] = (eye_matrix - np.outer(direction, direction)).dot(center)
    
    # Solve for the least squares solution
    X, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    return X

def calculate_reprojection_error(point_3d: np.ndarray, 
                               points_2d: List[np.ndarray], 
                               projection_matrices: List[np.ndarray]) -> float:
    """
    Calculate the mean reprojection error for a triangulated 3D point.
    
    Args:
        point_3d: 3D point [X, Y, Z] or [X, Y, Z, W] in homogeneous coordinates
        points_2d: List of original 2D points used for triangulation
        projection_matrices: List of projection matrices for each camera
        
    Returns:
        Mean reprojection error in pixels
    """
    # Ensure point_3d is in homogeneous coordinates
    if point_3d.shape[0] == 3:
        point_3d_h = np.append(point_3d, 1.0)
    else:
        point_3d_h = point_3d
    
    # Normalize homogeneous coordinates
    point_3d_h = point_3d_h / point_3d_h[-1]
    
    total_error = 0
    
    for i, (point_2d, proj_matrix) in enumerate(zip(points_2d, projection_matrices)):
        # Project 3D point back to 2D
        projected_point = proj_matrix.dot(point_3d_h)
        projected_point = projected_point / projected_point[2]  # Normalize
        projected_point = projected_point[:2]  # Keep only x, y
        
        # Calculate Euclidean distance between original and reprojected point
        error = np.linalg.norm(point_2d - projected_point)
        total_error += error
    
    # Return mean error
    return total_error / len(points_2d)

def linear_eigen_triangulation(points_2d: List[np.ndarray], 
                              projection_matrices: List[np.ndarray]) -> np.ndarray:
    """
    Triangulate a 3D point using the linear eigenvalue method.
    
    Args:
        points_2d: List of 2D points from multiple views
        projection_matrices: List of 3x4 projection matrices for each view
        
    Returns:
        3D point [X, Y, Z] in world coordinates
    """
    # Check for valid inputs
    if len(points_2d) != len(projection_matrices):
        raise ValueError("Number of points must match number of projection matrices")
    
    if len(points_2d) < 2:
        raise ValueError("At least two points required for triangulation")
    
    # Build the measurement matrix M
    M = np.zeros((4, 4))
    
    for point, P in zip(points_2d, projection_matrices):
        x, y = point
        p1, p2, p3 = P
        
        A = np.array([
            x * p3 - p1,
            y * p3 - p2
        ])
        
        # Accumulate outer products to build M
        for i in range(2):
            for j in range(4):
                for k in range(4):
                    M[j, k] += A[i, j] * A[i, k]
    
    # Find the eigenvector corresponding to the smallest eigenvalue of M
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    X = eigenvectors[:, 0]  # Column with smallest eigenvalue
    
    # Normalize and return 3D point
    X = X / X[3]
    return X[:3]

def compute_fundamental_matrix(K1: np.ndarray, 
                              K2: np.ndarray, 
                              R1: np.ndarray, 
                              R2: np.ndarray, 
                              t1: np.ndarray, 
                              t2: np.ndarray) -> np.ndarray:
    """
    Compute the fundamental matrix from camera parameters.
    
    Args:
        K1, K2: Intrinsic camera matrices for cameras 1 and 2
        R1, R2: Rotation matrices for cameras 1 and 2
        t1, t2: Translation vectors for cameras 1 and 2
        
    Returns:
        3x3 fundamental matrix
    """
    # Calculate the essential matrix first
    R_rel = R2.dot(R1.T)
    t_rel = t2 - R_rel.dot(t1)
    
    # Convert translation vector to skew-symmetric matrix
    t_cross = np.array([
        [0, -t_rel[2], t_rel[1]],
        [t_rel[2], 0, -t_rel[0]],
        [-t_rel[1], t_rel[0], 0]
    ])
    
    # Calculate essential matrix: E = t_cross * R
    E = t_cross.dot(R_rel)
    
    # Convert essential matrix to fundamental matrix: F = K2^-T * E * K1^-1
    F = np.linalg.inv(K2).T.dot(E).dot(np.linalg.inv(K1))
    
    return F 