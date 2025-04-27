#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Triangulation module.
This module contains the Triangulation class, which is used to calculate
3D coordinates from stereo image points.

[DEPRECATED] This is legacy code that will be removed in future versions.
Please use the triangulation_point/triangulate_points functions from geometry_utils instead.
"""

import numpy as np
import cv2
import logging
from typing import Tuple, Optional

# Add legacy warning logger
logger = logging.getLogger(__name__)

class Triangulation:
    """
    Triangulation class for stereo vision.
    
    This class calculates 3D coordinates from stereo image points using
    the Direct Linear Transform (DLT) method.
    
    [DEPRECATED] This is legacy code that will be removed in future versions.
    """
    
    def __init__(self):
        """Initialize the triangulation object."""
        # Log deprecation warning
        logger.warning("Triangulation class is deprecated and will be removed in future versions")
        
        # Camera matrices
        self.projection_matrix_left = None
        self.projection_matrix_right = None
        
        # Camera intrinsic parameters
        self.camera_matrix_left = None
        self.camera_matrix_right = None
        
        # Camera distortion parameters
        self.distortion_left = None
        self.distortion_right = None
        
        # Flag to indicate if the triangulation is calibrated
        self.is_calibrated = False
        
    def set_projection_matrices(self, proj_left: np.ndarray, proj_right: np.ndarray):
        """
        Set the projection matrices for the left and right cameras.
        
        Args:
            proj_left: 3x4 projection matrix for the left camera
            proj_right: 3x4 projection matrix for the right camera
        """
        self.projection_matrix_left = proj_left
        self.projection_matrix_right = proj_right
        self.is_calibrated = True
        
    def set_camera_parameters(self,
                             camera_matrix_left: np.ndarray,
                             camera_matrix_right: np.ndarray,
                             distortion_left: np.ndarray,
                             distortion_right: np.ndarray,
                             rotation: np.ndarray,
                             translation: np.ndarray):
        """
        Set the camera parameters for the left and right cameras.
        
        Args:
            camera_matrix_left: 3x3 intrinsic matrix for the left camera
            camera_matrix_right: 3x3 intrinsic matrix for the right camera
            distortion_left: Distortion coefficients for the left camera
            distortion_right: Distortion coefficients for the right camera
            rotation: 3x3 rotation matrix from right to left camera
            translation: 3x1 translation vector from right to left camera
        """
        self.camera_matrix_left = camera_matrix_left
        self.camera_matrix_right = camera_matrix_right
        self.distortion_left = distortion_left
        self.distortion_right = distortion_right
        
        # Compute projection matrices
        # Left camera is assumed to be at the origin
        rt_left = np.hstack((np.eye(3), np.zeros((3, 1))))
        self.projection_matrix_left = camera_matrix_left @ rt_left
        
        # Right camera is related to the left by R and t
        rt_right = np.hstack((rotation, translation.reshape(3, 1)))
        self.projection_matrix_right = camera_matrix_right @ rt_right
        
        self.is_calibrated = True
        
    def triangulate(self, 
                   point_left: Tuple[float, float], 
                   point_right: Tuple[float, float]) -> Optional[Tuple[float, float, float]]:
        """
        Triangulate a 3D point from corresponding points in the left and right images.
        
        Args:
            point_left: Point in the left image (x, y)
            point_right: Point in the right image (x, y)
            
        Returns:
            Tuple[float, float, float]: 3D coordinates (x, y, z) or None if triangulation fails
        """
        if not self.is_calibrated:
            print("Triangulation is not calibrated.")
            return None
        
        # Convert to numpy arrays
        point_left = np.array(point_left).reshape(1, 2)
        point_right = np.array(point_right).reshape(1, 2)
        
        # Undistort points if camera matrices and distortion coefficients are available
        if self.camera_matrix_left is not None and self.distortion_left is not None:
            point_left = cv2.undistortPoints(
                point_left, self.camera_matrix_left, self.distortion_left, None, self.camera_matrix_left
            )
            point_right = cv2.undistortPoints(
                point_right, self.camera_matrix_right, self.distortion_right, None, self.camera_matrix_right
            )
        
        # Triangulate
        point_4d = cv2.triangulatePoints(
            self.projection_matrix_left,
            self.projection_matrix_right,
            point_left.reshape(2, 1),
            point_right.reshape(2, 1)
        )
        
        # Convert from homogeneous coordinates to 3D
        point_3d = (point_4d / point_4d[3])[:3].flatten()
        
        return tuple(point_3d)
    
    def triangulate_batch(self, 
                         points_left: np.ndarray, 
                         points_right: np.ndarray) -> Optional[np.ndarray]:
        """
        Triangulate multiple 3D points from corresponding points in the left and right images.
        
        Args:
            points_left: Points in the left image, shape (N, 2)
            points_right: Points in the right image, shape (N, 2)
            
        Returns:
            np.ndarray: 3D coordinates, shape (N, 3) or None if triangulation fails
        """
        if not self.is_calibrated:
            print("Triangulation is not calibrated.")
            return None
        
        if len(points_left) != len(points_right):
            print("Number of points in left and right images must be the same.")
            return None
        
        if len(points_left) == 0:
            return np.empty((0, 3))
        
        # Ensure points are in the right format
        points_left = np.array(points_left, dtype=np.float32)
        points_right = np.array(points_right, dtype=np.float32)
        
        # Reshape to (2, N) as required by cv2.triangulatePoints
        points_left_reshaped = points_left.T
        points_right_reshaped = points_right.T
        
        # Undistort points if camera matrices and distortion coefficients are available
        if self.camera_matrix_left is not None and self.distortion_left is not None:
            points_left = cv2.undistortPoints(
                points_left.reshape(-1, 1, 2), 
                self.camera_matrix_left, 
                self.distortion_left, 
                None, 
                self.camera_matrix_left
            ).reshape(-1, 2)
            
            points_right = cv2.undistortPoints(
                points_right.reshape(-1, 1, 2), 
                self.camera_matrix_right, 
                self.distortion_right, 
                None, 
                self.camera_matrix_right
            ).reshape(-1, 2)
            
            # Reshape to (2, N)
            points_left_reshaped = points_left.T
            points_right_reshaped = points_right.T
        
        # Triangulate
        points_4d = cv2.triangulatePoints(
            self.projection_matrix_left,
            self.projection_matrix_right,
            points_left_reshaped,
            points_right_reshaped
        )
        
        # Convert from homogeneous coordinates to 3D
        points_3d = (points_4d / points_4d[3])[:3].T
        
        return points_3d
    
    def reproject(self, 
                 point_3d: Tuple[float, float, float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Reproject a 3D point back to the left and right images.
        
        Args:
            point_3d: 3D point (x, y, z)
            
        Returns:
            Tuple[Tuple[float, float], Tuple[float, float]]: Reprojected points in the left and right images
        """
        if not self.is_calibrated:
            print("Triangulation is not calibrated.")
            return (0, 0), (0, 0)
        
        # Convert to numpy array and homogeneous coordinates
        point_homo = np.ones(4)
        point_homo[:3] = point_3d
        
        # Project to left and right image planes
        point_left_homo = self.projection_matrix_left @ point_homo
        point_right_homo = self.projection_matrix_right @ point_homo
        
        # Convert to image coordinates
        point_left = (point_left_homo[:2] / point_left_homo[2]).flatten()
        point_right = (point_right_homo[:2] / point_right_homo[2]).flatten()
        
        return tuple(point_left), tuple(point_right)
    
    def get_reprojection_error(self, 
                              point_3d: Tuple[float, float, float],
                              point_left: Tuple[float, float],
                              point_right: Tuple[float, float]) -> Tuple[float, float]:
        """
        Calculate the reprojection error for a 3D point.
        
        Args:
            point_3d: 3D point (x, y, z)
            point_left: Point in the left image (x, y)
            point_right: Point in the right image (x, y)
            
        Returns:
            Tuple[float, float]: Reprojection errors in the left and right images
        """
        # Reproject the 3D point
        reproj_left, reproj_right = self.reproject(point_3d)
        
        # Calculate the reprojection error
        error_left = np.linalg.norm(np.array(point_left) - np.array(reproj_left))
        error_right = np.linalg.norm(np.array(point_right) - np.array(reproj_right))
        
        return error_left, error_right 