#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Triangulation Service module.
This module provides functionality for 3D triangulation from stereo camera views.
"""

import numpy as np
import logging
import cv2
from typing import Optional, Tuple, Dict, Any, Union, List

from src.geometry.court_frame import get_calibration_points
from src.utils.geometry_utils import (
    DEG2RAD, create_rotation_matrix, triangulate_points as utils_triangulate_points,
    triangulate_point as utils_triangulate_point, calculate_reprojection_error,
    transform_to_world
)

class TriangulationService:
    """
    Stereo triangulation service using pinhole camera model.
    Converts stereo image coordinates to 3D world coordinates.
    """

    def __init__(self, cam_cfg: Dict[str, Any] = None):
        """
        Initialize the triangulation service with camera configuration.
        
        Args:
            cam_cfg: Dictionary containing camera parameters
        """
        self.is_calibrated = False
        self.use_pnp = False
        self.pnp_calibrated = False
        
        # PnP calibration results
        self.left_rvec = None
        self.left_tvec = None
        self.right_rvec = None
        self.right_tvec = None
        
        # Projection matrices
        self.P_left = None
        self.P_right = None
        
        # Camera intrinsics
        self.K = None
        
        # Set initial camera parameters if provided
        if cam_cfg:
            self.set_camera(cam_cfg)

    def set_camera(self, cfg: Dict[str, Any]):
        """
        Set camera parameters for triangulation.
        
        Args:
            cfg: Dictionary containing camera parameters
        """
        self.cfg = cfg
        self.scale = cfg.get("resizing_scale", 1.0)
        w0, h0 = 1920, 1080                       # Original resolution
        self.w = int(w0 * self.scale)
        self.h = int(h0 * self.scale)

        # Ensure sensor dimensions are set correctly for focal length calculation
        sensor_width_mm = cfg.get("sensor_width_mm", 36.0)  # Default to full-frame 36mm if not specified
        sensor_height_mm = cfg.get("sensor_height_mm", 24.0)  # Default to full-frame 24mm if not specified
        
        # Log sensor dimensions
        logging.info(f"Using sensor dimensions: width={sensor_width_mm}mm, height={sensor_height_mm}mm")

        # Create camera intrinsic matrix - ensure all units are consistent
        focal_length_mm = cfg.get("focal_length_mm", 50.0)
        fx = focal_length_mm / sensor_width_mm * self.w
        fy = focal_length_mm / sensor_height_mm * self.h
        
        # Ensure principal points are specified or use defaults
        cx = cfg.get("principal_point_x", self.w / 2) * self.scale
        cy = cfg.get("principal_point_y", self.h / 2) * self.scale
        
        # Store the computed intrinsic matrix
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]], dtype=np.float32)  # Use float32 consistently
        
        logging.info(f"Camera intrinsics calculated: fx={fx:.1f}, fy={fy:.1f}, focal_length={focal_length_mm}mm")

        # Precise baseline measurement is critical for depth calculation
        # Use the exact value from config or CAD measurement
        baseline_m = cfg.get("baseline_m", 0.6)  # Use the exact measured value
        
        # Left and right camera transforms
        T_left = np.array([cfg.get("camera_location_x", 0.0) - baseline_m/2,
                           cfg.get("camera_location_y", 0.0),
                           cfg.get("camera_location_z", 0.0)], dtype=np.float32)
        T_right = T_left + np.array([baseline_m, 0, 0], dtype=np.float32)

        # Rotation matrix from Euler angles
        angles = np.array([cfg.get("camera_rotation_x", 0.0),
                          cfg.get("camera_rotation_y", 0.0),
                          cfg.get("camera_rotation_z", 0.0)]) * DEG2RAD
        R = create_rotation_matrix(*angles).astype(np.float32)  # Use float32 consistently
        
        # Camera projection matrices (legacy method)
        self.R_left = R
        self.T_left = T_left.reshape(3, 1)
        self.R_right = R
        self.T_right = T_right.reshape(3, 1)
        
        # Pre-compute camera-to-world transform
        self.R_T = R.T
        
        self.is_calibrated = True
        self.use_pnp = cfg.get("use_pnp", False)
        self.pnp_calibrated = False
        
        logging.info(f"Triangulation service initialized with baseline: {baseline_m}m, scale: {self.scale}, "
                    f"use_pnp: {self.use_pnp}")

    def calibrate_from_pnp(self, left_image_points: np.ndarray, right_image_points: np.ndarray, 
                          distortion_coeffs: np.ndarray = None) -> bool:
        """
        Calibrate camera poses using PnP with court landmarks.
        
        Args:
            left_image_points: 2D points in left image (Nx2)
            right_image_points: 2D points in right image (Nx2)
            distortion_coeffs: Camera distortion coefficients (default: None, no distortion)
            
        Returns:
            True if calibration successful, False otherwise
        """
        if not self.is_calibrated:
            logging.error("Cannot calibrate from PnP: camera intrinsics not set")
            return False
            
        if left_image_points.shape[0] < 4 or right_image_points.shape[0] < 4:
            logging.error(f"Not enough points for PnP: left={left_image_points.shape[0]}, right={right_image_points.shape[0]}")
            return False
            
        if left_image_points.shape != right_image_points.shape:
            logging.error(f"Mismatched point counts: left={left_image_points.shape}, right={right_image_points.shape}")
            return False
            
        # Get calibration points in world coordinates
        world_points = get_calibration_points()
        world_points_array = np.array(world_points[:left_image_points.shape[0]], dtype=np.float32)
        
        # Default to no distortion if not provided
        if distortion_coeffs is None:
            distortion_coeffs = np.zeros(5, dtype=np.float32)
            
        try:
            # Solve PnP for left camera
            _, left_rvec, left_tvec = cv2.solvePnP(
                world_points_array, 
                left_image_points, 
                self.K, 
                distortion_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            # Solve PnP for right camera
            _, right_rvec, right_tvec = cv2.solvePnP(
                world_points_array, 
                right_image_points, 
                self.K, 
                distortion_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            # Calculate reprojection error
            left_error = self._calculate_pnp_error(world_points_array, left_image_points, self.K, distortion_coeffs, left_rvec, left_tvec)
            right_error = self._calculate_pnp_error(world_points_array, right_image_points, self.K, distortion_coeffs, right_rvec, right_tvec)
            
            logging.info(f"PnP calibration reprojection error: left={left_error:.2f}px, right={right_error:.2f}px")
            
            # Store calibration results
            self.left_rvec = left_rvec
            self.left_tvec = left_tvec
            self.right_rvec = right_rvec
            self.right_tvec = right_tvec
            
            # Convert rotation vectors to matrices
            R_left, _ = cv2.Rodrigues(left_rvec)
            R_right, _ = cv2.Rodrigues(right_rvec)
            
            # Create projection matrices for triangulation
            self.P_left = self.K @ np.hstack((R_left, left_tvec))
            self.P_right = self.K @ np.hstack((R_right, right_tvec))
            
            self.pnp_calibrated = True
            self.use_pnp = True
            
            # Save the camera-to-world transforms for later use
            self.R_left = R_left
            self.T_left = left_tvec
            self.R_right = R_right
            self.T_right = right_tvec
            
            return True
            
        except Exception as e:
            logging.error(f"PnP calibration failed: {e}")
            return False

    def _calculate_pnp_error(self, world_points: np.ndarray, image_points: np.ndarray, 
                           camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                           rvec: np.ndarray, tvec: np.ndarray) -> float:
        """
        Calculate reprojection error for PnP calibration.
        
        Args:
            world_points: 3D world points
            image_points: 2D image points
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients
            rvec: Rotation vector
            tvec: Translation vector
            
        Returns:
            Average reprojection error in pixels
        """
        # Project 3D points to image plane
        projected_points, _ = cv2.projectPoints(world_points, rvec, tvec, camera_matrix, dist_coeffs)
        projected_points = projected_points.reshape(-1, 2)
        
        # Calculate error
        error = np.sqrt(np.sum((image_points - projected_points) ** 2, axis=1))
        return np.mean(error)

    def triangulate_points(self, points_left: np.ndarray, points_right: np.ndarray) -> np.ndarray:
        """
        Triangulate multiple 3D points from stereo image points.
        
        Args:
            points_left: Points in left image (Nx2)
            points_right: Points in right image (Nx2)
            
        Returns:
            Array of 3D points (Nx3)
        """
        if not self.is_calibrated:
            logging.error("Triangulation failed: system not calibrated")
            return np.array([])
            
        if points_left.shape[0] == 0 or points_right.shape[0] == 0:
            return np.array([])
            
        if points_left.shape != points_right.shape:
            logging.error(f"Point count mismatch: left={points_left.shape[0]}, right={points_right.shape[0]}")
            return np.array([])
        
        # Use PnP-based projection matrices if available
        if self.use_pnp and self.pnp_calibrated:
            P_left = self.P_left
            P_right = self.P_right
        else:
            # Otherwise, create projection matrices from calibration
            P_left = self.K @ np.hstack((self.R_left, self.T_left))
            P_right = self.K @ np.hstack((self.R_right, self.T_right))
        
        # Use the utility function for triangulation
        return utils_triangulate_points(points_left, points_right, P_left, P_right)

    def triangulate(self, uL: float, vL: float, uR: float, vR: float) -> Optional[np.ndarray]:
        """
        Triangulate a 3D point from stereo image points.
        
        Args:
            uL: x-coordinate in left image (pixels)
            vL: y-coordinate in left image (pixels)
            uR: x-coordinate in right image (pixels)
            vR: y-coordinate in right image (pixels)
            
        Returns:
            3D world coordinates (X,Y,Z) in meters, or None if invalid
        """
        if not self.is_calibrated:
            logging.warning("Triangulation failed: system not calibrated")
            return None
            
        # Use PnP-based projection matrices if available
        if self.use_pnp and self.pnp_calibrated:
            P_left = self.P_left
            P_right = self.P_right
        else:
            # Otherwise, create projection matrices from calibration
            P_left = self.K @ np.hstack((self.R_left, self.T_left))
            P_right = self.K @ np.hstack((self.R_right, self.T_right))
        
        # Use utility function for point triangulation
        left_point = (uL, vL)
        right_point = (uR, vR)
        return utils_triangulate_point(left_point, right_point, P_left, P_right)

    def get_reprojection_error(self, world_point: np.ndarray, image_points: Tuple[float, float, float, float]) -> float:
        """
        Calculate reprojection error for a triangulated point.
        
        Args:
            world_point: 3D point in world coordinates (X,Y,Z)
            image_points: Tuple of (uL, vL, uR, vR) image coordinates
            
        Returns:
            Average reprojection error in pixels
        """
        if not self.is_calibrated:
            return float('inf')
            
        # Extract image coordinates
        uL, vL, uR, vR = image_points
        left_point = (uL, vL)
        right_point = (uR, vR)
        
        # Use utility function for error calculation
        return calculate_reprojection_error(
            world_point, 
            left_point, 
            right_point, 
            self.K, 
            self.R_left, 
            self.T_left, 
            self.R_right, 
            self.T_right
        )

    def project_point_to_image(self, world_point: np.ndarray, camera: str = 'left') -> Optional[np.ndarray]:
        """
        Project a 3D world point back onto the camera image plane.
        
        Args:
            world_point: 3D point in world coordinates [X, Y, Z]
            camera: Which camera to project to ('left' or 'right')
            
        Returns:
            2D image point [u, v] or None if projection fails
        """
        if not self.is_calibrated:
            logging.warning("Cannot project point: system not calibrated")
            return None
            
        world_point = np.array(world_point).reshape(3, 1)
        
        # Select camera parameters
        if camera.lower() == 'left':
            R = self.R_left
            T = self.T_left
        elif camera.lower() == 'right':
            R = self.R_right
            T = self.T_right
        else:
            logging.error(f"Invalid camera: {camera}, must be 'left' or 'right'")
            return None
        
        # Transform point to camera coordinates
        camera_point = R @ (world_point - T)
        
        # Ensure the point is in front of the camera
        if camera_point[2, 0] <= 0:
            return None
            
        # Project to image plane
        z = camera_point[2, 0]
        x = self.K[0, 0] * camera_point[0, 0] / z + self.K[0, 2]
        y = self.K[1, 1] * camera_point[1, 0] / z + self.K[1, 2]
        
        return np.array([x, y])

    def get_calibration_status(self) -> Dict[str, Any]:
        """
        Get the current calibration status.
        
        Returns:
            Dictionary with calibration status
        """
        status = {
            "is_calibrated": self.is_calibrated,
            "use_pnp": self.use_pnp,
            "pnp_calibrated": self.pnp_calibrated,
        }
        
        if self.is_calibrated:
            status.update({
                "image_width": self.w,
                "image_height": self.h,
                "intrinsic_matrix": self.K.tolist() if self.K is not None else None,
            })
            
        if self.pnp_calibrated:
            status.update({
                "left_camera_rotation": self.R_left.tolist() if self.R_left is not None else None,
                "left_camera_translation": self.T_left.flatten().tolist() if self.T_left is not None else None,
                "right_camera_rotation": self.R_right.tolist() if self.R_right is not None else None,
                "right_camera_translation": self.T_right.flatten().tolist() if self.T_right is not None else None,
            })
            
        return status 