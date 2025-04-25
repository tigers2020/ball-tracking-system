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

DEG2RAD = np.pi / 180.0

def _rot_mat(rx, ry, rz):
    """
    Create a 3D rotation matrix from Euler angles.
    
    Args:
        rx: Rotation around X-axis (radians)
        ry: Rotation around Y-axis (radians)
        rz: Rotation around Z-axis (radians)
        
    Returns:
        3x3 rotation matrix (numpy array)
    """
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]])
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx          # Z-Y-X intrinsic

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

        # Create camera intrinsic matrix
        fx = cfg["focal_length_mm"] / cfg["sensor_width"] * self.w
        fy = cfg["focal_length_mm"] / cfg["sensor_height"] * self.h
        self.K = np.array([[fx, 0, cfg["principal_point_x"] * self.scale],
                           [0, fy, cfg["principal_point_y"] * self.scale],
                           [0, 0, 1]], dtype=np.float64)

        # Left and right camera transforms
        B = cfg["baseline_m"]
        T_left = np.array([cfg["camera_location_x"] - B/2,
                           cfg["camera_location_y"],
                           cfg["camera_location_z"]])
        T_right = T_left + np.array([B, 0, 0])

        # Rotation matrix from Euler angles
        angles = np.array([cfg["camera_rotation_x"],
                          cfg["camera_rotation_y"],
                          cfg["camera_rotation_z"]]) * DEG2RAD
        R = _rot_mat(*angles)
        
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
        
        logging.info(f"Triangulation service initialized with baseline: {B}m, scale: {self.scale}, "
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
        errors = np.sqrt(np.sum((image_points - projected_points) ** 2, axis=1))
        return np.mean(errors)

    def triangulate_points(self, points_left: np.ndarray, points_right: np.ndarray) -> np.ndarray:
        """
        Triangulate 3D world coordinates from stereo image points.
        Handles batch triangulation of multiple points at once.
        
        Args:
            points_left: Nx2 array of points in left image
            points_right: Nx2 array of points in right image
            
        Returns:
            Nx3 array of 3D points in world coordinates, or empty array if fails
        """
        if not self.is_calibrated:
            logging.error("Cannot triangulate: service not calibrated")
            return np.array([])
            
        if points_left.shape != points_right.shape:
            logging.error(f"Point count mismatch: left={points_left.shape}, right={points_right.shape}")
            return np.array([])
            
        if points_left.shape[0] == 0:
            return np.array([])
            
        try:
            # Use different triangulation methods based on calibration type
            if self.use_pnp and self.pnp_calibrated:
                # Use OpenCV triangulation with projection matrices from PnP
                points_4d = cv2.triangulatePoints(
                    self.P_left, 
                    self.P_right,
                    points_left.T, 
                    points_right.T
                )
                
                # Convert from homogeneous coordinates to 3D
                points_3d = points_4d[:3] / points_4d[3]
                return points_3d.T  # Return as Nx3
                
            else:
                # Legacy method using baseline and disparity
                # For compatibility with older code
                results = []
                for (left, right) in zip(points_left, points_right):
                    point = self.triangulate(left[0], left[1], right[0], right[1])
                    if point is not None:
                        results.append(point)
                
                if not results:
                    return np.array([])
                    
                return np.array(results)
                
        except Exception as e:
            logging.error(f"Triangulation failed: {e}")
            return np.array([])

    def triangulate(self, uL: float, vL: float, uR: float, vR: float) -> Optional[np.ndarray]:
        """
        Triangulate 3D world coordinates from a single pair of stereo image points.
        
        Args:
            uL: x-coordinate in left image (pixels)
            vL: y-coordinate in left image (pixels)
            uR: x-coordinate in right image (pixels)
            vR: y-coordinate in right image (pixels)
            
        Returns:
            3D point in world coordinates (X,Y,Z) in meters, or None if invalid
        """
        # Calculate disparity
        d = float(uL - uR)
        
        # Check for valid disparity
        if abs(d) < 0.1:  # Threshold to avoid division by zero or invalid results
            logging.warning(f"Invalid disparity: {d} (uL={uL}, uR={uR})")
            return None
            
        # Camera parameters
        fx = self.K[0, 0]
        B = self.cfg["baseline_m"]

        # Triangulate depth
        Z = fx * B / d                      # depth (m)
        X_cam = (uL - self.K[0, 2]) * Z / fx
        Y_cam = (vL - self.K[1, 2]) * Z / self.K[1, 1]
        
        cam_pt = np.array([[X_cam], [Y_cam], [Z]])

        # Transform from camera to world coordinates
        world_pt = self.R_T @ cam_pt + self.T_left
        
        result = world_pt.ravel()  # (X,Y,Z)
        
        # Sanity check on result
        if not np.all(np.isfinite(result)):
            logging.warning(f"Non-finite values in triangulated point: {result}")
            return None
            
        logging.debug(f"Triangulated point: {result} from image points L({uL},{vL}), R({uR},{vR})")
        return result

    def get_reprojection_error(self, world_point: np.ndarray, image_points: Tuple[float, float, float, float]) -> float:
        """
        Calculate reprojection error for a triangulated point.
        
        Args:
            world_point: 3D point in world coordinates (X,Y,Z)
            image_points: Tuple of (uL, vL, uR, vR) original image points
            
        Returns:
            Reprojection error in pixels (average of left and right)
        """
        uL, vL, uR, vR = image_points
        
        # Project world point back to image coordinates
        cam_pt = self.R_left @ (world_point.reshape(3, 1) - self.T_left)
        
        # Ensure positive depth
        if cam_pt[2, 0] <= 0:
            return float('inf')
            
        # Project to left and right images
        z = cam_pt[2, 0]
        x_left = self.K[0, 0] * cam_pt[0, 0] / z + self.K[0, 2]
        y_left = self.K[1, 1] * cam_pt[1, 0] / z + self.K[1, 2]
        
        # Right camera is shifted by baseline
        cam_pt_right = self.R_right @ (world_point.reshape(3, 1) - self.T_right)
        
        # Ensure positive depth for right camera
        if cam_pt_right[2, 0] <= 0:
            return float('inf')
            
        z_right = cam_pt_right[2, 0]
        x_right = self.K[0, 0] * cam_pt_right[0, 0] / z_right + self.K[0, 2]
        y_right = self.K[1, 1] * cam_pt_right[1, 0] / z_right + self.K[1, 2]
        
        # Calculate reprojection errors
        err_left = np.sqrt((x_left - uL)**2 + (y_left - vL)**2)
        err_right = np.sqrt((x_right - uR)**2 + (y_right - vR)**2)
        
        return (err_left + err_right) / 2.0 

    def project_point_to_image(self, world_point: np.ndarray, camera: str = 'left') -> Optional[np.ndarray]:
        """
        Project a 3D world point to 2D image coordinates.
        
        Args:
            world_point: 3D point in world coordinates (X,Y,Z)
            camera: Which camera to project to ('left' or 'right')
            
        Returns:
            2D point in image coordinates (u,v), or None if invalid
        """
        if not self.is_calibrated:
            logging.error("Cannot project: service not calibrated")
            return None
            
        world_point = np.array(world_point, dtype=np.float64).reshape(3, 1)
        
        try:
            if camera.lower() == 'left':
                R, T = self.R_left, self.T_left
            elif camera.lower() == 'right':
                R, T = self.R_right, self.T_right
            else:
                logging.error(f"Invalid camera: {camera}")
                return None
                
            # Transform from world to camera coordinates
            cam_pt = R @ (world_point - T)
            
            # Check if point is in front of camera
            if cam_pt[2, 0] <= 0:
                return None
                
            # Project to image coordinates
            u = self.K[0, 0] * cam_pt[0, 0] / cam_pt[2, 0] + self.K[0, 2]
            v = self.K[1, 1] * cam_pt[1, 0] / cam_pt[2, 0] + self.K[1, 2]
            
            return np.array([u, v])
            
        except Exception as e:
            logging.error(f"Projection error: {e}")
            return None

    def get_calibration_status(self) -> Dict[str, Any]:
        """
        Get current calibration status.
        
        Returns:
            Dictionary with calibration status information
        """
        return {
            "is_calibrated": self.is_calibrated,
            "use_pnp": self.use_pnp,
            "pnp_calibrated": self.pnp_calibrated,
            "has_intrinsics": self.K is not None
        } 