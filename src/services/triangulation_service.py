#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Triangulation Service module.
This module provides functionality for 3D triangulation from stereo camera views.
"""

import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any, Union

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

    def __init__(self, cam_cfg: Dict[str, Any]):
        """
        Initialize the triangulation service with camera configuration.
        
        Args:
            cam_cfg: Dictionary containing camera parameters
        """
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
        
        # Camera projection matrices
        self.P_L = (R, T_left.reshape(3, 1))
        self.P_R = (R, T_right.reshape(3, 1))
        
        # Pre-compute camera-to-world transform
        self.R_T = R.T
        
        logging.info(f"Triangulation service initialized with baseline: {B}m, scale: {self.scale}")

    def triangulate(self, uL: float, vL: float, uR: float, vR: float) -> Optional[np.ndarray]:
        """
        Triangulate 3D world coordinates from stereo image points.
        
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
        R, T = self.P_L
        world_pt = self.R_T @ cam_pt + T
        
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
        R, T = self.P_L
        cam_pt = R @ (world_point.reshape(3, 1) - T)
        
        # Ensure positive depth
        if cam_pt[2, 0] <= 0:
            return float('inf')
            
        # Project to left and right images
        z = cam_pt[2, 0]
        x_left = self.K[0, 0] * cam_pt[0, 0] / z + self.K[0, 2]
        y_left = self.K[1, 1] * cam_pt[1, 0] / z + self.K[1, 2]
        
        # Right camera is shifted by baseline
        R, T = self.P_R
        cam_pt_right = R @ (world_point.reshape(3, 1) - T)
        
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