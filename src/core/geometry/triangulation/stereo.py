#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stereo triangulation implementation.
Provides functionality for triangulating 3D points from stereo camera images.
"""

import numpy as np
import cv2
import logging
from typing import Optional, Tuple, Dict, Any, List, Union

from src.core.geometry.triangulation.base import AbstractTriangulator
from src.core.utils.geometry_utils import (
    triangulate_points as utils_triangulate_points,
    triangulate_point as utils_triangulate_point,
    calculate_reprojection_error
)

logger = logging.getLogger(__name__)

class StereoTriangulator(AbstractTriangulator):
    """
    Stereo triangulation implementation using OpenCV.
    """
    
    def __init__(self, camera_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the stereo triangulator.
        
        Args:
            camera_params: Dictionary containing camera parameters
        """
        super().__init__(camera_params)
        
        # Additional stereo-specific parameters
        self.use_pnp = False
        self.pnp_calibrated = False
        
        # Rectification parameters
        self.R1 = None  # Rectification transform for left camera
        self.R2 = None  # Rectification transform for right camera
        self.P1 = None  # Projection matrix for left camera after rectification
        self.P2 = None  # Projection matrix for right camera after rectification
        self.Q = None   # Disparity-to-depth mapping matrix
        
        # Rectification maps
        self.left_map1 = None
        self.left_map2 = None
        self.right_map1 = None
        self.right_map2 = None
    
    def set_camera_parameters(self, camera_params: Dict[str, Any]) -> None:
        """
        Set camera parameters for triangulation.
        
        Args:
            camera_params: Dictionary containing camera parameters
        """
        self.camera_params = camera_params
        
        # Extract camera intrinsic matrix
        if "camera_matrix" in camera_params:
            self.camera_matrix = np.array(camera_params["camera_matrix"], dtype=np.float32)
        elif all(k in camera_params for k in ["fx", "fy", "cx", "cy"]):
            # Create camera matrix from individual parameters
            fx = camera_params["fx"]
            fy = camera_params["fy"]
            cx = camera_params["cx"]
            cy = camera_params["cy"]
            self.camera_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
        else:
            logger.error("Camera matrix not provided in camera_params")
            return
        
        # Extract rotation and translation
        if "R_left" in camera_params and "T_left" in camera_params:
            self.R_left = np.array(camera_params["R_left"], dtype=np.float32)
            self.T_left = np.array(camera_params["T_left"], dtype=np.float32).reshape(3, 1)
        else:
            # Default to identity rotation and zero translation for left camera
            self.R_left = np.eye(3, dtype=np.float32)
            self.T_left = np.zeros((3, 1), dtype=np.float32)
        
        if "R_right" in camera_params and "T_right" in camera_params:
            self.R_right = np.array(camera_params["R_right"], dtype=np.float32)
            self.T_right = np.array(camera_params["T_right"], dtype=np.float32).reshape(3, 1)
        elif "baseline_m" in camera_params:
            # Default right camera is translated along X-axis by baseline
            baseline = camera_params["baseline_m"]
            self.R_right = self.R_left.copy()
            self.T_right = self.T_left.copy()
            self.T_right[0, 0] += baseline
        else:
            logger.warning("Right camera parameters not provided, using default values")
            self.R_right = self.R_left.copy()
            self.T_right = np.array([0.6, 0, 0], dtype=np.float32).reshape(3, 1)
        
        # Create projection matrices
        self.P_left = self.camera_matrix @ np.hstack((self.R_left, self.T_left))
        self.P_right = self.camera_matrix @ np.hstack((self.R_right, self.T_right))
        
        # Set calibration flag
        self.is_calibrated = True
        
        # Check if PnP calibration is available
        self.use_pnp = camera_params.get("use_pnp", False)
        self.pnp_calibrated = self.use_pnp and "P_left" in camera_params and "P_right" in camera_params
        
        if self.pnp_calibrated:
            self.P_left = np.array(camera_params["P_left"], dtype=np.float32)
            self.P_right = np.array(camera_params["P_right"], dtype=np.float32)
        
        # Set up for rectification if parameters are available
        if all(k in camera_params for k in ["R1", "R2", "P1", "P2", "Q"]):
            self.R1 = np.array(camera_params["R1"], dtype=np.float32)
            self.R2 = np.array(camera_params["R2"], dtype=np.float32)
            self.P1 = np.array(camera_params["P1"], dtype=np.float32)
            self.P2 = np.array(camera_params["P2"], dtype=np.float32)
            self.Q = np.array(camera_params["Q"], dtype=np.float32)
            
            # Get image size for rectification maps
            if "image_width" in camera_params and "image_height" in camera_params:
                w = camera_params["image_width"]
                h = camera_params["image_height"]
                
                # Compute rectification maps
                self._compute_rectification_maps((w, h))
    
    def _compute_rectification_maps(self, image_size: Tuple[int, int]) -> None:
        """
        Compute rectification maps for stereo rectification.
        
        Args:
            image_size: Image size (width, height)
        """
        if (self.R1 is None or self.R2 is None or 
            self.P1 is None or self.P2 is None or
            self.camera_matrix is None):
            logger.warning("Rectification parameters not available")
            return
            
        # Distortion coefficients (default to no distortion)
        left_dist = np.zeros(5, dtype=np.float32)
        right_dist = np.zeros(5, dtype=np.float32)
        
        if "left_distortion" in self.camera_params:
            left_dist = np.array(self.camera_params["left_distortion"], dtype=np.float32)
        if "right_distortion" in self.camera_params:
            right_dist = np.array(self.camera_params["right_distortion"], dtype=np.float32)
        
        # Compute rectification maps
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, left_dist, self.R1, self.P1,
            image_size, cv2.CV_32FC1
        )
        
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, right_dist, self.R2, self.P2,
            image_size, cv2.CV_32FC1
        )
    
    def rectify_images(self, left_image: np.ndarray, right_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectify stereo images.
        
        Args:
            left_image: Image from left camera
            right_image: Image from right camera
            
        Returns:
            Tuple of rectified (left_image, right_image)
        """
        if (self.left_map1 is None or self.left_map2 is None or 
            self.right_map1 is None or self.right_map2 is None):
            logger.warning("Rectification maps not computed. Images will not be rectified.")
            return left_image, right_image
            
        # Rectify images
        left_rectified = cv2.remap(left_image, self.left_map1, self.left_map2, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_image, self.right_map1, self.right_map2, cv2.INTER_LINEAR)
        
        return left_rectified, right_rectified
    
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
            logger.error("Triangulation failed: system not calibrated")
            return np.array([])
            
        if points_left.shape[0] == 0 or points_right.shape[0] == 0:
            return np.array([])
            
        if points_left.shape != points_right.shape:
            logger.error(f"Point count mismatch: left={points_left.shape[0]}, right={points_right.shape[0]}")
            return np.array([])
        
        # Use the utility function for triangulation
        return utils_triangulate_points(points_left, points_right, self.P_left, self.P_right)
    
    def triangulate_point(self, point_left: Tuple[float, float], point_right: Tuple[float, float]) -> Optional[np.ndarray]:
        """
        Triangulate a single 3D point from stereo image points.
        
        Args:
            point_left: Point in left image (x, y)
            point_right: Point in right image (x, y)
            
        Returns:
            3D point [X, Y, Z] or None if triangulation fails
        """
        if not self.is_calibrated:
            logger.warning("Triangulation failed: system not calibrated")
            return None
        
        # Use utility function for point triangulation
        return utils_triangulate_point(point_left, point_right, self.P_left, self.P_right)
    
    def calculate_reprojection_error(self, point_3d: np.ndarray, 
                                    left_point: Tuple[float, float], 
                                    right_point: Tuple[float, float]) -> float:
        """
        Calculate reprojection error for a triangulated point.
        
        Args:
            point_3d: 3D point in world coordinates (X,Y,Z)
            left_point: Point in left image (x, y)
            right_point: Point in right image (x, y)
            
        Returns:
            Average reprojection error in pixels
        """
        if not self.is_calibrated:
            return float('inf')
        
        # Use utility function for error calculation
        return calculate_reprojection_error(
            point_3d, 
            left_point, 
            right_point, 
            self.camera_matrix, 
            self.R_left, 
            self.T_left, 
            self.R_right, 
            self.T_right
        )
    
    def compute_disparity_map(self, left_image: np.ndarray, 
                             right_image: np.ndarray,
                             min_disparity: int = 0,
                             num_disparities: int = 64,
                             block_size: int = 11) -> np.ndarray:
        """
        Compute disparity map from stereo images.
        
        Args:
            left_image: Image from left camera
            right_image: Image from right camera
            min_disparity: Minimum disparity value
            num_disparities: Number of disparity values, must be divisible by 16
            block_size: Size of the block for matching, odd number
            
        Returns:
            Disparity map
        """
        # Rectify images if rectification maps are available
        if self.left_map1 is not None and self.right_map1 is not None:
            left_gray, right_gray = self.rectify_images(left_image, right_image)
        else:
            # Convert to grayscale if needed
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY) if len(left_image.shape) == 3 else left_image
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY) if len(right_image.shape) == 3 else right_image
        
        # Create StereoBM object
        stereo = cv2.StereoBM_create(
            numDisparities=num_disparities,
            blockSize=block_size
        )
        stereo.setMinDisparity(min_disparity)
        
        # Compute disparity map
        disparity = stereo.compute(left_gray, right_gray)
        
        return disparity
    
    def reproject_disparity_to_3d(self, disparity_map: np.ndarray) -> np.ndarray:
        """
        Reproject a disparity map to 3D points.
        
        Args:
            disparity_map: Disparity map computed with compute_disparity_map
            
        Returns:
            3D points map (height x width x 3)
        """
        if self.Q is None:
            logger.error("Q matrix not computed. Cannot reproject disparity.")
            return np.array([])
        
        # Reproject to 3D
        points_3d = cv2.reprojectImageTo3D(disparity_map, self.Q)
        
        return points_3d 