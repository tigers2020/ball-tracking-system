#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stereo camera calibration utilities.
This module provides functionality for calibrating stereo camera systems.
"""

import numpy as np
import cv2
import logging
from typing import Tuple, List, Optional, Dict, Any

logger = logging.getLogger(__name__)

class StereoCalibration:
    """
    Utility class for stereo camera calibration.
    """
    
    @staticmethod
    def compute_projection_matrices(camera_matrix: np.ndarray,
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
    
    @staticmethod
    def calibrate_from_chessboard(left_images: List[np.ndarray], 
                                 right_images: List[np.ndarray],
                                 board_size: Tuple[int, int] = (9, 6),
                                 square_size: float = 0.025) -> Dict[str, Any]:
        """
        Calibrate stereo camera system from images of a chessboard pattern.
        
        Args:
            left_images: List of images from left camera
            right_images: List of images from right camera
            board_size: Number of inner corners of the calibration board (width, height)
            square_size: Size of the squares on the calibration board in meters
            
        Returns:
            Dictionary with calibration parameters
        """
        if len(left_images) != len(right_images) or len(left_images) < 5:
            logger.error(f"Not enough image pairs for calibration: {len(left_images)} left, {len(right_images)} right")
            return {}
        
        # Prepare object points (3D points in real world)
        objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size
        
        # Arrays to store object points and image points
        objpoints = []  # 3D points in real world space
        left_imgpoints = []  # 2D points in left image plane
        right_imgpoints = []  # 2D points in right image plane
        
        # Find the corners in each stereo pair
        valid_pairs = 0
        for left_img, right_img in zip(left_images, right_images):
            # Convert to grayscale
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY) if len(left_img.shape) == 3 else left_img
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY) if len(right_img.shape) == 3 else right_img
            
            # Find chessboard corners
            left_ret, left_corners = cv2.findChessboardCorners(left_gray, board_size, None)
            right_ret, right_corners = cv2.findChessboardCorners(right_gray, board_size, None)
            
            # If both images have corners, add to arrays
            if left_ret and right_ret:
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                left_corners = cv2.cornerSubPix(left_gray, left_corners, (11, 11), (-1, -1), criteria)
                right_corners = cv2.cornerSubPix(right_gray, right_corners, (11, 11), (-1, -1), criteria)
                
                objpoints.append(objp)
                left_imgpoints.append(left_corners)
                right_imgpoints.append(right_corners)
                valid_pairs += 1
        
        logger.info(f"Found {valid_pairs} valid image pairs for calibration")
        
        if valid_pairs < 5:  # Need at least 5 valid stereo pairs
            logger.error(f"Not enough valid image pairs for calibration: {valid_pairs}")
            return {}
        
        # Get image size
        img_size = (left_gray.shape[1], left_gray.shape[0])
        
        try:
            # Calibrate left and right cameras separately
            left_ret, left_camera_matrix, left_distortion, left_rvecs, left_tvecs = cv2.calibrateCamera(
                objpoints, left_imgpoints, img_size, None, None
            )
            
            right_ret, right_camera_matrix, right_distortion, right_rvecs, right_tvecs = cv2.calibrateCamera(
                objpoints, right_imgpoints, img_size, None, None
            )
            
            # Calibrate stereo system
            stereocalib_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
            
            stereocalib_ret, camera_matrix, left_distortion, right_distortion, R, T, \
            E, F = cv2.stereoCalibrate(
                objpoints, left_imgpoints, right_imgpoints,
                left_camera_matrix, left_distortion,
                right_camera_matrix, right_distortion,
                img_size, flags=cv2.CALIB_FIX_INTRINSIC,
                criteria=stereocalib_criteria
            )
            
            # Compute rectification parameters
            R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                camera_matrix, left_distortion,
                camera_matrix, right_distortion,
                img_size, R, T,
                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
            )
            
            # Return calibration results
            return {
                "success": True,
                "error": stereocalib_ret,
                "image_size": img_size,
                "camera_matrix": camera_matrix.tolist(),
                "left_distortion": left_distortion.tolist(),
                "right_distortion": right_distortion.tolist(),
                "R": R.tolist(),  # Rotation from left to right camera
                "T": T.tolist(),  # Translation from left to right camera
                "E": E.tolist(),  # Essential matrix
                "F": F.tolist(),  # Fundamental matrix
                "R1": R1.tolist(),  # Rectification transform for left camera
                "R2": R2.tolist(),  # Rectification transform for right camera
                "P1": P1.tolist(),  # Projection matrix for left camera
                "P2": P2.tolist(),  # Projection matrix for right camera
                "Q": Q.tolist(),    # Disparity-to-depth mapping matrix
            }
            
        except Exception as e:
            logger.error(f"Stereo calibration failed: {e}")
            return {"success": False, "error": str(e)}
            
    @staticmethod
    def calibrate_from_points(world_points: np.ndarray,
                             left_image_points: np.ndarray,
                             right_image_points: np.ndarray,
                             camera_matrix: np.ndarray,
                             distortion_coeffs: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calibrate camera poses using PnP with 3D-2D correspondences.
        
        Args:
            world_points: 3D points in world coordinates (Nx3)
            left_image_points: 2D points in left image (Nx2)
            right_image_points: 2D points in right image (Nx2)
            camera_matrix: Camera intrinsic matrix (3x3)
            distortion_coeffs: Camera distortion coefficients (default: None, no distortion)
            
        Returns:
            Dictionary with calibration parameters
        """
        if len(world_points) < 4 or len(left_image_points) < 4 or len(right_image_points) < 4:
            logger.error(f"Not enough points for PnP calibration: "
                       f"world={len(world_points)}, left={len(left_image_points)}, right={len(right_image_points)}")
            return {}
            
        if len(world_points) != len(left_image_points) or len(world_points) != len(right_image_points):
            logger.error(f"Point count mismatch: world={len(world_points)}, "
                       f"left={len(left_image_points)}, right={len(right_image_points)}")
            return {}
            
        # Default to no distortion if not provided
        if distortion_coeffs is None:
            distortion_coeffs = np.zeros(5, dtype=np.float32)
            
        try:
            # Solve PnP for left camera
            left_ret, left_rvec, left_tvec = cv2.solvePnP(
                world_points, 
                left_image_points, 
                camera_matrix, 
                distortion_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            # Solve PnP for right camera
            right_ret, right_rvec, right_tvec = cv2.solvePnP(
                world_points, 
                right_image_points, 
                camera_matrix, 
                distortion_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            # Convert rotation vectors to matrices
            R_left, _ = cv2.Rodrigues(left_rvec)
            R_right, _ = cv2.Rodrigues(right_rvec)
            
            # Calculate reprojection error
            def calculate_error(world_pts, image_pts, rvec, tvec):
                projected_pts, _ = cv2.projectPoints(world_pts, rvec, tvec, camera_matrix, distortion_coeffs)
                projected_pts = projected_pts.reshape(-1, 2)
                error = np.sqrt(np.sum((image_pts - projected_pts) ** 2, axis=1))
                return np.mean(error)
            
            left_error = calculate_error(world_points, left_image_points, left_rvec, left_tvec)
            right_error = calculate_error(world_points, right_image_points, right_rvec, right_tvec)
            
            logger.info(f"PnP calibration reprojection error: left={left_error:.2f}px, right={right_error:.2f}px")
            
            # Create projection matrices
            P_left = camera_matrix @ np.hstack((R_left, left_tvec))
            P_right = camera_matrix @ np.hstack((R_right, right_tvec))
            
            # Return calibration results
            return {
                "success": True,
                "left_error": float(left_error),
                "right_error": float(right_error),
                "left_rvec": left_rvec.flatten().tolist(),
                "left_tvec": left_tvec.flatten().tolist(),
                "right_rvec": right_rvec.flatten().tolist(),
                "right_tvec": right_tvec.flatten().tolist(),
                "R_left": R_left.tolist(),
                "T_left": left_tvec.flatten().tolist(),
                "R_right": R_right.tolist(),
                "T_right": right_tvec.flatten().tolist(),
                "P_left": P_left.tolist(),
                "P_right": P_right.tolist(),
            }
            
        except Exception as e:
            logger.error(f"PnP calibration failed: {e}")
            return {"success": False, "error": str(e)} 