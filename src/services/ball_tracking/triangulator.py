#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Triangulation for tennis ball tracking using stereo vision.
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict, Any


class StereoTriangulator:
    """Triangulate 3D points from stereo camera images."""
    
    def __init__(
        self,
        left_camera_matrix: Optional[np.ndarray] = None,
        right_camera_matrix: Optional[np.ndarray] = None,
        left_distortion: Optional[np.ndarray] = None,
        right_distortion: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        T: Optional[np.ndarray] = None
    ):
        """
        Initialize the stereo triangulator.
        
        Args:
            left_camera_matrix: Intrinsic camera matrix for the left camera
            right_camera_matrix: Intrinsic camera matrix for the right camera
            left_distortion: Distortion coefficients for the left camera
            right_distortion: Distortion coefficients for the right camera
            R: Rotation matrix between left and right cameras
            T: Translation vector between left and right cameras
        """
        self.left_camera_matrix = left_camera_matrix
        self.right_camera_matrix = right_camera_matrix
        self.left_distortion = left_distortion
        self.right_distortion = right_distortion
        self.R = R
        self.T = T
        
        # Stereo rectification parameters (computed when calibrated)
        self.R1 = None  # Rectification transform for left camera
        self.R2 = None  # Rectification transform for right camera
        self.P1 = None  # Projection matrix for left camera
        self.P2 = None  # Projection matrix for right camera
        self.Q = None   # Disparity-to-depth mapping matrix
        
        # Rectification maps
        self.left_map1 = None
        self.left_map2 = None
        self.right_map1 = None
        self.right_map2 = None
        
        # Compute rectification parameters if all required matrices are provided
        if (left_camera_matrix is not None and right_camera_matrix is not None and 
            R is not None and T is not None):
            self._compute_rectification()
    
    def set_camera_parameters(
        self,
        left_camera_matrix: np.ndarray,
        right_camera_matrix: np.ndarray,
        left_distortion: np.ndarray,
        right_distortion: np.ndarray,
        R: np.ndarray,
        T: np.ndarray
    ):
        """
        Set stereo camera parameters.
        
        Args:
            left_camera_matrix: Intrinsic camera matrix for the left camera
            right_camera_matrix: Intrinsic camera matrix for the right camera
            left_distortion: Distortion coefficients for the left camera
            right_distortion: Distortion coefficients for the right camera
            R: Rotation matrix between left and right cameras
            T: Translation vector between left and right cameras
        """
        self.left_camera_matrix = left_camera_matrix
        self.right_camera_matrix = right_camera_matrix
        self.left_distortion = left_distortion
        self.right_distortion = right_distortion
        self.R = R
        self.T = T
        
        # Compute rectification with new parameters
        self._compute_rectification()
    
    def _compute_rectification(self, image_size: Tuple[int, int] = (640, 480)):
        """
        Compute stereo rectification parameters.
        
        Args:
            image_size: Size of the images (width, height)
        """
        if (self.left_camera_matrix is None or self.right_camera_matrix is None or
            self.R is None or self.T is None):
            return
            
        # Compute stereo rectification
        (self.R1, self.R2, self.P1, self.P2, self.Q, _, _) = cv2.stereoRectify(
            self.left_camera_matrix, self.left_distortion,
            self.right_camera_matrix, self.right_distortion,
            image_size, self.R, self.T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        
        # Compute rectification maps
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
            self.left_camera_matrix, self.left_distortion, self.R1, self.P1,
            image_size, cv2.CV_32FC1
        )
        
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            self.right_camera_matrix, self.right_distortion, self.R2, self.P2,
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
            raise ValueError("Rectification maps not computed. Call set_camera_parameters first.")
            
        # Rectify images
        left_rectified = cv2.remap(left_image, self.left_map1, self.left_map2, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_image, self.right_map1, self.right_map2, cv2.INTER_LINEAR)
        
        return left_rectified, right_rectified
    
    def triangulate_points(
        self, 
        left_points: np.ndarray, 
        right_points: np.ndarray,
        use_rectified: bool = True
    ) -> np.ndarray:
        """
        Triangulate 3D points from corresponding 2D points in stereo images.
        
        Args:
            left_points: Points in left image (Nx2)
            right_points: Corresponding points in right image (Nx2)
            use_rectified: Whether the points are from rectified images
            
        Returns:
            3D points in camera coordinate system (Nx3)
        """
        if left_points.shape[0] == 0 or right_points.shape[0] == 0:
            return np.array([])
            
        if left_points.shape != right_points.shape:
            raise ValueError("Number of points in left and right images must be the same")
            
        # Reshape to the format expected by OpenCV
        left_points = left_points.reshape(-1, 1, 2).astype(np.float32)
        right_points = right_points.reshape(-1, 1, 2).astype(np.float32)
        
        if use_rectified:
            if self.P1 is None or self.P2 is None:
                raise ValueError("Projection matrices not computed. Call set_camera_parameters first.")
                
            # For rectified images, use the projection matrices
            projection_matrix1 = self.P1
            projection_matrix2 = self.P2
        else:
            if (self.left_camera_matrix is None or self.right_camera_matrix is None or
                self.R is None or self.T is None):
                raise ValueError("Camera parameters not set. Call set_camera_parameters first.")
                
            # For non-rectified images, compute projection matrices
            # P1 = K1 * [I | 0]
            projection_matrix1 = np.hstack((self.left_camera_matrix, np.zeros((3, 1))))
            
            # P2 = K2 * [R | T]
            RT = np.hstack((self.R, self.T.reshape(3, 1)))
            projection_matrix2 = np.dot(self.right_camera_matrix, RT)
        
        # Triangulate points
        points_4d = cv2.triangulatePoints(
            projection_matrix1, projection_matrix2,
            left_points.reshape(-1, 2).T, right_points.reshape(-1, 2).T
        )
        
        # Convert from homogeneous coordinates to 3D
        points_3d = points_4d[:3, :] / points_4d[3, :]
        
        return points_3d.T
    
    def triangulate_point(
        self, 
        left_point: Tuple[float, float], 
        right_point: Tuple[float, float],
        use_rectified: bool = True
    ) -> np.ndarray:
        """
        Triangulate a single 3D point from corresponding stereo points.
        
        Args:
            left_point: Point in left image (x, y)
            right_point: Corresponding point in right image (x, y)
            use_rectified: Whether the points are from rectified images
            
        Returns:
            3D point in camera coordinate system [X, Y, Z]
        """
        left_points = np.array([left_point], dtype=np.float32)
        right_points = np.array([right_point], dtype=np.float32)
        
        points_3d = self.triangulate_points(left_points, right_points, use_rectified)
        
        if len(points_3d) == 0:
            return None
            
        return points_3d[0]
    
    def compute_disparity_map(
        self, 
        left_image: np.ndarray, 
        right_image: np.ndarray,
        min_disparity: int = 0,
        num_disparities: int = 64,
        block_size: int = 11,
        use_rectified: bool = True
    ) -> np.ndarray:
        """
        Compute disparity map from stereo images.
        
        Args:
            left_image: Image from left camera
            right_image: Image from right camera
            min_disparity: Minimum disparity value
            num_disparities: Number of disparity values, must be divisible by 16
            block_size: Size of the block for matching, odd number
            use_rectified: Whether to rectify images before computing disparity
            
        Returns:
            Disparity map
        """
        # Convert to grayscale if needed
        if len(left_image.shape) == 3:
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_image
            right_gray = right_image
        
        # Rectify images if required
        if use_rectified:
            left_gray, right_gray = self.rectify_images(left_gray, right_gray)
        
        # Create StereoBM object
        stereo = cv2.StereoBM_create(
            numDisparities=num_disparities,
            blockSize=block_size
        )
        stereo.setMinDisparity(min_disparity)
        
        # Compute disparity map
        disparity = stereo.compute(left_gray, right_gray)
        
        # Normalize for visualization
        normalized_disparity = cv2.normalize(
            disparity, None, alpha=0, beta=255, 
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        
        return normalized_disparity
    
    def reproject_disparity_to_3d(self, disparity_map: np.ndarray) -> np.ndarray:
        """
        Reproject a disparity map to 3D points.
        
        Args:
            disparity_map: Disparity map computed with compute_disparity_map
            
        Returns:
            3D points map (height x width x 3)
        """
        if self.Q is None:
            raise ValueError("Q matrix not computed. Call set_camera_parameters first.")
        
        # Reproject to 3D
        points_3d = cv2.reprojectImageTo3D(disparity_map, self.Q)
        
        return points_3d
    
    def calibrate_from_stereo_images(
        self,
        left_images: List[np.ndarray],
        right_images: List[np.ndarray],
        board_size: Tuple[int, int] = (9, 6),
        square_size: float = 0.025  # in meters
    ) -> bool:
        """
        Calibrate stereo camera system from images of a calibration board.
        
        Args:
            left_images: List of images from left camera
            right_images: List of images from right camera
            board_size: Number of inner corners of the calibration board (width, height)
            square_size: Size of the squares on the calibration board in meters
            
        Returns:
            True if calibration was successful, False otherwise
        """
        if len(left_images) != len(right_images) or len(left_images) < 5:
            return False
        
        # Prepare object points (3D points in real world)
        objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size
        
        # Arrays to store object points and image points
        objpoints = []  # 3D points in real world space
        left_imgpoints = []  # 2D points in left image plane
        right_imgpoints = []  # 2D points in right image plane
        
        # Find the corners in each stereo pair
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
        
        if len(objpoints) < 5:  # Need at least 5 valid stereo pairs
            return False
        
        # Get image size
        img_size = (left_gray.shape[1], left_gray.shape[0])
        
        # Calibrate left and right cameras separately
        left_ret, self.left_camera_matrix, self.left_distortion, left_rvecs, left_tvecs = cv2.calibrateCamera(
            objpoints, left_imgpoints, img_size, None, None
        )
        
        right_ret, self.right_camera_matrix, self.right_distortion, right_rvecs, right_tvecs = cv2.calibrateCamera(
            objpoints, right_imgpoints, img_size, None, None
        )
        
        # Calibrate stereo system
        stereocalib_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        
        stereocalib_ret, self.left_camera_matrix, self.left_distortion, \
        self.right_camera_matrix, self.right_distortion, self.R, self.T, \
        E, F = cv2.stereoCalibrate(
            objpoints, left_imgpoints, right_imgpoints,
            self.left_camera_matrix, self.left_distortion,
            self.right_camera_matrix, self.right_distortion,
            img_size, flags=cv2.CALIB_FIX_INTRINSIC,
            criteria=stereocalib_criteria
        )
        
        # Compute rectification
        self._compute_rectification(img_size)
        
        return stereocalib_ret < 1.0  # Typically, a good calibration has RMS error < 1.0
    
    def draw_epipolar_lines(
        self, 
        left_image: np.ndarray, 
        right_image: np.ndarray, 
        left_points: np.ndarray, 
        right_points: np.ndarray,
        use_rectified: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw epipolar lines on stereo images.
        
        Args:
            left_image: Image from left camera
            right_image: Image from right camera
            left_points: Points in left image (Nx2)
            right_points: Corresponding points in right image (Nx2)
            use_rectified: Whether the images are already rectified
            
        Returns:
            Tuple of (left_image, right_image) with epipolar lines drawn
        """
        if use_rectified:
            # For rectified images, epipolar lines are horizontal
            h, w = left_image.shape[:2]
            left_result = left_image.copy()
            right_result = right_image.copy()
            
            for pt_left, pt_right in zip(left_points, right_points):
                y = int(pt_left[1])  # In rectified images, y-coordinates should be the same
                
                # Draw horizontal lines
                cv2.line(left_result, (0, y), (w, y), (0, 255, 0), 1)
                cv2.line(right_result, (0, y), (w, y), (0, 255, 0), 1)
                
                # Draw the points
                cv2.circle(left_result, (int(pt_left[0]), int(pt_left[1])), 5, (0, 0, 255), -1)
                cv2.circle(right_result, (int(pt_right[0]), int(pt_right[1])), 5, (0, 0, 255), -1)
        else:
            # For non-rectified images, compute fundamental matrix
            F, _ = cv2.findFundamentalMat(left_points, right_points, cv2.FM_8POINT)
            
            # Draw epipolar lines in both images
            left_result, right_result = self._draw_epipolar_lines_with_F(
                left_image, right_image, left_points, right_points, F
            )
        
        return left_result, right_result
    
    def _draw_epipolar_lines_with_F(
        self, 
        left_image: np.ndarray, 
        right_image: np.ndarray, 
        left_points: np.ndarray, 
        right_points: np.ndarray, 
        F: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw epipolar lines using the fundamental matrix F.
        
        Args:
            left_image: Image from left camera
            right_image: Image from right camera
            left_points: Points in left image (Nx2)
            right_points: Corresponding points in right image (Nx2)
            F: Fundamental matrix
            
        Returns:
            Tuple of (left_image, right_image) with epipolar lines drawn
        """
        h, w = left_image.shape[:2]
        left_result = left_image.copy()
        right_result = right_image.copy()
        
        # Convert points to homogeneous coordinates if needed
        if left_points.shape[1] == 2:
            left_points_h = np.hstack((left_points, np.ones((left_points.shape[0], 1))))
            right_points_h = np.hstack((right_points, np.ones((right_points.shape[0], 1))))
        else:
            left_points_h = left_points
            right_points_h = right_points
        
        # Draw lines on the right image corresponding to left points
        for i, pt in enumerate(left_points_h):
            # Get epipolar line in the right image
            line_right = np.dot(F, pt)
            
            # Draw the line (ax + by + c = 0)
            a, b, c = line_right
            if abs(b) > 1e-5:
                x0, x1 = 0, w
                y0 = int(-(a * x0 + c) / b)
                y1 = int(-(a * x1 + c) / b)
                cv2.line(right_result, (x0, y0), (x1, y1), (0, 255, 0), 1)
            
            # Draw the point in the left image
            cv2.circle(left_result, (int(left_points[i][0]), int(left_points[i][1])), 5, (0, 0, 255), -1)
        
        # Draw lines on the left image corresponding to right points
        for i, pt in enumerate(right_points_h):
            # Get epipolar line in the left image
            line_left = np.dot(F.T, pt)
            
            # Draw the line (ax + by + c = 0)
            a, b, c = line_left
            if abs(b) > 1e-5:
                x0, x1 = 0, w
                y0 = int(-(a * x0 + c) / b)
                y1 = int(-(a * x1 + c) / b)
                cv2.line(left_result, (x0, y0), (x1, y1), (255, 0, 0), 1)
            
            # Draw the point in the right image
            cv2.circle(right_result, (int(right_points[i][0]), int(right_points[i][1])), 5, (0, 0, 255), -1)
        
        return left_result, right_result 