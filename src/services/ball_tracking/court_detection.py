#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tennis court detection module.
This module contains functions to detect the tennis court plane
and define the court coordinate system.
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional


class CourtDetector:
    """
    Tennis court detector class for detecting the court plane
    and calculating the court-to-world transformation.
    """

    def __init__(self):
        """Initialize the court detector with default parameters."""
        # Court dimensions in meters (standard doubles court)
        self.court_width = 10.97  # width of doubles court
        self.court_length = 23.77  # length of court
        self.service_line_distance = 6.40  # distance from baseline to service line
        self.center_mark_distance = 11.885  # distance from baseline to center mark
        
        # Court corner points in world coordinate system (meters)
        # Origin at the center of the court, z=0 at court level
        self.court_points_3d = {
            'top_left': np.array([-self.court_width/2, -self.court_length/2, 0]),
            'top_right': np.array([self.court_width/2, -self.court_length/2, 0]),
            'bottom_left': np.array([-self.court_width/2, self.court_length/2, 0]),
            'bottom_right': np.array([self.court_width/2, self.court_length/2, 0]),
            'service_top_left': np.array([-self.court_width/2, -self.court_length/2 + self.service_line_distance, 0]),
            'service_top_right': np.array([self.court_width/2, -self.court_length/2 + self.service_line_distance, 0]),
            'service_bottom_left': np.array([-self.court_width/2, self.court_length/2 - self.service_line_distance, 0]),
            'service_bottom_right': np.array([self.court_width/2, self.court_length/2 - self.service_line_distance, 0]),
            'center_top': np.array([0, -self.court_length/2, 0]),
            'center_bottom': np.array([0, self.court_length/2, 0]),
            'center_net': np.array([0, 0, 0]),
            'center_service_top': np.array([0, -self.court_length/2 + self.service_line_distance, 0]),
            'center_service_bottom': np.array([0, self.court_length/2 - self.service_line_distance, 0]),
        }
        
        # Transformation matrices
        self.R_world = None  # Rotation matrix from camera to world
        self.T_world = None  # Translation vector from camera to world
        
        # Court plane equation: ax + by + cz + d = 0
        self.court_plane = None  # [a, b, c, d]
        
        # Line detector parameters
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150
        self.hough_threshold = 50
        self.hough_min_line_length = 50
        self.hough_max_line_gap = 10
        
    def detect_court_lines(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Detect court lines in the image.
        
        Args:
            img: Input image
            
        Returns:
            List of detected lines in format [x1, y1, x2, y2]
        """
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)
        
        # Apply Hough Line Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )
        
        # Convert lines to [x1, y1, x2, y2] format
        detected_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                detected_lines.append(np.array([x1, y1, x2, y2]))
        
        return detected_lines

    def filter_horizontal_vertical_lines(
        self, lines: List[np.ndarray], angle_threshold: float = 10
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Separate detected lines into horizontal and vertical lines.
        
        Args:
            lines: List of detected lines [x1, y1, x2, y2]
            angle_threshold: Angle threshold in degrees
            
        Returns:
            Tuple of (horizontal lines, vertical lines)
        """
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            
            # Calculate angle with horizontal
            dx = x2 - x1
            dy = y2 - y1
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            
            # Classify as horizontal or vertical
            if angle < angle_threshold or angle > 180 - angle_threshold:
                horizontal_lines.append(line)
            elif angle > 90 - angle_threshold and angle < 90 + angle_threshold:
                vertical_lines.append(line)
        
        return horizontal_lines, vertical_lines

    def find_line_intersections(
        self, horizontal_lines: List[np.ndarray], vertical_lines: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Find intersections between horizontal and vertical lines.
        
        Args:
            horizontal_lines: List of horizontal lines [x1, y1, x2, y2]
            vertical_lines: List of vertical lines [x1, y1, x2, y2]
            
        Returns:
            List of intersection points [x, y]
        """
        intersections = []
        
        for h_line in horizontal_lines:
            x1, y1, x2, y2 = h_line
            h_a = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
            h_b = y1 - h_a * x1 if x2 != x1 else None
            
            for v_line in vertical_lines:
                x3, y3, x4, y4 = v_line
                v_a = (y4 - y3) / (x4 - x3) if x4 != x3 else float('inf')
                v_b = y3 - v_a * x3 if x4 != x3 else None
                
                # Check if lines are parallel
                if abs(h_a - v_a) < 1e-6:
                    continue
                
                # Calculate intersection
                if h_a == float('inf'):
                    x_intersect = x1
                    y_intersect = v_a * x_intersect + v_b
                elif v_a == float('inf'):
                    x_intersect = x3
                    y_intersect = h_a * x_intersect + h_b
                else:
                    x_intersect = (v_b - h_b) / (h_a - v_a)
                    y_intersect = h_a * x_intersect + h_b
                
                # Check if intersection is within line segments
                if (min(x1, x2) <= x_intersect <= max(x1, x2) and
                    min(y1, y2) <= y_intersect <= max(y1, y2) and
                    min(x3, x4) <= x_intersect <= max(x3, x4) and
                    min(y3, y4) <= y_intersect <= max(y3, y4)):
                    intersections.append(np.array([x_intersect, y_intersect]))
        
        return intersections

    def match_court_corners(
        self, intersections: List[np.ndarray], img_shape: Tuple[int, int]
    ) -> Dict[str, np.ndarray]:
        """
        Match detected intersections to court corners.
        
        Args:
            intersections: List of intersection points [x, y]
            img_shape: Image shape (height, width)
            
        Returns:
            Dictionary mapping corner names to 2D coordinates
        """
        if len(intersections) < 4:
            raise ValueError("Not enough intersections to match court corners")
        
        # Convert to numpy array
        points = np.array(intersections)
        
        # Sort points by y-coordinate (top to bottom)
        sorted_by_y = points[np.argsort(points[:, 1])]
        
        # Split into top and bottom halves
        middle_y = img_shape[0] // 2
        top_points = sorted_by_y[sorted_by_y[:, 1] < middle_y]
        bottom_points = sorted_by_y[sorted_by_y[:, 1] >= middle_y]
        
        # For both halves, sort by x-coordinate (left to right)
        if len(top_points) >= 2:
            top_points = top_points[np.argsort(top_points[:, 0])]
        if len(bottom_points) >= 2:
            bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]
        
        # Assign corners
        corners = {}
        if len(top_points) >= 2:
            corners['top_left'] = top_points[0]
            corners['top_right'] = top_points[-1]
        if len(bottom_points) >= 2:
            corners['bottom_left'] = bottom_points[0]
            corners['bottom_right'] = bottom_points[-1]
        
        return corners

    def calculate_homography(
        self, image_corners: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Calculate homography matrix from image corners to real-world court corners.
        
        Args:
            image_corners: Dictionary mapping corner names to 2D image coordinates
            
        Returns:
            Homography matrix H
        """
        # Check if we have the basic four corners
        required_corners = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
        if not all(corner in image_corners for corner in required_corners):
            raise ValueError("Missing required court corners")
        
        # Prepare source and destination points
        src_points = []
        dst_points = []
        
        for corner in required_corners:
            src_points.append(image_corners[corner])
            # Get x, y coordinates (ignore z=0)
            dst_points.append(self.court_points_3d[corner][:2])
        
        # Calculate homography
        src_points = np.array(src_points)
        dst_points = np.array(dst_points)
        H, _ = cv2.findHomography(src_points, dst_points)
        
        return H

    def estimate_court_plane(
        self, corners_2d: Dict[str, np.ndarray], camera_matrix: np.ndarray
    ) -> None:
        """
        Estimate court plane equation using PnP.
        
        Args:
            corners_2d: Dictionary mapping corner names to 2D image coordinates
            camera_matrix: Camera intrinsic matrix (3x3)
            
        Returns:
            None (sets self.court_plane, self.R_world, self.T_world)
        """
        # Check if we have at least 4 corners
        if len(corners_2d) < 4:
            raise ValueError("Need at least 4 corners to estimate plane")
        
        # Prepare object and image points
        object_points = []
        image_points = []
        
        for corner_name, corner_pos in corners_2d.items():
            if corner_name in self.court_points_3d:
                object_points.append(self.court_points_3d[corner_name])
                image_points.append(corner_pos)
        
        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)
        
        # Solve PnP
        _, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            None
        )
        
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Store transformation matrices
        self.R_world = R
        self.T_world = tvec
        
        # Calculate plane normal in camera coordinates
        # Court plane normal in world coordinates is [0, 0, 1]
        normal_world = np.array([0, 0, 1])
        normal_cam = R.T @ normal_world
        
        # Calculate plane equation in camera coordinates: ax + by + cz + d = 0
        a, b, c = normal_cam
        d = -np.dot(normal_cam, tvec.flatten())
        
        self.court_plane = np.array([a, b, c, d])

    def project_point_to_court(
        self, point_3d: np.ndarray
    ) -> np.ndarray:
        """
        Project a 3D point onto the court plane.
        
        Args:
            point_3d: 3D point in camera coordinates
            
        Returns:
            Projected point on court plane [x, y, z]
        """
        if self.court_plane is None:
            raise ValueError("Court plane not defined")
        
        # Extract plane parameters
        a, b, c, d = self.court_plane
        
        # Calculate intersection of line from origin through point with plane
        # Parametric line equation: point = t * direction
        direction = point_3d / np.linalg.norm(point_3d)
        
        # Solve for t: a*t*dx + b*t*dy + c*t*dz + d = 0
        t = -d / (a * direction[0] + b * direction[1] + c * direction[2])
        
        # Calculate intersection point
        intersection = t * direction
        
        return intersection

    def is_point_inside_court(
        self, point_3d: np.ndarray, margin: float = 0.0
    ) -> bool:
        """
        Check if a 3D point is inside the court boundaries.
        
        Args:
            point_3d: 3D point in world coordinates
            margin: Margin to extend court boundaries (meters)
            
        Returns:
            True if point is inside court, False otherwise
        """
        # Transform to world coordinates if needed
        if self.R_world is not None and self.T_world is not None:
            point_world = self.R_world @ point_3d + self.T_world.flatten()
        else:
            point_world = point_3d
        
        # Check if point is within court boundaries
        half_width = self.court_width / 2 + margin
        half_length = self.court_length / 2 + margin
        
        return (
            -half_width <= point_world[0] <= half_width and
            -half_length <= point_world[1] <= half_length
        )

    def calculate_court_to_image_homography(
        self, corners_2d: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Calculate homography matrix from world court to image.
        
        Args:
            corners_2d: Dictionary mapping corner names to 2D image coordinates
            
        Returns:
            Homography matrix H (court to image)
        """
        # Check if we have the basic four corners
        required_corners = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
        if not all(corner in corners_2d for corner in required_corners):
            raise ValueError("Missing required court corners")
        
        # Prepare source and destination points
        src_points = []
        dst_points = []
        
        for corner in required_corners:
            # Get x, y coordinates (ignore z=0)
            src_points.append(self.court_points_3d[corner][:2])
            dst_points.append(corners_2d[corner])
        
        # Calculate homography
        src_points = np.array(src_points)
        dst_points = np.array(dst_points)
        H, _ = cv2.findHomography(src_points, dst_points)
        
        return H

    def transform_world_to_image(
        self, point_world: np.ndarray, H: np.ndarray
    ) -> np.ndarray:
        """
        Transform a world point to image coordinates using homography.
        
        Args:
            point_world: Point in world coordinates [x, y, z]
            H: Homography matrix from world to image
            
        Returns:
            Point in image coordinates [x, y]
        """
        # Get x, y coordinates (ignore z)
        point = point_world[:2]
        
        # Convert to homogeneous coordinates
        point_h = np.array([point[0], point[1], 1.0])
        
        # Apply homography
        transformed_h = H @ point_h
        
        # Convert back from homogeneous coordinates
        transformed = transformed_h[:2] / transformed_h[2]
        
        return transformed

    def detect_and_initialize(
        self, img: np.ndarray, camera_matrix: np.ndarray
    ) -> bool:
        """
        Detect court and initialize transformation matrices.
        
        Args:
            img: Input image
            camera_matrix: Camera intrinsic matrix (3x3)
            
        Returns:
            True if detection and initialization successful, False otherwise
        """
        try:
            # Detect court lines
            lines = self.detect_court_lines(img)
            
            # Filter lines
            horizontal_lines, vertical_lines = self.filter_horizontal_vertical_lines(lines)
            
            # Find intersections
            intersections = self.find_line_intersections(horizontal_lines, vertical_lines)
            
            # Match court corners
            corners = self.match_court_corners(intersections, img.shape[:2])
            
            # Calculate court plane and transformations
            self.estimate_court_plane(corners, camera_matrix)
            
            return True
        except Exception as e:
            print(f"Court detection failed: {e}")
            return False

    def draw_court_overlay(
        self, img: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2
    ) -> np.ndarray:
        """
        Draw court overlay on image.
        
        Args:
            img: Input image
            color: Line color (B, G, R)
            thickness: Line thickness
            
        Returns:
            Image with court overlay
        """
        if self.R_world is None or self.T_world is None:
            return img
        
        # Create copy of image
        result = img.copy()
        
        # Get court points in world coordinates
        court_points = list(self.court_points_3d.values())
        
        # Project points to image
        projected_points = []
        for point in court_points:
            # Transform from world to camera coordinates
            point_cam = np.linalg.inv(self.R_world) @ (point - self.T_world.flatten())
            
            # Project to image
            point_img = cv2.projectPoints(
                point_cam.reshape(1, 3),
                np.zeros(3),
                np.zeros(3),
                np.eye(3),
                None
            )[0][0][0]
            
            projected_points.append(point_img)
        
        # Draw court lines
        # Main court rectangle
        cv2.line(result, tuple(map(int, projected_points[0])), tuple(map(int, projected_points[1])), color, thickness)
        cv2.line(result, tuple(map(int, projected_points[1])), tuple(map(int, projected_points[3])), color, thickness)
        cv2.line(result, tuple(map(int, projected_points[3])), tuple(map(int, projected_points[2])), color, thickness)
        cv2.line(result, tuple(map(int, projected_points[2])), tuple(map(int, projected_points[0])), color, thickness)
        
        # Service lines
        cv2.line(result, tuple(map(int, projected_points[4])), tuple(map(int, projected_points[5])), color, thickness)
        cv2.line(result, tuple(map(int, projected_points[6])), tuple(map(int, projected_points[7])), color, thickness)
        
        # Center line
        cv2.line(result, tuple(map(int, projected_points[8])), tuple(map(int, projected_points[9])), color, thickness)
        
        return result 