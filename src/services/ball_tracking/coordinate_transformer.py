#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Coordinate Transformer module.
This module provides functionality for transforming between image and world coordinates.

[DEPRECATED] This is legacy code that will be removed in future versions.
Please use the new coordinate transformation system instead.
"""

import numpy as np
import cv2
import logging
from typing import Tuple, List, Optional, Dict, Any

# Add legacy warning logger
logger = logging.getLogger(__name__)

class SingleViewCoordinateTransformer:
    """
    Transform between image and world coordinates for a single camera view.
    
    [DEPRECATED] This is legacy code that will be removed in future versions.
    """
    
    def __init__(self, 
                 camera_matrix: np.ndarray, 
                 rotation_vector: np.ndarray, 
                 translation_vector: np.ndarray,
                 distortion_coeffs: Optional[np.ndarray] = None):
        """
        Initialize the coordinate transformer.
        
        Args:
            camera_matrix: Camera intrinsic matrix (3x3)
            rotation_vector: Camera rotation vector (3,) in Rodrigues form
            translation_vector: Camera translation vector (3,)
            distortion_coeffs: Camera distortion coefficients
        """
        # Log deprecation warning
        logger.warning("SingleViewCoordinateTransformer is deprecated and will be removed in future versions")
        
        self.camera_matrix = camera_matrix
        self.rotation_vector = rotation_vector
        self.translation_vector = translation_vector
        self.distortion_coeffs = distortion_coeffs if distortion_coeffs is not None else np.zeros(5)
        
        # Convert rotation vector to matrix
        self.rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Cache the inverse matrices
        self.inv_camera_matrix = np.linalg.inv(camera_matrix)
        self.inv_rotation_matrix = np.linalg.inv(self.rotation_matrix)
        
        # Homography from image to world (calculated when calibrated)
        self.H_image_to_world = None
        
        # Court corner points in world coordinates
        self.court_corners_world = None
        
    def set_extrinsic_parameters(self, rotation_matrix: np.ndarray, translation_vector: np.ndarray):
        """
        Set extrinsic camera parameters.
        
        Args:
            rotation_matrix: 3x3 rotation matrix from world to camera
            translation_vector: 3x1 translation vector from world to camera
        """
        self.rotation_matrix = rotation_matrix
        self.translation_vector = translation_vector
        
    def set_intrinsic_parameters(self, camera_matrix: np.ndarray):
        """
        Set intrinsic camera parameters.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
        """
        self.camera_matrix = camera_matrix
        
    def calibrate_from_court_corners(
        self,
        image_points: List[Tuple[float, float]],
        world_points: List[Tuple[float, float, float]],
    ) -> bool:
        """
        Calibrate transformer from corresponding image and world court corner points.
        
        Args:
            image_points: List of corner points in image coordinates [(x1, y1), ...]
            world_points: List of corner points in world coordinates [(X1, Y1, Z1), ...]
            
        Returns:
            True if calibration was successful, False otherwise
        """
        if len(image_points) < 4 or len(world_points) < 4:
            return False
        
        # Store court corners in world coordinates
        self.court_corners_world = np.array(world_points)
        
        # Convert world points to 2D (assume Z=0 for court plane)
        world_points_2d = np.array([[p[0], p[1]] for p in world_points], dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)
        
        # Calculate homography from image to world
        self.H_image_to_world, _ = cv2.findHomography(image_points, world_points_2d)
        
        # Estimate extrinsic parameters if camera matrix is available
        if self.camera_matrix is not None:
            # Use PnP to get better extrinsic parameters
            world_points_3d = np.array(world_points, dtype=np.float32)
            _, self.rotation_matrix, self.translation_vector = cv2.solvePnP(
                world_points_3d, 
                image_points, 
                self.camera_matrix, 
                None, 
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            # Convert rotation vector to rotation matrix
            self.rotation_matrix, _ = cv2.Rodrigues(self.rotation_matrix)
            
        return True
    
    def camera_to_world(self, camera_point: np.ndarray) -> np.ndarray:
        """
        Transform a point from camera coordinates to world coordinates.
        
        Args:
            camera_point: 3D point in camera coordinates [x, y, z]
            
        Returns:
            3D point in world coordinates [X, Y, Z]
        """
        if self.rotation_matrix is None or self.translation_vector is None:
            raise ValueError("Camera extrinsic parameters not set. Call calibrate_from_court_corners first.")
        
        # p_world = R^T * (p_camera - t)
        camera_point = np.array(camera_point).reshape(3, 1)
        world_point = np.dot(self.rotation_matrix.T, camera_point - self.translation_vector)
        
        return world_point.flatten()
    
    def world_to_camera(self, world_point: np.ndarray) -> np.ndarray:
        """
        Transform a point from world coordinates to camera coordinates.
        
        Args:
            world_point: 3D point in world coordinates [X, Y, Z]
            
        Returns:
            3D point in camera coordinates [x, y, z]
        """
        if self.rotation_matrix is None or self.translation_vector is None:
            raise ValueError("Camera extrinsic parameters not set. Call calibrate_from_court_corners first.")
        
        # p_camera = R * p_world + t
        world_point = np.array(world_point).reshape(3, 1)
        camera_point = np.dot(self.rotation_matrix, world_point) + self.translation_vector
        
        return camera_point.flatten()
    
    def world_to_image(self, world_point: np.ndarray) -> Tuple[float, float]:
        """
        Project a 3D world point to image coordinates.
        
        Args:
            world_point: 3D point in world coordinates [X, Y, Z]
            
        Returns:
            2D point in image coordinates (u, v)
        """
        if self.camera_matrix is None:
            raise ValueError("Camera intrinsic parameters not set")
        
        # First transform to camera coordinates
        camera_point = self.world_to_camera(world_point)
        
        # Project to image coordinates
        camera_point_h = camera_point.reshape(3, 1)
        image_point_h = np.dot(self.camera_matrix, camera_point_h)
        
        # Convert from homogeneous coordinates
        u = image_point_h[0, 0] / image_point_h[2, 0]
        v = image_point_h[1, 0] / image_point_h[2, 0]
        
        return (u, v)
    
    def image_to_court_plane(self, image_point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Convert image point to world coordinates on the court plane (Z=0).
        
        Args:
            image_point: 2D point in image coordinates (u, v)
            
        Returns:
            2D point on court plane (X, Y) where Z=0
        """
        if self.H_image_to_world is None:
            raise ValueError("Homography not calculated. Call calibrate_from_court_corners first.")
        
        # Convert to homogeneous coordinates
        image_point_h = np.array([[image_point[0], image_point[1], 1]], dtype=np.float32).T
        
        # Apply homography
        world_point_h = np.dot(self.H_image_to_world, image_point_h)
        
        # Convert from homogeneous coordinates
        X = world_point_h[0, 0] / world_point_h[2, 0]
        Y = world_point_h[1, 0] / world_point_h[2, 0]
        
        return (X, Y)
    
    def stereo_to_world(
        self, 
        point_3d: np.ndarray, 
        camera_position: str = 'left'
    ) -> np.ndarray:
        """
        Transform a 3D point from stereo camera coordinates to world coordinates.
        
        Args:
            point_3d: 3D point from stereo triangulation [X, Y, Z]
            camera_position: Which camera serves as the reference ('left' or 'right')
            
        Returns:
            3D point in world coordinates [X, Y, Z]
        """
        # The point_3d is already in camera coordinates system
        # Just need to transform from camera to world
        return self.camera_to_world(point_3d)
    
    def ray_plane_intersection(
        self, 
        camera_center: np.ndarray, 
        ray_direction: np.ndarray, 
        plane_point: np.ndarray = np.array([0, 0, 0]), 
        plane_normal: np.ndarray = np.array([0, 0, 1])
    ) -> Optional[np.ndarray]:
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
    
    def image_to_ray(self, image_point: Tuple[float, float]) -> np.ndarray:
        """
        Convert an image point to a ray direction in camera coordinates.
        
        Args:
            image_point: 2D point in image coordinates (u, v)
            
        Returns:
            Ray direction vector in camera coordinates [dx, dy, dz]
        """
        if self.camera_matrix is None:
            raise ValueError("Camera intrinsic parameters not set")
        
        # Get normalized image coordinates
        u, v = image_point
        K_inv = np.linalg.inv(self.camera_matrix)
        
        # Homogeneous image coordinates
        image_point_h = np.array([u, v, 1]).reshape(3, 1)
        
        # Ray direction in camera coordinates
        ray = np.dot(K_inv, image_point_h)
        ray = ray / np.linalg.norm(ray)
        
        return ray.flatten()
    
    def image_to_world_ray(self, image_point: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert an image point to a ray (origin and direction) in world coordinates.
        
        Args:
            image_point: 2D point in image coordinates (u, v)
            
        Returns:
            Tuple of (ray_origin, ray_direction) in world coordinates
        """
        if self.rotation_matrix is None or self.translation_vector is None or self.camera_matrix is None:
            raise ValueError("Camera parameters not set")
        
        # Get ray in camera coordinates
        ray_camera = self.image_to_ray(image_point)
        
        # Camera center in world coordinates
        camera_center_world = self.camera_to_world(np.zeros(3))
        
        # Transform ray direction to world coordinates
        ray_world = np.dot(self.rotation_matrix.T, ray_camera.reshape(3, 1)).flatten()
        ray_world = ray_world / np.linalg.norm(ray_world)
        
        return camera_center_world, ray_world
    
    def tennis_court_to_world_points(self, court_dims: Dict[str, float]) -> Dict[str, np.ndarray]:
        """
        Generate world coordinates for standard tennis court points based on dimensions.
        
        Args:
            court_dims: Dictionary with court dimensions in meters
                - 'length': Full court length (baseline to baseline)
                - 'width': Full court width (sideline to sideline)
                - 'service_line': Distance from baseline to service line
                - 'center_service_line': Length of center service line
                
        Returns:
            Dictionary mapping point names to world coordinates [X, Y, Z]
        """
        length = court_dims.get('length', 23.77)  # Standard tennis court length
        width = court_dims.get('width', 8.23)     # Singles court width
        service_line = court_dims.get('service_line', 6.40)
        center_line = court_dims.get('center_service_line', 4.115)  # Half of singles width
        
        # Define the origin at the center of the court
        half_length = length / 2
        half_width = width / 2
        
        # Define all court points
        court_points = {
            # Corners of the court
            'bottom_left': np.array([-half_width, -half_length, 0]),
            'bottom_right': np.array([half_width, -half_length, 0]),
            'top_left': np.array([-half_width, half_length, 0]),
            'top_right': np.array([half_width, half_length, 0]),
            
            # Corners of the service boxes
            'service_bottom_left': np.array([-half_width, -half_length + service_line, 0]),
            'service_bottom_right': np.array([half_width, -half_length + service_line, 0]),
            'service_top_left': np.array([-half_width, half_length - service_line, 0]),
            'service_top_right': np.array([half_width, half_length - service_line, 0]),
            
            # Center points
            'center_mark_bottom': np.array([0, -half_length, 0]),
            'center_mark_top': np.array([0, half_length, 0]),
            'center_service_line_bottom': np.array([0, -half_length + service_line, 0]),
            'center_service_line_top': np.array([0, half_length - service_line, 0]),
            'center_net': np.array([0, 0, 0]),
            
            # Center line points
            'center_baseline_bottom': np.array([0, -half_length, 0]),
            'center_baseline_top': np.array([0, half_length, 0]),
            
            # Net posts
            'net_post_left': np.array([-half_width - 0.914, 0, 1.07]),  # 3 feet beyond sideline, net height
            'net_post_right': np.array([half_width + 0.914, 0, 1.07]),
        }
        
        return court_points
    
    def draw_court_on_image(self, frame: np.ndarray, thickness: int = 2) -> np.ndarray:
        """
        Draw the tennis court on the image based on the calibration.
        
        Args:
            frame: Input image frame
            thickness: Line thickness
            
        Returns:
            Frame with court lines drawn
        """
        if self.court_corners_world is None or self.H_image_to_world is None:
            # Not calibrated yet
            return frame
            
        # Standard tennis court dimensions
        court_dims = {
            'length': 23.77,
            'width': 8.23,
            'service_line': 6.40,
            'center_service_line': 4.115
        }
        
        # Get court points in world coordinates
        court_points = self.tennis_court_to_world_points(court_dims)
        
        # Create a copy of the frame to draw on
        result = frame.copy()
        
        # List of court lines to draw (pairs of points)
        court_lines = [
            # Outer rectangle
            ('bottom_left', 'bottom_right'),
            ('bottom_right', 'top_right'),
            ('top_right', 'top_left'),
            ('top_left', 'bottom_left'),
            
            # Service lines
            ('service_bottom_left', 'service_bottom_right'),
            ('service_bottom_right', 'service_top_right'),
            ('service_top_right', 'service_top_left'),
            ('service_top_left', 'service_bottom_left'),
            
            # Center service line
            ('center_service_line_bottom', 'center_service_line_top'),
            
            # Center marks on baselines
            ('center_baseline_bottom', 'center_service_line_bottom'),
            ('center_baseline_top', 'center_service_line_top'),
        ]
        
        # Draw each line
        for start_name, end_name in court_lines:
            start_point_world = court_points[start_name]
            end_point_world = court_points[end_name]
            
            # Project to image coordinates
            try:
                start_point_img = self.world_to_image(start_point_world)
                end_point_img = self.world_to_image(end_point_world)
                
                # Convert to integer coordinates
                start_point_img = (int(start_point_img[0]), int(start_point_img[1]))
                end_point_img = (int(end_point_img[0]), int(end_point_img[1]))
                
                # Draw the line
                cv2.line(result, start_point_img, end_point_img, (255, 255, 255), thickness)
            except:
                # Skip lines that can't be projected correctly
                continue
        
        return result


class MultiViewCoordinateTransformer:
    """
    Coordinate transformer for multiple camera views.
    """
    
    def __init__(self):
        """Initialize the multi-view transformer with empty camera list."""
        self.camera_transformers: Dict[str, SingleViewCoordinateTransformer] = {}
        
    def add_camera(self, camera_id: str, transformer: SingleViewCoordinateTransformer):
        """
        Add a camera to the multi-view system.
        
        Args:
            camera_id: Unique identifier for the camera
            transformer: CoordinateTransformer for this camera
        """
        self.camera_transformers[camera_id] = transformer
        
    def remove_camera(self, camera_id: str):
        """
        Remove a camera from the multi-view system.
        
        Args:
            camera_id: Camera identifier to remove
        """
        if camera_id in self.camera_transformers:
            del self.camera_transformers[camera_id]
            
    def triangulate_point(
        self, 
        image_points: Dict[str, Tuple[float, float]]
    ) -> Optional[np.ndarray]:
        """
        Triangulate a 3D point from multiple camera views.
        
        Args:
            image_points: Dictionary mapping camera_id to image point (u, v)
            
        Returns:
            3D point in world coordinates [X, Y, Z] or None if triangulation fails
        """
        if len(image_points) < 2:
            return None
            
        # Collect rays from all cameras
        origins = []
        directions = []
        
        for camera_id, point in image_points.items():
            if camera_id not in self.camera_transformers:
                continue
                
            transformer = self.camera_transformers[camera_id]
            try:
                origin, direction = transformer.image_to_world_ray(point)
                origins.append(origin)
                directions.append(direction)
            except:
                continue
                
        if len(origins) < 2:
            return None
            
        # Convert to numpy arrays
        origins = np.array(origins)
        directions = np.array(directions)
        
        # Triangulate using least squares method
        # Solve: Ax = b for ray intersection
        A = np.zeros((3 * len(origins), 3 + len(origins)))
        b = np.zeros(3 * len(origins))
        
        for i in range(len(origins)):
            # For each ray: origin + t*direction = point
            A[3*i:3*i+3, 0:3] = np.eye(3)
            A[3*i:3*i+3, 3+i] = -directions[i].reshape(3)
            b[3*i:3*i+3] = origins[i].reshape(3)
            
        # Solve using least squares
        try:
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            triangulated_point = x[0:3]
            return triangulated_point
        except:
            return None
    
    def calibrate_all_cameras(
        self, 
        image_points_dict: Dict[str, List[Tuple[float, float]]],
        world_points: List[Tuple[float, float, float]]
    ) -> bool:
        """
        Calibrate all cameras from corresponding image and world points.
        
        Args:
            image_points_dict: Dictionary mapping camera_id to list of image points
            world_points: List of corresponding world points
            
        Returns:
            True if all calibrations were successful, False otherwise
        """
        success = True
        
        for camera_id, image_points in image_points_dict.items():
            if camera_id not in self.camera_transformers:
                success = False
                continue
                
            transformer = self.camera_transformers[camera_id]
            if not transformer.calibrate_from_court_corners(image_points, world_points):
                success = False
                
        return success 