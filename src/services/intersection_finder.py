#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Intersection Finder Service.
This service provides functionality for finding intersections between lines in images.
"""

import logging
import numpy as np
import cv2
from typing import List, Tuple, Optional
from sklearn.cluster import KMeans


class IntersectionFinder:
    """
    Service for finding intersections between lines in images.
    """
    
    def __init__(self):
        """Initialize the intersection finder service."""
        self.logger = logging.getLogger(__name__)
        
    def find_intersections(self, image: np.ndarray, roi_padding: int = 100) -> List[Tuple[int, int]]:
        """
        Find court line intersections in an image.
        
        Args:
            image: The image to find intersections in
            roi_padding: Padding for the region of interest (default 100 pixels)
            
        Returns:
            List of (x, y) intersection points
        """
        # Adjust parameters based on image size
        height, width = image.shape[:2]
        self.logger.info(f"Finding intersections in image of size {width}x{height}")
        
        # Adaptive parameters based on image size
        min_line_length = min(width, height) // 15
        max_line_gap = min(width, height) // 40
        threshold = max(50, min(width, height) // 20)
        rho = 1
        theta = np.pi / 180
        
        # Create a larger ROI with more padding
        roi_padding = max(roi_padding, int(min(width, height) * 0.1))  # At least 10% of smaller dimension
        self.logger.info(f"Using ROI padding: {roi_padding}")
        
        # Preprocess the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Create mask from edges to restrict ROI to areas with potential lines
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(edges)
        
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small noise
                x, y, w, h = cv2.boundingRect(contour)
                # Expand ROI with padding
                x_min = max(0, x - roi_padding)
                y_min = max(0, y - roi_padding)
                x_max = min(width, x + w + roi_padding)
                y_max = min(height, y + h + roi_padding)
                cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)
        
        # Apply mask to edges
        edges = cv2.bitwise_and(edges, edges, mask=mask)
        
        # Find lines using HoughLinesP
        lines = cv2.HoughLinesP(
            edges, 
            rho=rho, 
            theta=theta, 
            threshold=threshold, 
            minLineLength=min_line_length, 
            maxLineGap=max_line_gap
        )
        
        # If no lines found or very few, try with more lenient parameters
        if lines is None or len(lines) < 5:
            self.logger.warning(f"Few or no lines found ({0 if lines is None else len(lines)}). Trying more lenient parameters.")
            lines = cv2.HoughLinesP(
                edges, 
                rho=rho, 
                theta=theta, 
                threshold=max(20, threshold // 2), 
                minLineLength=min_line_length // 2, 
                maxLineGap=max_line_gap * 2
            )
        
        # If still insufficient lines, try using contour analysis approach
        if lines is None or len(lines) < 5:
            self.logger.warning("Still insufficient lines. Trying contour analysis.")
            return self._find_intersections_from_contours(image, edges, mask)
        
        # Corner detection as a backup/supplement
        corner_points = self._detect_corners(gray, mask)
        
        # Extend lines and find intersections
        intersections = []
        
        if lines is not None and len(lines) > 0:
            self.logger.info(f"Found {len(lines)} lines. Finding intersections...")
            extended_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Skip too short lines
                if np.sqrt((x2 - x1)**2 + (y2 - y1)**2) < min_line_length / 2:
                    continue
                
                # Calculate line equation: ax + by + c = 0
                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 - slope * x1
                    a, b, c = slope, -1, intercept
                else:
                    # Vertical line
                    a, b, c = 1, 0, -x1
                
                extended_lines.append((a, b, c))
            
            # Find intersections between all pairs of lines
            for i in range(len(extended_lines)):
                for j in range(i + 1, len(extended_lines)):
                    a1, b1, c1 = extended_lines[i]
                    a2, b2, c2 = extended_lines[j]
                    
                    # Check if lines are parallel
                    det = a1 * b2 - a2 * b1
                    if abs(det) < 1e-8:
                        continue
                    
                    # Find intersection point
                    x = (b1 * c2 - b2 * c1) / det
                    y = (a2 * c1 - a1 * c2) / det
                    
                    # Only add if within image bounds
                    if 0 <= x < width and 0 <= y < height:
                        intersections.append((int(x), int(y)))
        
        # Combine with corner points
        intersections.extend(corner_points)
        
        # Cluster nearby intersection points to avoid duplicates
        if intersections:
            intersections = self._cluster_points(intersections, distance_threshold=10)
            self.logger.info(f"Found {len(intersections)} intersections after clustering")
        else:
            self.logger.warning("No intersections found")
        
        return intersections
    
    def _detect_corners(self, gray_image: np.ndarray, mask: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
        """
        Detect corners using Harris corner detector.
        
        Args:
            gray_image: Grayscale image
            mask: Optional mask to restrict corner detection
            
        Returns:
            List of (x, y) corner points
        """
        # Harris corner parameters
        block_size = 2
        ksize = 3
        k = 0.04
        
        # Apply Harris corner detector
        dst = cv2.cornerHarris(gray_image, block_size, ksize, k)
        
        # Apply mask if provided
        if mask is not None:
            dst = cv2.bitwise_and(dst, dst, mask=mask)
        
        # Threshold and convert to coordinates
        # Apply threshold for corner response
        threshold = 0.01 * dst.max()
        corner_mask = dst > threshold
        
        # Find coordinates of corners
        corner_points = []
        y_coords, x_coords = np.where(corner_mask)
        
        for x, y in zip(x_coords, y_coords):
            corner_points.append((int(x), int(y)))
        
        # Limit the number of corners to prevent too many false positives
        if len(corner_points) > 100:
            # Sort by corner response
            corner_values = [dst[y, x] for x, y in corner_points]
            sorted_indices = np.argsort(corner_values)[::-1]  # Descending order
            corner_points = [corner_points[i] for i in sorted_indices[:100]]
        
        self.logger.info(f"Detected {len(corner_points)} corners")
        return corner_points
    
    def _find_intersections_from_contours(self, image: np.ndarray, edges: np.ndarray, mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find intersections using contour analysis when line detection fails.
        
        Args:
            image: Original image
            edges: Edge detected image
            mask: ROI mask
            
        Returns:
            List of (x, y) intersection points
        """
        # Apply morphological operations to enhance line structure
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        
        # Skeletonize the lines
        skeleton = self._skeletonize(eroded)
        
        # Find endpoints and junctions in the skeleton
        # (these are potential intersection points)
        height, width = skeleton.shape[:2]
        intersections = []
        
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if skeleton[y, x] == 0:
                    continue
                
                # 3x3 neighborhood
                neighbors = skeleton[y-1:y+2, x-1:x+2].copy()
                neighbors[1, 1] = 0  # Remove center point
                
                # Count neighbors
                neighbor_count = np.sum(neighbors > 0)
                
                # Points with 3+ neighbors are potential junctions
                if neighbor_count >= 3:
                    intersections.append((x, y))
        
        self.logger.info(f"Found {len(intersections)} potential intersection points from skeleton analysis")
        
        # Cluster nearby points
        if intersections:
            intersections = self._cluster_points(intersections, distance_threshold=15)
        
        return intersections
    
    def _skeletonize(self, img: np.ndarray) -> np.ndarray:
        """
        Skeletonize a binary image.
        
        Args:
            img: Binary image to skeletonize
            
        Returns:
            Skeletonized binary image
        """
        # Ensure binary image
        img = img.copy()
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # Convert to correct format
        img = img // 255
        
        # Distance transform approach
        dist_transform = cv2.distanceTransform(img, cv2.DIST_L2, 5)
        _, skeleton = cv2.threshold(dist_transform, 0.7, 1, cv2.THRESH_BINARY)
        
        # Convert back to uint8
        skeleton = np.uint8(skeleton * 255)
        
        # Thin the skeleton
        kernel = np.ones((3, 3), np.uint8)
        skeleton = cv2.erode(skeleton, kernel, iterations=1)
        
        return skeleton
    
    def _cluster_points(self, points: List[Tuple[int, int]], distance_threshold: int = 10) -> List[Tuple[int, int]]:
        """
        Cluster nearby points to remove duplicates.
        
        Args:
            points: List of points to cluster
            distance_threshold: Maximum distance between points in the same cluster
            
        Returns:
            List of clustered points (one per cluster)
        """
        if not points:
            return []
        
        # Convert to numpy for faster operations
        points_array = np.array(points)
        
        # Track which points have been assigned to clusters
        assigned = np.zeros(len(points), dtype=bool)
        clusters = []
        
        for i in range(len(points)):
            if assigned[i]:
                continue
                
            # Start a new cluster
            cluster_indices = [i]
            assigned[i] = True
            
            # Find all points within threshold distance
            for j in range(i + 1, len(points)):
                if assigned[j]:
                    continue
                    
                # Calculate Euclidean distance
                dist = np.sqrt(np.sum((points_array[i] - points_array[j]) ** 2))
                if dist <= distance_threshold:
                    cluster_indices.append(j)
                    assigned[j] = True
            
            # Calculate cluster center (average)
            cluster_points = points_array[cluster_indices]
            center_x = int(np.mean(cluster_points[:, 0]))
            center_y = int(np.mean(cluster_points[:, 1]))
            
            clusters.append((center_x, center_y))
        
        return clusters
    
    @staticmethod
    def match_raw_to_fine(raw_pts: List[Tuple[int, int]], 
                         fine_pts: List[Tuple[float, float]], 
                         max_distance: float = 20.0) -> List[Tuple[float, float]]:
        """
        Match raw points to fine-tuned intersection points.
        
        Args:
            raw_pts (List[Tuple[int, int]]): Original raw points selected by user
            fine_pts (List[Tuple[float, float]]): Fine-tuned intersection points
            max_distance (float): Maximum allowed distance for matching
            
        Returns:
            List[Tuple[float, float]]: List of fine-tuned points matching the raw points
        """
        if not raw_pts or not fine_pts:
            return []
            
        matched_points = []
        
        for raw_pt in raw_pts:
            rx, ry = raw_pt
            
            # Find closest fine point
            closest_pt = None
            min_dist = float('inf')
            
            for fine_pt in fine_pts:
                fx, fy = fine_pt
                dist = np.sqrt((rx - fx)**2 + (ry - fy)**2)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_pt = fine_pt
            
            # Add the closest point if within max distance
            if min_dist <= max_distance and closest_pt is not None:
                matched_points.append(closest_pt)
            else:
                # If no matching fine point, use the raw point as fallback
                matched_points.append((float(rx), float(ry)))
                
        return matched_points 