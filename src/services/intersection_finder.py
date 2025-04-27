#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Intersection Finder service.
This module contains functions for finding intersections in skeletonized images.
"""

import logging
import numpy as np
import cv2
from typing import List, Tuple, Optional, Any, Dict
from collections import defaultdict

from src.utils.error_handling import handle_errors, ErrorAction

logger = logging.getLogger(__name__)

@handle_errors(
    action=ErrorAction.RETURN_DEFAULT,
    default_return=[],
    message="Error finding intersections: {error}"
)
def find_intersections(skeleton: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find intersections in a skeletonized image using kernel convolution.
    
    Args:
        skeleton (np.ndarray): Skeletonized binary image
        
    Returns:
        List[Tuple[int, int]]: List of (x, y) coordinates of intersection points
    """
    # Ensure image is valid
    if skeleton is None or skeleton.size == 0:
        logger.error("Invalid skeleton: None or empty")
        return []
        
    # Ensure image is binary
    if skeleton.dtype != np.uint8:
        skeleton = skeleton.astype(np.uint8)
    
    # Normalize to 0 and 1
    skeleton = skeleton / 255 if np.max(skeleton) > 1 else skeleton
    
    # Create kernel for detecting intersections (3x3 neighborhood)
    kernel = np.array([
        [1, 1, 1],
        [1, 10, 1],
        [1, 1, 1]
    ], dtype=np.uint8)
    
    # Convolve the image with the kernel
    convolved = cv2.filter2D(skeleton, -1, kernel)
    
    # Find potential intersection points (value >= 13 means at least 3 neighbors)
    intersections = np.where(convolved >= 13)
    
    # Convert to list of (x, y) tuples
    intersection_points = [(int(x), int(y)) for y, x in zip(intersections[0], intersections[1])]
    
    # Merge nearby intersections (within 5 pixels)
    merged_points = _merge_nearby_points(intersection_points, max_distance=5)
    
    logger.debug(f"Found {len(merged_points)} intersection points after merging")
    
    # If still no intersections found, try Hough transform
    if not merged_points:
        merged_points = find_intersections_hough(skeleton)
        logger.debug(f"Used Hough method, found {len(merged_points)} intersections")
        
    # If still no intersections found, try harris corner detector
    if not merged_points:
        merged_points = find_intersections_harris(skeleton)
        logger.debug(f"Used Harris method, found {len(merged_points)} intersections")
        
    return merged_points

def _merge_nearby_points(points: List[Tuple[int, int]], max_distance: int = 5) -> List[Tuple[int, int]]:
    """
    Merge nearby intersection points to prevent duplicates.
    
    Args:
        points (List[Tuple[int, int]]): List of points to merge
        max_distance (int): Maximum distance for merging points
        
    Returns:
        List[Tuple[int, int]]: List of merged points
    """
    if not points:
        return []
        
    # Sort points to ensure consistent merging
    sorted_points = sorted(points)
    
    merged = []
    clusters = []
    
    # Group points into clusters based on distance
    for point in sorted_points:
        x, y = point
        
        # Check if point belongs to an existing cluster
        added = False
        for i, cluster in enumerate(clusters):
            for cluster_point in cluster:
                cx, cy = cluster_point
                dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                if dist <= max_distance:
                    clusters[i].append(point)
                    added = True
                    break
            if added:
                break
                
        # If not added to any cluster, create a new one
        if not added:
            clusters.append([point])
    
    # Calculate the centroid of each cluster
    for cluster in clusters:
        if len(cluster) == 1:
            merged.append(cluster[0])
        else:
            # Compute average position (centroid)
            avg_x = sum(p[0] for p in cluster) // len(cluster)
            avg_y = sum(p[1] for p in cluster) // len(cluster)
            merged.append((avg_x, avg_y))
    
    return merged

@handle_errors(
    action=ErrorAction.RETURN_DEFAULT,
    default_return=[],
    message="Error finding intersections with Hough transform: {error}"
)
def find_intersections_hough(skeleton: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find intersections using Hough transform line detection.
    
    Args:
        skeleton (np.ndarray): Skeletonized binary image
        
    Returns:
        List[Tuple[int, int]]: List of (x, y) coordinates of intersection points
    """
    # Ensure image is valid
    if skeleton is None or skeleton.size == 0:
        logger.error("Invalid skeleton for Hough: None or empty")
        return []
        
    # Ensure image is binary and in uint8 format
    binary = (skeleton > 0).astype(np.uint8) * 255
    
    # Apply Hough Transform to detect lines with different parameters
    # Try multiple parameter sets to maximize detection
    parameter_sets = [
        {'rho': 1, 'theta': np.pi/180, 'threshold': 20, 'minLineLength': 20, 'maxLineGap': 10},
        {'rho': 1, 'theta': np.pi/180, 'threshold': 15, 'minLineLength': 15, 'maxLineGap': 15},
        {'rho': 0.5, 'theta': np.pi/180, 'threshold': 10, 'minLineLength': 10, 'maxLineGap': 5}
    ]
    
    all_lines = []
    for params in parameter_sets:
        lines = cv2.HoughLinesP(
            binary, 
            rho=params['rho'], 
            theta=params['theta'], 
            threshold=params['threshold'],
            minLineLength=params['minLineLength'], 
            maxLineGap=params['maxLineGap']
        )
        
        if lines is not None:
            all_lines.extend(lines)
            
    if not all_lines:
        logger.debug("No lines found with HoughLinesP")
        return []
    
    intersections = []
    
    # Find intersections between all lines
    for i in range(len(all_lines)):
        for j in range(i + 1, len(all_lines)):
            x1, y1, x2, y2 = all_lines[i][0]
            x3, y3, x4, y4 = all_lines[j][0]
            
            # Find intersection between two lines
            intersection = line_intersection((x1, y1, x2, y2), (x3, y3, x4, y4))
            
            if intersection:
                x, y = intersection
                # Check if the point is within image boundaries
                height, width = skeleton.shape[:2]
                if 0 <= x < width and 0 <= y < height:
                    # Check if the point is close to actual skeleton pixels
                    if is_near_skeleton(skeleton, (int(x), int(y)), max_distance=3):
                        intersections.append((int(x), int(y)))
    
    # Merge nearby intersections
    merged_intersections = _merge_nearby_points(intersections)
    
    logger.debug(f"Found {len(merged_intersections)} intersection points using Hough transform")
    return merged_intersections

@handle_errors(
    action=ErrorAction.RETURN_DEFAULT,
    default_return=[],
    message="Error finding intersections with Harris corner detector: {error}"
)
def find_intersections_harris(skeleton: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find intersections using Harris corner detection.
    
    Args:
        skeleton (np.ndarray): Skeletonized binary image
        
    Returns:
        List[Tuple[int, int]]: List of (x, y) coordinates of intersection points
    """
    # Ensure image is valid
    if skeleton is None or skeleton.size == 0:
        logger.error("Invalid skeleton for Harris: None or empty")
        return []
        
    # Ensure image is binary and in uint8 format
    binary = (skeleton > 0).astype(np.uint8) * 255
    
    # Detect corners using Harris corner detector
    corners = cv2.cornerHarris(binary, blockSize=3, ksize=3, k=0.04)
    
    # Normalizing
    cv2.normalize(corners, corners, 0, 255, cv2.NORM_MINMAX)
    
    # Thresholding
    threshold = 0.01 * corners.max()
    corner_points = np.where(corners > threshold)
    
    # Get corner coordinates
    intersections = [(int(x), int(y)) for y, x in zip(corner_points[0], corner_points[1])]
    
    # Filter to keep only points that are on the skeleton
    filtered_intersections = []
    for point in intersections:
        if is_near_skeleton(skeleton, point, max_distance=2):
            filtered_intersections.append(point)
    
    # Merge nearby intersections
    merged_intersections = _merge_nearby_points(filtered_intersections)
    
    logger.debug(f"Found {len(merged_intersections)} intersection points using Harris corner detector")
    return merged_intersections

def find_corners(skeleton: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find corners in a skeletonized image using Shi-Tomasi corner detector.
    
    Args:
        skeleton (np.ndarray): Skeletonized binary image
        
    Returns:
        List[Tuple[int, int]]: List of (x, y) coordinates of corner points
    """
    try:
        # Ensure binary image
        binary = (skeleton > 0).astype(np.uint8) * 255
        
        # Use Shi-Tomasi corner detector
        corners = cv2.goodFeaturesToTrack(
            binary,
            maxCorners=25,
            qualityLevel=0.01,
            minDistance=10
        )
        
        if corners is None:
            logger.debug("No corners found using Shi-Tomasi detector")
            return []
            
        # Convert to list of tuples
        corner_points = [(int(corner[0][0]), int(corner[0][1])) for corner in corners]
        
        # Filter to keep only points that are on the skeleton
        filtered_corners = []
        for point in corner_points:
            if is_near_skeleton(skeleton, point, max_distance=2):
                filtered_corners.append(point)
        
        logger.debug(f"Found {len(filtered_corners)} corner points using Shi-Tomasi detector")
        return filtered_corners
        
    except Exception as e:
        logger.error(f"Error finding corners: {e}")
        return []

def line_intersection(line1: Tuple[float, float, float, float], 
                     line2: Tuple[float, float, float, float]) -> Optional[Tuple[float, float]]:
    """
    Calculate the intersection point of two lines.
    
    Args:
        line1 (Tuple[float, float, float, float]): First line as (x1, y1, x2, y2)
        line2 (Tuple[float, float, float, float]): Second line as (x3, y3, x4, y4)
        
    Returns:
        Optional[Tuple[float, float]]: Intersection point (x, y) or None if no intersection
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    # Calculate slopes
    try:
        # Convert to homogeneous coordinates
        hx1, hy1, hz1 = x1, y1, 1
        hx2, hy2, hz2 = x2, y2, 1
        hx3, hy3, hz3 = x3, y3, 1
        hx4, hy4, hz4 = x4, y4, 1
        
        # Get lines in homogeneous form (a, b, c) where ax + by + c = 0
        line1 = np.cross([hx1, hy1, hz1], [hx2, hy2, hz2])
        line2 = np.cross([hx3, hy3, hz3], [hx4, hy4, hz4])
        
        # Calculate intersection
        intersection = np.cross(line1, line2)
        
        # Convert back to Cartesian coordinates
        if abs(intersection[2]) < 1e-8:  # Check if lines are parallel
            return None
            
        ix = intersection[0] / intersection[2]
        iy = intersection[1] / intersection[2]
        
        # Check if intersection is within the line segments
        def is_between(p, p1, p2):
            return min(p1, p2) - 1 <= p <= max(p1, p2) + 1
            
        if (is_between(ix, x1, x2) and is_between(iy, y1, y2) and 
            is_between(ix, x3, x4) and is_between(iy, y3, y4)):
            return (ix, iy)
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error calculating line intersection: {e}")
        return None

def is_near_skeleton(skeleton: np.ndarray, point: Tuple[int, int], max_distance: int = 2) -> bool:
    """
    Check if a point is near a skeleton pixel.
    
    Args:
        skeleton (np.ndarray): Skeletonized binary image
        point (Tuple[int, int]): Point to check (x, y)
        max_distance (int): Maximum allowed distance to skeleton
        
    Returns:
        bool: True if the point is near the skeleton, False otherwise
    """
    x, y = point
    height, width = skeleton.shape[:2]
    
    # Check if point is within image boundaries
    if x < 0 or x >= width or y < 0 or y >= height:
        return False
        
    # If directly on skeleton
    if skeleton[y, x] > 0:
        return True
        
    # Check neighborhood
    start_y = max(0, y - max_distance)
    end_y = min(height, y + max_distance + 1)
    start_x = max(0, x - max_distance)
    end_x = min(width, x + max_distance + 1)
    
    neighborhood = skeleton[start_y:end_y, start_x:end_x]
    
    return np.any(neighborhood > 0)

def find_and_sort_intersections(skeleton: np.ndarray,
                               origin: Tuple[float, float] = None,
                               max_points: int = 10) -> List[Tuple[int, int]]:
    """
    Find intersections and sort them by distance from origin or center of image.
    
    Args:
        skeleton (np.ndarray): Skeletonized binary image
        origin (Tuple[float, float], optional): Reference point for sorting (center of image if None)
        max_points (int): Maximum number of points to return
        
    Returns:
        List[Tuple[int, int]]: Sorted list of intersection points
    """
    # Find intersections
    intersections = find_intersections(skeleton)
    
    # If no intersections found, try other methods
    if not intersections:
        # Try finding corners as a fallback
        intersections = find_corners(skeleton)
        
        # If still no points, just return any skeleton pixel close to center
        if not intersections:
            # Find all skeleton points
            points = np.where(skeleton > 0)
            if len(points[0]) > 0:
                # Convert to (x, y) tuples
                skeleton_points = [(int(x), int(y)) for y, x in zip(points[0], points[1])]
                intersections = skeleton_points
    
    # If no origin provided, use center of image
    if not origin:
        height, width = skeleton.shape[:2]
        origin = (width / 2, height / 2)
    
    # Sort by distance from origin
    def distance(point):
        return ((point[0] - origin[0]) ** 2 + (point[1] - origin[1]) ** 2) ** 0.5
    
    sorted_intersections = sorted(intersections, key=distance)
    
    # Return the closest max_points intersections
    return sorted_intersections[:max_points] 