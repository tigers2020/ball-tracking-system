#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Intersection Finder service.
This module contains functions for finding intersections in skeletonized images.
"""

import logging
import numpy as np
import cv2
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

def find_intersections(skeleton: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find intersections in a skeletonized image using kernel convolution.
    
    Args:
        skeleton (np.ndarray): Skeletonized binary image
        
    Returns:
        List[Tuple[int, int]]: List of (x, y) coordinates of intersection points
    """
    try:
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
        
        # Threshold to find intersection points (value > 12 means at least 3 neighbors)
        intersections = np.where(convolved >= 13) 
        
        # Convert to list of (x, y) tuples
        intersection_points = [(int(x), int(y)) for y, x in zip(intersections[0], intersections[1])]
        
        logger.debug(f"Found {len(intersection_points)} intersection points using kernel method")
        
        # If no intersections found with kernel method, try Hough transform
        if not intersection_points:
            intersection_points = find_intersections_hough(skeleton)
            
        return intersection_points
        
    except Exception as e:
        logger.error(f"Error finding intersections: {e}")
        return []

def find_intersections_hough(skeleton: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find intersections using Hough transform line detection.
    
    Args:
        skeleton (np.ndarray): Skeletonized binary image
        
    Returns:
        List[Tuple[int, int]]: List of (x, y) coordinates of intersection points
    """
    try:
        # Ensure image is binary and in uint8 format
        binary = (skeleton > 0).astype(np.uint8) * 255
        
        # Apply Hough Transform to detect lines
        lines = cv2.HoughLinesP(
            binary, 
            rho=1, 
            theta=np.pi/180, 
            threshold=20,
            minLineLength=20, 
            maxLineGap=10
        )
        
        intersections = []
        
        # If lines are found
        if lines is not None and len(lines) >= 2:
            for i in range(len(lines)):
                for j in range(i + 1, len(lines)):
                    x1, y1, x2, y2 = lines[i][0]
                    x3, y3, x4, y4 = lines[j][0]
                    
                    # Find intersection between two lines
                    intersection = line_intersection((x1, y1, x2, y2), (x3, y3, x4, y4))
                    
                    if intersection:
                        x, y = intersection
                        # Check if the point is within image boundaries
                        height, width = skeleton.shape[:2]
                        if 0 <= x < width and 0 <= y < height:
                            intersections.append((int(x), int(y)))
            
            logger.debug(f"Found {len(intersections)} intersection points using Hough transform")
        else:
            # If Hough transform fails, try corner detection
            intersections = find_corners(skeleton)
            
        return intersections
        
    except Exception as e:
        logger.error(f"Error finding intersections with Hough transform: {e}")
        return []

def find_corners(image: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find corners in an image using Shi-Tomasi corner detection.
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        List[Tuple[int, int]]: List of (x, y) coordinates of corners
    """
    try:
        # Ensure image is in the right format
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        # Apply Shi-Tomasi corner detection
        corners = cv2.goodFeaturesToTrack(
            image,
            maxCorners=10,
            qualityLevel=0.1,
            minDistance=10
        )
        
        if corners is not None:
            # Convert corners to list of tuples
            corner_points = [(int(corner[0][0]), int(corner[0][1])) for corner in corners]
            logger.debug(f"Found {len(corner_points)} corners")
            return corner_points
        else:
            logger.warning("No corners found")
            return []
            
    except Exception as e:
        logger.error(f"Error finding corners: {e}")
        return []

def line_intersection(line1: Tuple[int, int, int, int], 
                     line2: Tuple[int, int, int, int]) -> Optional[Tuple[float, float]]:
    """
    Find the intersection point of two lines.
    
    Args:
        line1 (Tuple[int, int, int, int]): First line (x1, y1, x2, y2)
        line2 (Tuple[int, int, int, int]): Second line (x3, y3, x4, y4)
        
    Returns:
        Optional[Tuple[float, float]]: Intersection point (x, y) or None if no intersection
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    # Calculate determinants
    d = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    
    # Lines are parallel if d is zero
    if d == 0:
        return None
        
    # Calculate intersection point
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / d
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / d
    
    # Check if intersection lies on both line segments
    if 0 <= ua <= 1 and 0 <= ub <= 1:
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return (x, y)
    else:
        return None

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