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
    
    @staticmethod
    def find_intersections(skel_img: np.ndarray, 
                          min_line_length: int = 8, 
                          max_line_gap: int = 2) -> List[Tuple[float, float]]:
        """
        Find intersections in a skeletonized image.
        
        Args:
            skel_img (np.ndarray): Skeletonized binary image
            min_line_length (int, optional): Minimum line length for HoughLinesP
            max_line_gap (int, optional): Maximum allowed gap between line segments
            
        Returns:
            List[Tuple[float, float]]: List of intersection points (x, y)
        """
        if skel_img is None or skel_img.size == 0:
            logging.error("Invalid image provided for intersection finding")
            return []
            
        # Find lines using Probabilistic Hough Transform
        lines = cv2.HoughLinesP(
            skel_img, 
            rho=1, 
            theta=np.pi/180, 
            threshold=20, 
            minLineLength=min_line_length, 
            maxLineGap=max_line_gap
        )
        
        if lines is None or len(lines) < 2:
            logging.warning("Insufficient lines detected for intersection finding")
            return []
            
        # Find intersections between all line pairs
        intersections = []
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                x1, y1, x2, y2 = lines[i][0]
                x3, y3, x4, y4 = lines[j][0]
                
                point = IntersectionFinder._line_intersection(
                    (x1, y1), (x2, y2), (x3, y3), (x4, y4)
                )
                
                if point is not None:
                    intersections.append(point)
        
        # Cluster nearby intersections if we have multiple detections
        clustered_points = IntersectionFinder._cluster_points(intersections)
        
        return clustered_points
    
    @staticmethod
    def _line_intersection(p1: Tuple[int, int], 
                         p2: Tuple[int, int], 
                         p3: Tuple[int, int], 
                         p4: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        """
        Find the intersection point of two lines defined by two points each.
        
        Args:
            p1, p2: Points defining the first line
            p3, p4: Points defining the second line
            
        Returns:
            Optional[Tuple[float, float]]: Intersection point (x, y) or None if lines are parallel
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        # Calculate the denominator
        denominator = ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
        
        # Lines are parallel
        if denominator == 0:
            return None
            
        # Calculate intersection point    
        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator
        
        # Check if intersection is on both line segments
        if 0 <= ua <= 1 and 0 <= ub <= 1:
            x = x1 + ua * (x2 - x1)
            y = y1 + ua * (y2 - y1)
            return (x, y)
            
        return None
    
    @staticmethod
    def _cluster_points(points: List[Tuple[float, float]], 
                      threshold: float = 5.0) -> List[Tuple[float, float]]:
        """
        Cluster nearby intersection points to eliminate duplicates.
        When multiple nearby intersection points exist, replace them with their centroid.
        
        Args:
            points (List[Tuple[float, float]]): List of intersection points
            threshold (float, optional): Distance threshold for clustering
            
        Returns:
            List[Tuple[float, float]]: List of clustered/deduplicated intersection points
        """
        if not points:
            return []
            
        if len(points) == 1:
            return points
            
        # Convert to numpy array
        points_array = np.array(points)
        
        # Cluster intersections using K-means
        # Determine number of clusters based on distances
        from scipy.cluster.hierarchy import fclusterdata
        try:
            # Use hierarchical clustering to find clusters
            cluster_indices = fclusterdata(points_array, threshold, criterion='distance')
            
            # Group points by cluster
            clusters = {}
            for i, cluster_idx in enumerate(cluster_indices):
                if cluster_idx not in clusters:
                    clusters[cluster_idx] = []
                clusters[cluster_idx].append(points_array[i])
                
            # Calculate centroid for each cluster
            centroids = []
            for cluster in clusters.values():
                cluster_array = np.array(cluster)
                centroid = np.mean(cluster_array, axis=0)
                centroids.append((centroid[0], centroid[1]))
                
            return centroids
            
        except Exception as e:
            logging.warning(f"Clustering failed, falling back to k-means: {e}")
            
            # Fallback to k-means if scipy fails
            # Start with estimating number of clusters
            n_clusters = min(len(points), 10)  # Limit to reasonable number
            
            # Apply k-means
            kmeans = KMeans(n_clusters=n_clusters, n_init=10)
            kmeans.fit(points_array)
            
            # Get centroids
            centroids = kmeans.cluster_centers_
            
            return [(float(x), float(y)) for x, y in centroids]
    
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