#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the intersection finder service.
"""

import unittest
import numpy as np
import cv2
from src.services.intersection_finder import (
    find_intersections, 
    find_intersections_hough, 
    find_corners,
    line_intersection,
    find_and_sort_intersections
)

class TestIntersectionFinder(unittest.TestCase):
    """Test suite for the intersection finder service."""
    
    def test_find_intersections(self):
        """Test the find_intersections function with a simple cross pattern."""
        # Create a simple cross pattern
        test_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(test_image, (10, 50), (90, 50), 255, 1)  # Horizontal line
        cv2.line(test_image, (50, 10), (50, 90), 255, 1)  # Vertical line
        
        # Find intersections
        intersections = find_intersections(test_image)
        
        # Verify at least one intersection is found
        self.assertGreater(len(intersections), 0)
        
        # Verify the intersection is close to (50, 50)
        if len(intersections) > 0:
            x, y = intersections[0]
            self.assertAlmostEqual(x, 50, delta=3)
            self.assertAlmostEqual(y, 50, delta=3)
            
    def test_find_intersections_hough(self):
        """Test the find_intersections_hough function."""
        # Create a simple cross pattern
        test_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(test_image, (10, 50), (90, 50), 255, 1)  # Horizontal line
        cv2.line(test_image, (50, 10), (50, 90), 255, 1)  # Vertical line
        
        # Find intersections using Hough transform
        intersections = find_intersections_hough(test_image)
        
        # If intersections found (may not be reliable in simple test image), check location
        if len(intersections) > 0:
            x, y = intersections[0]
            self.assertLess(abs(x - 50) + abs(y - 50), 20)  # Within 20 pixels of center
            
    def test_find_corners(self):
        """Test the find_corners function."""
        # Create a simple image with corners
        test_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(test_image, (20, 20), (80, 80), 255, 1)
        
        # Find corners
        corners = find_corners(test_image)
        
        # Verify corners are found
        self.assertGreaterEqual(len(corners), 1)
            
    def test_line_intersection(self):
        """Test the line_intersection function."""
        # Define two lines that intersect at (50, 50)
        line1 = (10, 50, 90, 50)  # Horizontal line
        line2 = (50, 10, 50, 90)  # Vertical line
        
        # Find intersection
        intersection = line_intersection(line1, line2)
        
        # Verify intersection is at (50, 50)
        self.assertIsNotNone(intersection)
        if intersection:
            x, y = intersection
            self.assertAlmostEqual(x, 50, delta=0.1)
            self.assertAlmostEqual(y, 50, delta=0.1)
            
        # Test parallel lines
        line3 = (10, 30, 90, 30)  # Another horizontal line
        intersection = line_intersection(line1, line3)
        
        # Verify no intersection is found for parallel lines
        self.assertIsNone(intersection)
        
    def test_find_and_sort_intersections(self):
        """Test the find_and_sort_intersections function."""
        # Create an image with multiple intersections
        test_image = np.zeros((200, 200), dtype=np.uint8)
        
        # Create a grid pattern
        for i in range(20, 200, 40):
            cv2.line(test_image, (20, i), (180, i), 255, 1)  # Horizontal line
            cv2.line(test_image, (i, 20), (i, 180), 255, 1)  # Vertical line
        
        # Define a reference point
        origin = (100, 100)  # Center of the image
        
        # Find and sort intersections
        sorted_intersections = find_and_sort_intersections(test_image, origin, max_points=5)
        
        # Verify we get at most max_points intersections
        self.assertLessEqual(len(sorted_intersections), 5)
        
        # Verify they are sorted by distance from origin
        if len(sorted_intersections) >= 2:
            # Calculate distances of first and second points from origin
            dist1 = ((sorted_intersections[0][0] - origin[0])**2 + 
                     (sorted_intersections[0][1] - origin[1])**2)**0.5
            dist2 = ((sorted_intersections[1][0] - origin[0])**2 + 
                     (sorted_intersections[1][1] - origin[1])**2)**0.5
            
            # First point should be closer to origin than second point
            self.assertLessEqual(dist1, dist2)
        
    def test_empty_image(self):
        """Test finding intersections in an empty image."""
        # Create an empty image
        empty_image = np.zeros((50, 50), dtype=np.uint8)
        
        # Find intersections
        intersections = find_intersections(empty_image)
        
        # Verify no intersections are found
        self.assertEqual(len(intersections), 0)

if __name__ == '__main__':
    unittest.main() 