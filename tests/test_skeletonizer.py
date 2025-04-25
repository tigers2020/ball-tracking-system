#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the skeletonizer service.
"""

import unittest
import numpy as np
import cv2
from src.services.skeletonizer import skeletonize_image, skeletonize_roi

class TestSkeletonizer(unittest.TestCase):
    """Test suite for the skeletonizer service."""
    
    def test_skeletonize_image(self):
        """Test the basic skeletonize_image function."""
        # Create a simple test image (a rectangle)
        test_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(test_image, (20, 20), (80, 80), 255, -1)
        
        # Skeletonize the image
        skeleton = skeletonize_image(test_image)
        
        # Verify the skeleton is not empty
        self.assertTrue(np.any(skeleton > 0))
        
        # Verify the skeleton has fewer non-zero pixels than the original
        self.assertLess(np.count_nonzero(skeleton), np.count_nonzero(test_image))
        
    def test_skeletonize_roi(self):
        """Test the skeletonize_roi function with preprocessing."""
        # Create a test ROI with a cross pattern
        test_roi = np.zeros((50, 50), dtype=np.uint8)
        cv2.line(test_roi, (10, 25), (40, 25), 255, 3)  # Horizontal line
        cv2.line(test_roi, (25, 10), (25, 40), 255, 3)  # Vertical line
        
        # Add some noise
        noise = np.random.randint(0, 20, test_roi.shape, dtype=np.uint8)
        test_roi = cv2.add(test_roi, noise)
        
        # Skeletonize the ROI
        skeleton = skeletonize_roi(test_roi)
        
        # Verify the skeleton is not empty
        self.assertTrue(np.any(skeleton > 0))
        
        # Verify the skeleton has fewer non-zero pixels than the original
        self.assertLess(np.count_nonzero(skeleton), np.count_nonzero(test_roi))
        
    def test_skeletonize_empty_image(self):
        """Test skeletonizing an empty image."""
        # Create an empty image
        empty_image = np.zeros((50, 50), dtype=np.uint8)
        
        # Skeletonize the empty image
        skeleton = skeletonize_image(empty_image)
        
        # Verify the skeleton is also empty
        self.assertEqual(np.count_nonzero(skeleton), 0)
        
    def test_skeletonize_image_robustness(self):
        """Test skeletonize_image robustness with various input types."""
        # Test with float image
        float_image = np.zeros((50, 50), dtype=np.float32)
        cv2.circle(float_image, (25, 25), 15, 1.0, -1)
        
        skeleton = skeletonize_image(float_image)
        self.assertIsNotNone(skeleton)
        
        # Test with 3-channel image (should convert to grayscale internally)
        rgb_image = np.zeros((50, 50, 3), dtype=np.uint8)
        cv2.circle(rgb_image, (25, 25), 15, (255, 255, 255), -1)
        
        skeleton = skeletonize_roi(rgb_image)
        self.assertIsNotNone(skeleton)

if __name__ == '__main__':
    unittest.main() 