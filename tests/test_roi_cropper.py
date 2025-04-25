#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the ROI cropper service.
"""

import unittest
import numpy as np
from src.services.roi_cropper import crop_roi, crop_roi_with_padding

class TestRoiCropper(unittest.TestCase):
    """Test suite for the ROI cropper service."""
    
    def test_crop_roi(self):
        """Test the basic crop_roi function."""
        # Create a test image
        test_image = np.zeros((200, 200), dtype=np.uint8)
        
        # Center point
        center = (100, 100)
        
        # Crop with different radii
        for radius in [10, 20, 30]:
            roi = crop_roi(test_image, center, radius)
            
            # Verify the ROI dimensions
            expected_size = max(int(radius * 2.5), 40)
            self.assertLessEqual(roi.shape[0], expected_size + 1)  # Allow for rounding
            self.assertLessEqual(roi.shape[1], expected_size + 1)
            
    def test_crop_roi_with_padding(self):
        """Test the crop_roi_with_padding function."""
        # Create a test image
        test_image = np.zeros((200, 200), dtype=np.uint8)
        
        # Test with a point near the edge
        center = (10, 10)
        radius = 20
        
        # Crop with padding
        roi, offset = crop_roi_with_padding(test_image, center, radius)
        
        # Verify the ROI is not None
        self.assertIsNotNone(roi)
        
        # Verify the ROI dimensions
        expected_size = max(int(radius * 2.5), 40)
        self.assertEqual(roi.shape[0], expected_size)
        self.assertEqual(roi.shape[1], expected_size)
        
        # Verify the offset is correct (should indicate top-left position)
        self.assertEqual(offset[0], 0 - (expected_size//2 - 10))  # 0 - (half_size - x)
        self.assertEqual(offset[1], 0 - (expected_size//2 - 10))  # 0 - (half_size - y)
        
    def test_crop_roi_center_point(self):
        """Test cropping ROI with point at the center of the image."""
        # Create a test image
        test_image = np.zeros((200, 200), dtype=np.uint8)
        
        # Center point
        center = (100, 100)
        
        # Crop with radius
        radius = 30
        roi = crop_roi(test_image, center, radius)
        
        # Expected size
        expected_size = max(int(radius * 2.5), 40)
        
        # Verify the ROI dimensions are close to expected size
        self.assertAlmostEqual(roi.shape[0], expected_size, delta=2)
        self.assertAlmostEqual(roi.shape[1], expected_size, delta=2)
        
    def test_crop_roi_edge_cases(self):
        """Test cropping ROI with edge cases."""
        # Create a test image
        test_image = np.zeros((200, 200), dtype=np.uint8)
        
        # Test with point outside the image (should adjust to boundary)
        center = (250, 150)
        radius = 20
        
        roi = crop_roi(test_image, center, radius)
        
        # Verify the ROI is not None (should adjust to boundary)
        self.assertIsNotNone(roi)
        
        # Test with empty image
        empty_image = np.array([])
        roi = crop_roi(empty_image, (10, 10), 10)
        
        # Should return None for invalid image
        self.assertIsNone(roi)
        
    def test_crop_roi_colored_image(self):
        """Test cropping ROI from a colored image."""
        # Create a colored test image
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Center point
        center = (100, 100)
        
        # Draw a colored circle
        radius = 30
        for y in range(test_image.shape[0]):
            for x in range(test_image.shape[1]):
                distance = ((x - center[0])**2 + (y - center[1])**2)**0.5
                if distance < radius:
                    test_image[y, x] = [255, 0, 0]  # Red circle
        
        # Crop with radius
        roi = crop_roi(test_image, center, radius)
        
        # Verify the ROI has 3 channels
        self.assertEqual(len(roi.shape), 3)
        self.assertEqual(roi.shape[2], 3)
        
        # Verify color is preserved
        self.assertTrue(np.any(roi[:, :, 0] > 0))  # Red channel

if __name__ == '__main__':
    unittest.main() 