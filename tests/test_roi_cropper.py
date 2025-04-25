#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test module for ROI Cropper service.
"""

import unittest
import numpy as np

from src.services.roi_cropper import crop_roi, crop_roi_with_offset, adjust_point_to_roi, adjust_point_from_roi


class TestRoiCropper(unittest.TestCase):
    """Test cases for ROI cropper service."""

    def setUp(self):
        """Set up test cases."""
        # Create a test image (200x200 with a value gradient)
        self.test_image = np.zeros((200, 200), dtype=np.uint8)
        for i in range(200):
            for j in range(200):
                self.test_image[i, j] = (i + j) % 256

    def test_crop_roi_dimensions(self):
        """Test that the cropped ROI has the correct dimensions."""
        # Test with default radius (20.0)
        roi = crop_roi(self.test_image, (100, 100))
        expected_size = max(40, int(20.0 * 2.5))
        if expected_size % 2 != 0:
            expected_size += 1
            
        # Check if dimensions are as expected (or slightly smaller due to boundary constraints)
        self.assertEqual(roi.shape[0], expected_size)
        self.assertEqual(roi.shape[1], expected_size)
        
        # Test with a larger radius
        roi = crop_roi(self.test_image, (100, 100), 30.0)
        expected_size = max(40, int(30.0 * 2.5))
        if expected_size % 2 != 0:
            expected_size += 1
            
        self.assertEqual(roi.shape[0], expected_size)
        self.assertEqual(roi.shape[1], expected_size)

    def test_crop_roi_center(self):
        """Test that the center of the ROI corresponds to the input center."""
        center = (100, 100)
        roi = crop_roi(self.test_image, center)
        
        # The center of the ROI should have the same value as the center in the original image
        roi_center_y, roi_center_x = roi.shape[0] // 2, roi.shape[1] // 2
        original_value = self.test_image[center[1], center[0]]
        roi_value = roi[roi_center_y, roi_center_x]
        
        self.assertEqual(roi_value, original_value)

    def test_crop_roi_near_boundary(self):
        """Test cropping near the image boundary."""
        # Test near top-left corner
        roi = crop_roi(self.test_image, (10, 10))
        self.assertGreaterEqual(roi.shape[0], 10)
        self.assertGreaterEqual(roi.shape[1], 10)
        
        # Test near bottom-right corner
        roi = crop_roi(self.test_image, (190, 190))
        self.assertGreaterEqual(roi.shape[0], 10)
        self.assertGreaterEqual(roi.shape[1], 10)

    def test_crop_roi_with_offset(self):
        """Test cropping with offset information."""
        center = (100, 100)
        roi, offset = crop_roi_with_offset(self.test_image, center)
        
        # Check that the offset is correct
        expected_size = max(40, int(20.0 * 2.5))
        if expected_size % 2 != 0:
            expected_size += 1
        half_size = expected_size // 2
        
        self.assertEqual(offset, (center[0] - half_size, center[1] - half_size))
        
        # Check ROI dimensions
        self.assertEqual(roi.shape[0], expected_size)
        self.assertEqual(roi.shape[1], expected_size)

    def test_adjust_point_to_roi(self):
        """Test adjusting a point to ROI coordinates."""
        # Create a point in the original image
        point = (120, 130)
        
        # Define an ROI offset
        roi_offset = (100, 100)
        
        # Adjust the point to ROI coordinates
        roi_point = adjust_point_to_roi(point, roi_offset)
        
        # Check that the point was correctly adjusted
        self.assertEqual(roi_point, (20, 30))

    def test_adjust_point_from_roi(self):
        """Test adjusting a point from ROI back to original coordinates."""
        # Create a point in ROI coordinates
        roi_point = (20, 30)
        
        # Define an ROI offset
        roi_offset = (100, 100)
        
        # Adjust the point back to original coordinates
        point = adjust_point_from_roi(roi_point, roi_offset)
        
        # Check that the point was correctly adjusted
        self.assertEqual(point, (120, 130))

    def test_invalid_input(self):
        """Test handling of invalid inputs."""
        # Test with None image
        with self.assertRaises(ValueError):
            crop_roi(None, (100, 100))
        
        # Test with empty image
        with self.assertRaises(ValueError):
            crop_roi(np.array([]), (100, 100))

    def test_roi_too_small(self):
        """Test handling of ROIs that would be too small."""
        # Create a very small image
        small_image = np.zeros((5, 5), dtype=np.uint8)
        
        # Try to crop an ROI that would be smaller than the minimum allowed
        with self.assertRaises(ValueError):
            crop_roi(small_image, (2, 2))


if __name__ == '__main__':
    unittest.main() 