#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for coord_utils module.
"""

import unittest
from src.utils.coord_utils import fuse_coordinates, get_2d_point_from_kalman


class TestCoordUtils(unittest.TestCase):
    """Test cases for coord_utils module."""
    
    def test_fuse_coordinates_empty(self):
        """Test fuse_coordinates with empty list."""
        result = fuse_coordinates([])
        self.assertIsNone(result)
    
    def test_fuse_coordinates_single(self):
        """Test fuse_coordinates with single coordinate."""
        coords = [(10.5, 20.5)]
        result = fuse_coordinates(coords)
        self.assertEqual(result, (10.5, 20.5))
    
    def test_fuse_coordinates_multiple(self):
        """Test fuse_coordinates with multiple coordinates."""
        coords = [(10.0, 20.0), (20.0, 30.0), (30.0, 40.0)]
        result = fuse_coordinates(coords)
        self.assertEqual(result, (20.0, 30.0))
    
    def test_get_2d_point_from_kalman(self):
        """Test get_2d_point_from_kalman."""
        kalman_state = (10.5, 20.5, 1.0, 2.0)
        result = get_2d_point_from_kalman(kalman_state)
        self.assertEqual(result, (10.5, 20.5))


if __name__ == "__main__":
    unittest.main() 