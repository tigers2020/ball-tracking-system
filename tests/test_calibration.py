#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calibration module tests.
"""

import os
import json
import tempfile
import unittest
from pathlib import Path

import pytest
from PySide6.QtCore import QPointF

from src.models.calibration_model import CalibrationModel
from src.utils.config_manager import ConfigManager
from src.controllers.calibration_controller import CalibrationController
from src.views.calibration_view import CalibrationView


class MockConfigManager:
    """Mock ConfigManager for testing."""
    
    def __init__(self):
        self.calibration_points = {}
        self.save_called = False
    
    def get_calibration_points(self):
        return self.calibration_points
    
    def set_calibration_points(self, points):
        self.calibration_points = points
    
    def save_config(self, force=False):
        self.save_called = True
        return True


class MockSignal:
    """Mock signal class with connect method."""
    
    def __init__(self):
        self.connected_slots = []
    
    def connect(self, slot):
        """Mock connect method."""
        self.connected_slots.append(slot)
    
    def emit(self, *args):
        """Mock emit method."""
        for slot in self.connected_slots:
            slot(*args)


class MockCalibrationView:
    """Mock version of CalibrationView for testing."""
    
    def __init__(self):
        """Initialize with mock signals."""
        self.controller = None
        # Mock signals
        self.point_added = MockSignal()
        self.point_moved = MockSignal()
        self.fine_tune_requested = MockSignal()
        self.save_calibration_requested = MockSignal()
        self.load_calibration_requested = MockSignal()
        self.clear_points_requested = MockSignal()
        self.load_images_requested = MockSignal()
        self.load_current_frame_requested = MockSignal()
    
    def set_left_image(self, pixmap):
        """Mock method."""
        pass
    
    def set_right_image(self, pixmap):
        """Mock method."""
        pass
    
    def clear_points(self):
        """Mock method."""
        pass
    
    def update_point(self, side, index, position, is_fine_tuned):
        """Mock method."""
        pass
    
    def _rebuild_points(self, side):
        """Mock method."""
        pass


class TestCalibrationModel(unittest.TestCase):
    """Test CalibrationModel functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.config_manager = MockConfigManager()
        self.model = CalibrationModel(self.config_manager)
        
    def test_add_point(self):
        """Test adding points."""
        # Add point to left side
        self.model.add_point("left", QPointF(100, 100))
        self.assertEqual(len(self.model.left_points), 1)
        self.assertEqual(self.model.left_points[0]['position'].x(), 100)
        self.assertEqual(self.model.left_points[0]['position'].y(), 100)
        self.assertFalse(self.model.left_points[0]['is_fine_tuned'])
        
        # Add point to right side
        self.model.add_point("right", QPointF(200, 200))
        self.assertEqual(len(self.model.right_points), 1)
        self.assertEqual(self.model.right_points[0]['position'].x(), 200)
        self.assertEqual(self.model.right_points[0]['position'].y(), 200)
        self.assertFalse(self.model.right_points[0]['is_fine_tuned'])
        
    def test_update_point(self):
        """Test updating points."""
        # Add points
        self.model.add_point("left", QPointF(100, 100))
        self.model.add_point("right", QPointF(200, 200))
        
        # Update points
        self.model.update_point("left", 0, QPointF(150, 150), True)
        self.model.update_point("right", 0, QPointF(250, 250))
        
        # Check updated values
        self.assertEqual(self.model.left_points[0]['position'].x(), 150)
        self.assertEqual(self.model.left_points[0]['position'].y(), 150)
        self.assertTrue(self.model.left_points[0]['is_fine_tuned'])
        
        self.assertEqual(self.model.right_points[0]['position'].x(), 250)
        self.assertEqual(self.model.right_points[0]['position'].y(), 250)
        self.assertFalse(self.model.right_points[0]['is_fine_tuned'])
        
    def test_clear_points(self):
        """Test clearing points."""
        # Add points
        self.model.add_point("left", QPointF(100, 100))
        self.model.add_point("right", QPointF(200, 200))
        
        # Clear points
        self.model.clear_points()
        
        # Check that points are cleared
        self.assertEqual(len(self.model.left_points), 0)
        self.assertEqual(len(self.model.right_points), 0)
        

class TestSaveLoadOverwrite(unittest.TestCase):
    """Test save/load functionality with overwrite capability."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary file for testing
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        
        # Create model with direct file I/O
        self.model = CalibrationModel()
        
        # Set image sizes
        self.model.set_image_sizes((800, 600), (800, 600))
        
    def tearDown(self):
        """Clean up after test."""
        # Remove temporary file
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_save_load_roundtrip(self):
        """Test saving and loading preserves point count and coordinates."""
        # Add several points
        num_points = 5
        for i in range(num_points):
            self.model.add_point("left", QPointF(100 + i*50, 100 + i*50))
            self.model.add_point("right", QPointF(500 + i*50, 100 + i*50))
        
        # Verify initial counts
        self.assertEqual(len(self.model.left_points), num_points)
        self.assertEqual(len(self.model.right_points), num_points)
        
        # Save to file
        self.assertTrue(self.model.save_to_json(self.temp_file.name))
        
        # Create a new model and load from file
        new_model = CalibrationModel()
        self.assertTrue(new_model.load_from_json(self.temp_file.name))
        
        # Verify counts and coordinates preserved
        self.assertEqual(len(new_model.left_points), num_points)
        self.assertEqual(len(new_model.right_points), num_points)
        
        # Check coordinates (allow small floating-point differences)
        for i in range(num_points):
            self.assertAlmostEqual(new_model.left_points[i]['position'].x(), 100 + i*50, places=1)
            self.assertAlmostEqual(new_model.left_points[i]['position'].y(), 100 + i*50, places=1)
            self.assertAlmostEqual(new_model.right_points[i]['position'].x(), 500 + i*50, places=1)
            self.assertAlmostEqual(new_model.right_points[i]['position'].y(), 100 + i*50, places=1)
    
    def test_overwrite_with_dict_format(self):
        """Test that overwriting points works correctly with dictionary format."""
        # Add initial points
        self.model.add_point("left", QPointF(100, 100))
        self.model.add_point("left", QPointF(200, 200))
        
        # Save to file
        self.assertTrue(self.model.save_to_json(self.temp_file.name))
        
        # Examine the JSON structure
        with open(self.temp_file.name, 'r') as f:
            data = json.load(f)
        
        # Verify dictionary format
        self.assertEqual(data.get('calib_ver'), 1.2)
        self.assertIsInstance(data.get('left'), dict)
        self.assertIn('p00', data.get('left'))
        self.assertIn('p01', data.get('left'))
        
        # Modify a point
        self.model.update_point("left", 0, QPointF(150, 150))
        
        # Save again
        self.assertTrue(self.model.save_to_json(self.temp_file.name))
        
        # Check the JSON structure again
        with open(self.temp_file.name, 'r') as f:
            data = json.load(f)
        
        # Verify point count hasn't changed
        self.assertEqual(len(data.get('left')), 2)
        
        # Load into a new model
        new_model = CalibrationModel()
        self.assertTrue(new_model.load_from_json(self.temp_file.name))
        
        # Verify point was overwritten
        self.assertEqual(len(new_model.left_points), 2)
        self.assertAlmostEqual(new_model.left_points[0]['position'].x(), 150, places=1)
        self.assertAlmostEqual(new_model.left_points[0]['position'].y(), 150, places=1)
    
    def test_controller_overwrite(self):
        """Test that controller handles point overwriting correctly."""
        # Create a separate test for controller overwrite without dependencies
        test_model = CalibrationModel()
        # Use a mock view to avoid Qt event loop issues
        test_view = MockCalibrationView()
        
        # Create controller with fresh model and mock view
        controller = CalibrationController(test_model, test_view)
        
        # Set max_points to a small value for testing
        controller.max_points = 2  # Only allow 2 points
        print("Test started. Max points:", controller.max_points)
        
        # Add points up to max
        controller.add_point("left", QPointF(100, 100))
        print("Added first point. Left points count:", len(test_model.left_points))
        controller.add_point("left", QPointF(200, 200))
        print("Added second point. Left points count:", len(test_model.left_points))
        
        # Verify count
        self.assertEqual(len(test_model.left_points), 2)
        print("Verified count equals 2")
        
        # Try to add a third point (should overwrite first point)
        test_point = QPointF(500, 500)
        print("About to add third point at", test_point.x(), test_point.y())
        controller.add_point("left", test_point)
        print("Added third point (should overwrite). Left points count:", len(test_model.left_points))
        
        # Verify count hasn't changed
        self.assertEqual(len(test_model.left_points), 2)
        print("Verified count still equals 2")
        
        # Check if one of the points matches the test point (meaning overwrite worked)
        overwritten = False
        for i, point in enumerate(test_model.left_points):
            pos = point['position']
            print(f"Point {i}: x={pos.x()}, y={pos.y()}")
            if abs(pos.x() - test_point.x()) < 0.1 and abs(pos.y() - test_point.y()) < 0.1:
                overwritten = True
                print(f"Found overwritten point at index {i}")
                break
        
        self.assertTrue(overwritten, "No point was overwritten")
        print("Test completed successfully")


if __name__ == '__main__':
    pytest.main(['-v']) 