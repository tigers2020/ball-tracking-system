#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test module for CalibrationController.
"""

import pytest
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import unittest

from PySide6.QtCore import QPointF, Qt, QObject
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFileDialog, QMessageBox

from src.models.calibration_model import CalibrationModel
from src.views.calibration_tab import CalibrationTab
from src.controllers.calibration_controller import CalibrationController
from src.utils.config_manager import ConfigManager


class FakeSignal:
    """
    Fake signal class for testing.
    """
    def __init__(self):
        self.callbacks = []
    
    def connect(self, callback):
        self.callbacks.append(callback)
        
    def emit(self, *args):
        for callback in self.callbacks:
            callback(*args)


class FakeView(QObject):
    """
    Fake view class for testing the controller without a real QWidget.
    """
    def __init__(self):
        super().__init__()
        self.point_added = FakeSignal()
        self.point_moved = FakeSignal()
        self.clear_button = MagicMock()
        self.fine_tune_button = MagicMock()
        self.save_button = MagicMock()
        self.load_button = MagicMock()
        self.load_current_frame_button = MagicMock()
        
        self.added_items = []
        self.updated_items = []
        self.grid_lines = {}
        self.left_image = None
        self.right_image = None
        
    def add_point_item(self, side, x, y, index):
        self.added_items.append((side, x, y, index))
        
    def update_point_item(self, side, index, x, y):
        self.updated_items.append((side, index, x, y))
        
    def clear_points(self):
        self.added_items = []
        self.updated_items = []
        self.grid_lines = {}
        
    def draw_grid_lines(self, side, points, rows, cols):
        self.grid_lines[side] = (points, rows, cols)
    
    def set_images(self, left_image, right_image):
        self.left_image = left_image
        self.right_image = right_image


@pytest.fixture
def setup_controller():
    """Set up the controller with fake models and views for testing."""
    model = CalibrationModel()
    view = FakeView()
    config_manager = MagicMock(spec=ConfigManager)
    controller = CalibrationController(model, view, config_manager)
    return model, view, controller


@pytest.fixture
def temp_calibration_file():
    """Create a temporary calibration file for testing."""
    # Create a temporary file and directory
    temp_dir = tempfile.mkdtemp()
    temp_file = Path(temp_dir) / "test_calibration.json"
    
    # Create sample calibration data
    calibration_data = {
        "points": {
            "left": [[100, 100], [200, 100], [100, 200], [200, 200]],
            "right": [[300, 300], [400, 300], [300, 400], [400, 400]]
        }
    }
    
    # Write the data to the file
    with open(temp_file, 'w') as f:
        json.dump(calibration_data, f)
    
    yield temp_file
    
    # Clean up
    if os.path.exists(temp_file):
        os.remove(temp_file)
    os.rmdir(temp_dir)


def test_init(setup_controller):
    """Test initialization of the controller."""
    model, view, controller = setup_controller
    
    # Check that model and view are set
    assert controller.model == model
    assert controller.view == view
    
    # Check that signals are connected
    assert len(view.point_added.callbacks) == 1
    assert len(view.point_moved.callbacks) == 1
    
    # Check that buttons are connected
    view.clear_button.clicked.connect.assert_called_once()
    view.fine_tune_button.clicked.connect.assert_called_once()
    view.save_button.clicked.connect.assert_called_once()
    view.load_button.clicked.connect.assert_called_once()


def test_on_add_point(setup_controller):
    """Test adding a point via the controller."""
    model, view, controller = setup_controller
    
    # Emit the point_added signal
    view.point_added.emit('left', 100, 200)
    
    # Check that the point was added to the model
    assert model.get_points('left') == [(100, 200)]
    
    # Check that the view was updated
    assert view.added_items == [('left', 100, 200, 0)]


def test_on_move_point(setup_controller):
    """Test moving a point via the controller."""
    model, view, controller = setup_controller
    
    # Add a point first
    model.add_point('left', (100, 200))
    view.added_items = [('left', 100, 200, 0)]  # Simulate the view state
    
    # Emit the point_moved signal
    view.point_moved.emit('left', 0, 150, 250)
    
    # Check that the point was updated in the model
    assert model.get_points('left') == [(150, 250)]


def test_on_clear_points(setup_controller):
    """Test clearing points via the controller."""
    model, view, controller = setup_controller
    
    # Add some points first
    model.add_point('left', (100, 200))
    model.add_point('right', (300, 400))
    view.added_items = [('left', 100, 200, 0), ('right', 300, 400, 0)]
    
    # Call the clear method
    controller.on_clear_points()
    
    # Check that the model and view were cleared
    assert model.get_points('left') == []
    assert model.get_points('right') == []
    assert view.added_items == []


def test_update_grid_lines_not_enough_points(setup_controller):
    """Test that grid lines are not drawn with fewer than 4 points."""
    model, view, controller = setup_controller
    
    # Add 3 points (not enough for grid)
    model.add_point('left', (100, 100))
    model.add_point('left', (200, 100))
    model.add_point('left', (100, 200))
    
    # Update grid lines
    controller._update_grid_lines('left')
    
    # Check that no grid lines were drawn
    assert 'left' not in view.grid_lines


def test_update_grid_lines(setup_controller):
    """Test drawing grid lines."""
    model, view, controller = setup_controller
    
    # Add 4 points (2x2 grid)
    model.add_point('left', (100, 100))
    model.add_point('left', (200, 100))
    model.add_point('left', (100, 200))
    model.add_point('left', (200, 200))
    
    # Update grid lines
    controller._update_grid_lines('left')
    
    # Check that grid lines were drawn
    assert 'left' in view.grid_lines
    points, rows, cols = view.grid_lines['left']
    assert rows == 2
    assert cols == 2
    assert len(points) == 4


def test_set_images(setup_controller, monkeypatch):
    """Test setting images in the view."""
    model, view, controller = setup_controller
    
    # Mock QPixmap to avoid issues with headless testing
    class MockQPixmap:
        def __init__(self, width, height):
            self.width = width
            self.height = height
            
    # Patch QPixmap
    monkeypatch.setattr("PySide6.QtGui.QPixmap", MockQPixmap)
    
    # Create test images
    left_image = MockQPixmap(400, 300)
    right_image = MockQPixmap(400, 300)
    
    # Set images
    controller.set_images(left_image, right_image)
    
    # Check that images were set in the view
    assert view.left_image is left_image
    assert view.right_image is right_image


@patch('PySide6.QtWidgets.QFileDialog.getSaveFileName')
@patch('PySide6.QtWidgets.QMessageBox.information')
def test_on_save(mock_message_box, mock_file_dialog, setup_controller, temp_calibration_file):
    """Test saving calibration points to a file."""
    model, view, controller = setup_controller
    
    # Setup mock to return a file path
    mock_file_dialog.return_value = (str(temp_calibration_file), "JSON Files (*.json)")
    
    # Add some points
    model.add_point('left', (100, 100))
    model.add_point('left', (200, 100))
    
    # Call the save method
    controller.on_save()
    
    # Check that file dialog was called
    mock_file_dialog.assert_called_once()
    
    # Check that success message was shown
    mock_message_box.assert_called_once()
    
    # Check that the file was created
    assert os.path.exists(temp_calibration_file)
    
    # Check the file content
    with open(temp_calibration_file, 'r') as f:
        data = json.load(f)
    
    assert "points" in data
    assert "left" in data["points"]
    assert data["points"]["left"] == [[100, 100], [200, 100]]


@patch('PySide6.QtWidgets.QFileDialog.getOpenFileName')
@patch('PySide6.QtWidgets.QMessageBox.information')
def test_on_load(mock_message_box, mock_file_dialog, setup_controller, temp_calibration_file):
    """Test loading calibration points from a file."""
    model, view, controller = setup_controller
    
    # Setup mock to return a file path
    mock_file_dialog.return_value = (str(temp_calibration_file), "JSON Files (*.json)")
    
    # Call the load method
    controller.on_load()
    
    # Check that file dialog was called
    mock_file_dialog.assert_called_once()
    
    # Check that success message was shown
    mock_message_box.assert_called_once()
    
    # Check that the model was updated
    assert len(model.get_points('left')) == 4
    assert len(model.get_points('right')) == 4
    
    # Check that the view was updated
    assert len(view.added_items) == 8  # 4 left + 4 right points
    assert 'left' in view.grid_lines
    assert 'right' in view.grid_lines


def test_render_loaded_points(setup_controller):
    """Test rendering loaded points in the view."""
    model, view, controller = setup_controller
    
    # Add some points to the model
    model.add_point('left', (100, 100))
    model.add_point('left', (200, 100))
    model.add_point('left', (100, 200))
    model.add_point('left', (200, 200))
    
    model.add_point('right', (300, 300))
    model.add_point('right', (400, 300))
    
    # Call the render method
    controller._render_loaded_points()
    
    # Check that the view was updated
    assert len(view.added_items) == 6  # 4 left + 2 right points
    
    # Check that grid lines were drawn for left side (4 points)
    assert 'left' in view.grid_lines
    left_points, left_rows, left_cols = view.grid_lines['left']
    assert left_rows == 2
    assert left_cols == 2
    
    # Check that grid lines were not drawn for right side (only 2 points)
    assert 'right' not in view.grid_lines 


class TestCalibrationController(unittest.TestCase):
    """Test suite for the CalibrationController class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mocks
        self.model = CalibrationModel()
        self.view = MagicMock(spec=CalibrationTab)
        self.config_manager = MagicMock(spec=ConfigManager)
        
        # Initialize controller
        self.controller = CalibrationController(self.model, self.view, self.config_manager)
        
    def test_on_add_point(self):
        """Test adding a point."""
        # Call the method
        self.controller.on_add_point('left', 10.0, 20.0)
        
        # Check model
        self.assertEqual(self.model.get_points('left'), [(10.0, 20.0)])
        
        # Check view method calls
        self.view.add_point_item.assert_called_with('left', 10.0, 20.0, 0)
        
    def test_on_move_point(self):
        """Test moving a point."""
        # Add a point first
        self.model.add_point('left', (10.0, 20.0))
        
        # Call the method
        self.controller.on_move_point('left', 0, 15.0, 25.0)
        
        # Check model
        self.assertEqual(self.model.get_points('left'), [(15.0, 25.0)])
        
    def test_on_clear_points(self):
        """Test clearing points."""
        # Add some points
        self.model.add_point('left', (10.0, 20.0))
        self.model.add_point('right', (30.0, 40.0))
        
        # Call the method
        self.controller.on_clear_points()
        
        # Check model
        self.assertEqual(self.model.get_points('left'), [])
        self.assertEqual(self.model.get_points('right'), [])
        
        # Check view method calls
        self.view.clear_points.assert_called_once()
        
    @patch('src.services.roi_cropper.crop_roi')
    @patch('src.services.roi_cropper.crop_roi_with_padding')
    @patch('src.services.skeletonizer.skeletonize_roi')
    @patch('src.services.intersection_finder.find_and_sort_intersections')
    def test_fine_tune_points(self, mock_find_intersections, mock_skeletonize, 
                             mock_crop_with_padding, mock_crop_roi):
        """Test fine-tuning points with mocked service functions."""
        # Mock stereo_image_model and current_frame
        mock_stereo_model = MagicMock()
        mock_frame = MagicMock()
        
        # Create a test image
        test_image = np.zeros((100, 100), dtype=np.uint8)
        
        # Mock the get_left_image and get_right_image methods
        mock_frame.get_left_image.return_value = test_image
        mock_frame.get_right_image.return_value = test_image
        
        # Mock stereo_image_model to return the mock frame
        mock_stereo_model.get_current_frame.return_value = mock_frame
        
        # Set stereo_image_model
        self.controller.stereo_image_model = mock_stereo_model
        
        # Add points to the model
        self.model.add_point('left', (50.0, 50.0))
        self.model.add_point('right', (60.0, 60.0))
        
        # Set up mocks for services
        roi = np.zeros((50, 50), dtype=np.uint8)
        mock_crop_roi.return_value = roi
        
        skeleton = np.zeros((50, 50), dtype=np.uint8)
        mock_skeletonize.return_value = skeleton
        
        # Mock intersection finder to return a better point
        mock_find_intersections.return_value = [(25, 25)]  # First point in the ROI
        
        # Mock crop_roi_with_padding to return a padded ROI and offset
        mock_crop_with_padding.return_value = (roi, (40, 40))
        
        # Call fine-tune method
        self.controller.on_fine_tune()
        
        # Verify service calls
        mock_crop_roi.assert_called()
        mock_skeletonize.assert_called()
        mock_find_intersections.assert_called()
        mock_crop_with_padding.assert_called()
        
        # Verify points were updated in the model
        # The new point should be at (40+25, 40+25) = (65, 65) for left point
        left_points = self.model.get_points('left')
        right_points = self.model.get_points('right')
        
        # Check if at least one point was updated
        self.assertEqual(left_points[0], (65, 65))
        self.assertEqual(right_points[0], (65, 65))
        
        # Verify view was updated
        self.view.update_point_item.assert_called()
        self.view.hide_roi.assert_called()  # Should hide ROI after processing
        
    def test_fine_tune_no_stereo_model(self):
        """Test fine-tune handling when no stereo model is available."""
        # Ensure no stereo model
        self.controller.stereo_image_model = None
        
        # Call fine-tune method
        self.controller.on_fine_tune()
        
        # Verify warning dialog was shown
        self.view.QMessageBox.warning.assert_called()
        
    def test_fine_tune_no_intersections(self):
        """Test fine-tune when no intersections are found."""
        # Mock stereo_image_model and current_frame
        mock_stereo_model = MagicMock()
        mock_frame = MagicMock()
        
        # Create a test image
        test_image = np.zeros((100, 100), dtype=np.uint8)
        
        # Mock the get_left_image and get_right_image methods
        mock_frame.get_left_image.return_value = test_image
        mock_frame.get_right_image.return_value = test_image
        
        # Mock stereo_image_model to return the mock frame
        mock_stereo_model.get_current_frame.return_value = mock_frame
        
        # Set stereo_image_model
        self.controller.stereo_image_model = mock_stereo_model
        
        # Add a point to the model
        self.model.add_point('left', (50.0, 50.0))
        
        # Mock the services to simulate no intersections found
        with patch('src.services.roi_cropper.crop_roi', return_value=np.zeros((50, 50))), \
             patch('src.services.skeletonizer.skeletonize_roi', return_value=np.zeros((50, 50))), \
             patch('src.services.intersection_finder.find_and_sort_intersections', return_value=[]):
            
            # Call fine-tune method
            self.controller.on_fine_tune()
            
            # Verify point wasn't updated (should remain at original position)
            self.assertEqual(self.model.get_points('left')[0], (50.0, 50.0)) 