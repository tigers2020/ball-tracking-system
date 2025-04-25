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

from PySide6.QtCore import QPointF, Qt, QObject
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFileDialog, QMessageBox

from src.models.calibration_model import CalibrationModel
from src.views.calibration_tab import CalibrationTab
from src.controllers.calibration_controller import CalibrationController


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
    controller = CalibrationController(model, view)
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


def test_on_load_current_frame(setup_controller):
    """Test loading the current frame from the stereo image model."""
    model, view, controller = setup_controller
    
    # Create a mock for the stereo image model
    mock_stereo_model = MagicMock()
    mock_frame = MagicMock()
    
    # Set up mock return values
    mock_stereo_model.get_current_frame.return_value = mock_frame
    mock_frame.get_left_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_frame.get_right_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Set the stereo image model in the controller
    controller.set_stereo_image_model(mock_stereo_model)
    
    # Call the method being tested
    controller.on_load_current_frame()
    
    # Verify that the stereo image model was queried for the current frame
    mock_stereo_model.get_current_frame.assert_called_once()
    
    # Verify that the frame's images were retrieved
    mock_frame.get_left_image.assert_called_once()
    mock_frame.get_right_image.assert_called_once()
    
    # Verify that images were set in the view
    assert view.left_image is not None
    assert view.right_image is not None 