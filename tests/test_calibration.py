#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the calibration functionality.
"""

import os
import sys
import json
import pytest
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt, QPointF
from PySide6.QtWidgets import QApplication

from src.models.calibration_model import CalibrationModel
from src.views.calibration_tab import CalibrationTab, CalibrationPointItem
from src.controllers.calibration_controller import CalibrationController
from src.utils.ui_constants import Calibration


@pytest.fixture
def calibration_model():
    """Fixture for a calibration model."""
    return CalibrationModel()


@pytest.fixture
def calibration_view(qtbot):
    """Fixture for a calibration view."""
    view = CalibrationTab()
    qtbot.addWidget(view)
    return view


@pytest.fixture
def calibration_controller(calibration_model, calibration_view, monkeypatch):
    """Fixture for a calibration controller with a temporary config file."""
    # Create a temporary directory for config
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_config = os.path.join(temp_dir, "calibration.json")
        
        # Monkeypatch the config file path
        monkeypatch.setattr(Calibration, "CONFIG_FILE", temp_config)
        
        # Create controller
        controller = CalibrationController(calibration_model, calibration_view)
        controller.initialize()
        
        yield controller


class TestCalibrationModel:
    """Tests for the CalibrationModel class."""
    
    def test_add_point(self, calibration_model):
        """Test adding a point to the model."""
        # Add a point to the left side
        assert calibration_model.add_point("left", (100, 200))
        assert len(calibration_model.left_pts) == 1
        assert calibration_model.left_pts[0] == (100, 200)
        
        # Add a point to the right side
        assert calibration_model.add_point("right", (150, 250))
        assert len(calibration_model.right_pts) == 1
        assert calibration_model.right_pts[0] == (150, 250)
        
        # Add a point to an invalid side
        assert not calibration_model.add_point("invalid", (200, 300))
    
    def test_update_point(self, calibration_model):
        """Test updating a point in the model."""
        # Add points
        calibration_model.add_point("left", (100, 200))
        calibration_model.add_point("right", (150, 250))
        
        # Update points
        assert calibration_model.update_point("left", 0, (110, 210))
        assert calibration_model.left_pts[0] == (110, 210)
        
        assert calibration_model.update_point("right", 0, (160, 260))
        assert calibration_model.right_pts[0] == (160, 260)
        
        # Update with invalid index
        assert not calibration_model.update_point("left", 1, (120, 220))
        assert not calibration_model.update_point("right", 1, (170, 270))
        
        # Update with invalid side
        assert not calibration_model.update_point("invalid", 0, (130, 230))
    
    def test_remove_point(self, calibration_model):
        """Test removing a point from the model."""
        # Add points
        calibration_model.add_point("left", (100, 200))
        calibration_model.add_point("left", (110, 210))
        calibration_model.add_point("right", (150, 250))
        calibration_model.add_point("right", (160, 260))
        
        # Remove points
        assert calibration_model.remove_point("left", 0)
        assert len(calibration_model.left_pts) == 1
        assert calibration_model.left_pts[0] == (110, 210)
        
        assert calibration_model.remove_point("right", 1)
        assert len(calibration_model.right_pts) == 1
        assert calibration_model.right_pts[0] == (150, 250)
        
        # Remove with invalid index
        assert not calibration_model.remove_point("left", 1)
        assert not calibration_model.remove_point("right", 1)
        
        # Remove with invalid side
        assert not calibration_model.remove_point("invalid", 0)
    
    def test_clear_points(self, calibration_model):
        """Test clearing points from the model."""
        # Add points
        calibration_model.add_point("left", (100, 200))
        calibration_model.add_point("left", (110, 210))
        calibration_model.add_point("right", (150, 250))
        calibration_model.add_point("right", (160, 260))
        
        # Clear left points
        calibration_model.clear_points("left")
        assert len(calibration_model.left_pts) == 0
        assert len(calibration_model.right_pts) == 2
        
        # Clear right points
        calibration_model.clear_points("right")
        assert len(calibration_model.left_pts) == 0
        assert len(calibration_model.right_pts) == 0
        
        # Add points again
        calibration_model.add_point("left", (100, 200))
        calibration_model.add_point("right", (150, 250))
        
        # Clear all points
        calibration_model.clear_points()
        assert len(calibration_model.left_pts) == 0
        assert len(calibration_model.right_pts) == 0
    
    def test_to_dict(self, calibration_model):
        """Test converting the model to a dictionary."""
        # Add points
        calibration_model.add_point("left", (100, 200))
        calibration_model.add_point("right", (150, 250))
        
        # Convert to dictionary
        data = calibration_model.to_dict()
        
        # Check structure
        assert "points" in data
        assert "left" in data["points"]
        assert "right" in data["points"]
        assert "calib_ver" in data
        
        # Check data
        assert data["points"]["left"] == [(100, 200)]
        assert data["points"]["right"] == [(150, 250)]
        assert data["calib_ver"] == 1.0
    
    def test_from_dict(self, calibration_model):
        """Test loading the model from a dictionary."""
        # Create dictionary
        data = {
            "points": {
                "left": [(100, 200), (110, 210)],
                "right": [(150, 250), (160, 260)]
            },
            "calib_ver": 1.0
        }
        
        # Load from dictionary
        assert calibration_model.from_dict(data)
        
        # Check data
        assert len(calibration_model.left_pts) == 2
        assert len(calibration_model.right_pts) == 2
        assert calibration_model.left_pts[0] == (100, 200)
        assert calibration_model.left_pts[1] == (110, 210)
        assert calibration_model.right_pts[0] == (150, 250)
        assert calibration_model.right_pts[1] == (160, 260)
        
        # Test with old format
        old_data = {
            "raw_points": {
                "left": [(120, 220)],
                "right": [(170, 270)]
            }
        }
        
        # Clear and load from old format
        calibration_model.clear_points()
        assert calibration_model.from_dict(old_data)
        
        # Check data
        assert len(calibration_model.left_pts) == 1
        assert len(calibration_model.right_pts) == 1
        assert calibration_model.left_pts[0] == (120, 220)
        assert calibration_model.right_pts[0] == (170, 270)
        
        # Test with invalid data
        invalid_data = {"invalid_key": "invalid_value"}
        
        # Clear and attempt to load invalid data
        calibration_model.clear_points()
        assert not calibration_model.from_dict(invalid_data)
        
        # Check that no data was loaded
        assert len(calibration_model.left_pts) == 0
        assert len(calibration_model.right_pts) == 0


class TestCalibrationController:
    """Tests for the CalibrationController class."""
    
    def test_add_point_signal(self, calibration_controller, qtbot):
        """Test that adding a point updates the model and view."""
        model = calibration_controller.model
        view = calibration_controller.view
        
        # Mock the view's update_points method
        called = {"left": False, "right": False}
        
        def mock_update_left(side, points):
            if side == "left":
                called["left"] = True
                assert points == [(100, 200)]
        
        def mock_update_right(side, points):
            if side == "right":
                called["right"] = True
                assert points == [(150, 250)]
        
        # Apply the mock
        original_update_points = view.update_points
        view.update_points = mock_update_left
        
        # Emit the signal
        view.point_added.emit("left", (100, 200))
        
        # Check that the model was updated
        assert len(model.left_pts) == 1
        assert model.left_pts[0] == (100, 200)
        
        # Check that the view's update_points was called
        assert called["left"]
        
        # Now test for right side
        view.update_points = mock_update_right
        view.point_added.emit("right", (150, 250))
        
        # Check that the model was updated
        assert len(model.right_pts) == 1
        assert model.right_pts[0] == (150, 250)
        
        # Check that the view's update_points was called
        assert called["right"]
        
        # Restore the original method
        view.update_points = original_update_points
    
    def test_update_point_signal(self, calibration_controller, qtbot):
        """Test that updating a point updates the model and view."""
        model = calibration_controller.model
        view = calibration_controller.view
        
        # Add points first
        model.add_point("left", (100, 200))
        model.add_point("right", (150, 250))
        
        # Mock the view's update_points method
        called = {"left": False, "right": False}
        
        def mock_update_left(side, points):
            if side == "left":
                called["left"] = True
                assert points == [(110, 210)]
        
        def mock_update_right(side, points):
            if side == "right":
                called["right"] = True
                assert points == [(160, 260)]
        
        # Apply the mock
        original_update_points = view.update_points
        view.update_points = mock_update_left
        
        # Emit the signal
        view.point_updated.emit("left", 0, (110, 210))
        
        # Check that the model was updated
        assert model.left_pts[0] == (110, 210)
        
        # Check that the view's update_points was called
        assert called["left"]
        
        # Now test for right side
        view.update_points = mock_update_right
        view.point_updated.emit("right", 0, (160, 260))
        
        # Check that the model was updated
        assert model.right_pts[0] == (160, 260)
        
        # Check that the view's update_points was called
        assert called["right"]
        
        # Restore the original method
        view.update_points = original_update_points
    
    def test_clear_points_signal(self, calibration_controller, qtbot):
        """Test that clearing points updates the model and view."""
        model = calibration_controller.model
        view = calibration_controller.view
        
        # Add points first
        model.add_point("left", (100, 200))
        model.add_point("right", (150, 250))
        
        # Mock the view's update_points method
        called = {"left": False, "right": False}
        
        def mock_update_points(side, points):
            called[side] = True
            assert points == []
        
        # Apply the mock
        original_update_points = view.update_points
        view.update_points = mock_update_points
        
        # Emit the signal for left side
        view.points_cleared.emit("left")
        
        # Check that the model was updated
        assert len(model.left_pts) == 0
        assert len(model.right_pts) == 1
        
        # Check that the view's update_points was called
        assert called["left"]
        assert not called["right"]
        
        # Reset the mock and test for None (both sides)
        called["left"] = False
        called["right"] = False
        
        # Add points again to both sides
        model.add_point("left", (100, 200))
        
        # Create a new mock function that doesn't assert empty list
        # since we'll manually check the model state after the call
        def mock_update_points_no_assert(side, points):
            called[side] = True
        
        # Apply the new mock
        view.update_points = mock_update_points_no_assert
        
        # Emit the signal for both sides
        view.points_cleared.emit(None)
        
        # Check that the view's update_points was called for both sides
        assert called["left"]
        assert called["right"]
        
        # Check that the model was updated
        assert len(model.left_pts) == 0
        assert len(model.right_pts) == 0
        
        # Restore the original method
        view.update_points = original_update_points
    
    def test_save_load_calibration(self, calibration_controller, qtbot, monkeypatch):
        """Test saving and loading calibration data."""
        model = calibration_controller.model
        view = calibration_controller.view
        
        # Add points
        model.add_point("left", (100, 200))
        model.add_point("right", (150, 250))
        
        # Mock the view's show_info method
        info_called = False
        
        def mock_show_info(message):
            nonlocal info_called
            info_called = True
        
        # Apply the mock
        original_show_info = view.show_info
        view.show_info = mock_show_info
        
        # Save calibration
        calibration_controller.save_calibration()
        
        # Check that show_info was called
        assert info_called
        
        # Check that the file was created
        assert os.path.exists(Calibration.CONFIG_FILE)
        
        # Clear the model
        model.clear_points()
        
        # Reset the mock
        info_called = False
        
        # Load calibration
        calibration_controller.load_calibration()
        
        # Check that show_info was called
        assert info_called
        
        # Check that the model was loaded
        assert len(model.left_pts) == 1
        assert len(model.right_pts) == 1
        
        # Convert to tuple for comparison if needed
        left_point = tuple(model.left_pts[0]) if isinstance(model.left_pts[0], list) else model.left_pts[0]
        right_point = tuple(model.right_pts[0]) if isinstance(model.right_pts[0], list) else model.right_pts[0]
        
        assert left_point == (100, 200)
        assert right_point == (150, 250)
        
        # Restore the original method
        view.show_info = original_show_info


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 