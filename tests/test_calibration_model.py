#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test module for CalibrationModel.
"""

import pytest
import json
import os
import tempfile
from pathlib import Path

from src.models.calibration_model import CalibrationModel


@pytest.fixture
def model():
    """Create a CalibrationModel for testing."""
    return CalibrationModel()


@pytest.fixture
def model_with_points():
    """Create a CalibrationModel with test points."""
    model = CalibrationModel()
    # Add left points
    model.add_point('left', (100, 100))
    model.add_point('left', (200, 100))
    model.add_point('left', (100, 200))
    model.add_point('left', (200, 200))
    # Add right points
    model.add_point('right', (300, 300))
    model.add_point('right', (400, 300))
    model.add_point('right', (300, 400))
    model.add_point('right', (400, 400))
    return model


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)


def test_init(model):
    """Test initialization of the model."""
    assert model.left_pts == []
    assert model.right_pts == []


def test_add_point(model):
    """Test adding points to the model."""
    model.add_point('left', (100, 100))
    model.add_point('right', (200, 200))
    
    assert model.left_pts == [(100, 100)]
    assert model.right_pts == [(200, 200)]


def test_update_point(model):
    """Test updating points in the model."""
    # Add points first
    model.add_point('left', (100, 100))
    model.add_point('right', (200, 200))
    
    # Then update them
    model.update_point('left', 0, (150, 150))
    model.update_point('right', 0, (250, 250))
    
    assert model.left_pts == [(150, 150)]
    assert model.right_pts == [(250, 250)]


def test_get_points(model):
    """Test getting points from the model."""
    # Add points
    model.add_point('left', (100, 100))
    model.add_point('right', (200, 200))
    
    # Get points
    left_points = model.get_points('left')
    right_points = model.get_points('right')
    
    assert left_points == [(100, 100)]
    assert right_points == [(200, 200)]


def test_clear_points(model):
    """Test clearing points from the model."""
    # Add points
    model.add_point('left', (100, 100))
    model.add_point('right', (200, 200))
    
    # Clear points
    model.clear_points()
    
    assert model.left_pts == []
    assert model.right_pts == []


def test_to_dict(model_with_points):
    """Test converting the model to a dictionary."""
    # Convert to dictionary
    data = model_with_points.to_dict()
    
    # Check structure
    assert "points" in data
    assert "left" in data["points"]
    assert "right" in data["points"]
    
    # Check point values
    assert len(data["points"]["left"]) == 4
    assert len(data["points"]["right"]) == 4
    assert data["points"]["left"][0] == (100, 100)
    assert data["points"]["right"][0] == (300, 300)


def test_from_dict(model):
    """Test loading the model from a dictionary."""
    # Prepare test data
    data = {
        "points": {
            "left": [(100, 100), (200, 100), (100, 200), (200, 200)],
            "right": [(300, 300), (400, 300), (300, 400), (400, 400)]
        }
    }
    
    # Load data
    model.from_dict(data)
    
    # Check that points were loaded
    assert len(model.left_pts) == 4
    assert len(model.right_pts) == 4
    assert model.left_pts[0] == (100, 100)
    assert model.right_pts[0] == (300, 300)


def test_from_dict_invalid(model):
    """Test handling of invalid dictionary data."""
    # Prepare invalid data (no 'points' key)
    data = {"invalid": "data"}
    
    # Try to load data, should raise ValueError
    with pytest.raises(ValueError):
        model.from_dict(data)


def test_save_to_file(model_with_points, temp_dir):
    """Test saving the model to a file."""
    # Define file path
    file_path = os.path.join(temp_dir, "calibration.json")
    
    # Save to file
    success = model_with_points.save_to_file(file_path)
    
    # Check that save was successful
    assert success
    assert os.path.exists(file_path)
    
    # Check file content
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    assert "points" in data
    assert "left" in data["points"]
    assert "right" in data["points"]
    assert len(data["points"]["left"]) == 4
    assert len(data["points"]["right"]) == 4


def test_load_from_file(model, temp_dir):
    """Test loading the model from a file."""
    # Define file path
    file_path = os.path.join(temp_dir, "calibration.json")
    
    # Prepare test data
    data = {
        "points": {
            "left": [[100, 100], [200, 100], [100, 200], [200, 200]],
            "right": [[300, 300], [400, 300], [300, 400], [400, 400]]
        }
    }
    
    # Write data to file
    with open(file_path, 'w') as f:
        json.dump(data, f)
    
    # Load from file
    success = model.load_from_file(file_path)
    
    # Check that load was successful
    assert success
    assert len(model.left_pts) == 4
    assert len(model.right_pts) == 4
    
    # Check that points were converted to tuples
    assert isinstance(model.left_pts[0], tuple)
    assert isinstance(model.right_pts[0], tuple)


def test_load_from_nonexistent_file(model):
    """Test loading from a file that doesn't exist."""
    success = model.load_from_file("nonexistent_file.json")
    
    # Should return False but not crash
    assert not success 