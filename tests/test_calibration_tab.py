#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test module for CalibrationTab.
"""

import pytest
from pytestqt.qt_compat import qt_api
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPixmap

from src.views.calibration_tab import CalibrationTab


@pytest.fixture
def calibration_tab(qtbot):
    """Create a CalibrationTab widget for testing."""
    tab = CalibrationTab()
    qtbot.addWidget(tab)
    
    # Create dummy images
    left_image = QPixmap(400, 300)
    left_image.fill(Qt.white)
    
    right_image = QPixmap(400, 300)
    right_image.fill(Qt.white)
    
    # Set images
    tab.set_images(left_image, right_image)
    
    return tab


def test_init(calibration_tab):
    """Test initialization of the CalibrationTab widget."""
    # Check that the views and scenes are created
    assert calibration_tab.left_scene is not None
    assert calibration_tab.right_scene is not None
    assert calibration_tab.left_view is not None
    assert calibration_tab.right_view is not None
    
    # Check that the points dictionaries are empty
    assert len(calibration_tab.left_points) == 0
    assert len(calibration_tab.right_points) == 0


def test_click_emits_point_added(qtbot, calibration_tab):
    """Test that clicking on the view emits point_added signal."""
    # Create a local slot to receive the signal
    received_signal = False
    received_args = []
    
    def slot(side, x, y):
        nonlocal received_signal, received_args
        received_signal = True
        received_args = [side, x, y]
    
    # Connect our local slot to the signal
    calibration_tab.point_added.connect(slot)
    
    # Click on the left view
    qtbot.mouseClick(calibration_tab.left_view.viewport(), Qt.LeftButton, pos=QPoint(100, 100))
    
    # Wait for slot to be called
    qtbot.wait(100)
    
    # Check that our slot was called with correct parameters
    assert received_signal, "Signal was not emitted"
    assert received_args[0] == 'left'  # side
    assert isinstance(received_args[1], float)  # x
    assert isinstance(received_args[2], float)  # y
    
    # Clean up
    calibration_tab.point_added.disconnect(slot)


def test_add_point_item(calibration_tab):
    """Test adding a point item to the view."""
    # Add a point to the left view
    calibration_tab.add_point_item('left', 100, 100, 0)
    
    # Check that the point was added to the dictionary
    assert 0 in calibration_tab.left_points
    assert len(calibration_tab.left_points) == 1
    
    # Add a point to the right view
    calibration_tab.add_point_item('right', 200, 200, 0)
    
    # Check that the point was added to the dictionary
    assert 0 in calibration_tab.right_points
    assert len(calibration_tab.right_points) == 1


def test_update_point_item(calibration_tab):
    """Test updating a point item's position."""
    # First add a point
    calibration_tab.add_point_item('left', 100, 100, 0)
    
    # Update the point
    calibration_tab.update_point_item('left', 0, 150, 150)
    
    # Check that the point's position was updated
    point = calibration_tab.left_points[0]
    center_x = point.pos().x() + point.radius
    center_y = point.pos().y() + point.radius
    
    assert abs(center_x - 150) < 1e-5
    assert abs(center_y - 150) < 1e-5


def test_clear_points(calibration_tab):
    """Test clearing all points."""
    # Add points
    calibration_tab.add_point_item('left', 100, 100, 0)
    calibration_tab.add_point_item('right', 200, 200, 0)
    
    # Clear points
    calibration_tab.clear_points()
    
    # Check that the points were cleared
    assert len(calibration_tab.left_points) == 0
    assert len(calibration_tab.right_points) == 0


def test_show_and_hide_roi(calibration_tab):
    """Test showing and hiding ROI overlay."""
    # Show ROI
    calibration_tab.show_roi('left', (100, 100), 20)
    
    # Check that the ROI overlay was created
    assert calibration_tab.left_roi_overlay is not None
    
    # Hide ROI
    calibration_tab.hide_roi('left')
    
    # Check that the ROI overlay was removed
    assert calibration_tab.left_roi_overlay is None


def test_draw_grid_lines(calibration_tab):
    """Test drawing grid lines."""
    # Define points in a 2x2 grid
    points = [(100, 100), (200, 100), (100, 200), (200, 200)]
    
    # Draw grid lines
    calibration_tab.draw_grid_lines('left', points, 2, 2)
    
    # Check that grid lines were created
    assert len(calibration_tab.left_grid_lines) == 4  # 2 horizontal + 2 vertical
    
    # Clear grid lines
    calibration_tab._clear_grid_lines()
    
    # Check that grid lines were cleared
    assert len(calibration_tab.left_grid_lines) == 0 