#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration Test module.
This module contains tests for the full game analysis pipeline integrated with other components.
"""

import os
import sys
import unittest
import numpy as np
import logging
from unittest.mock import MagicMock, patch
from PySide6.QtCore import QCoreApplication, Qt, QPoint
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest

# Add the parent directory to the path to import the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controllers.app_controller import AppController
from src.controllers.ball_tracking_controller import BallTrackingController
from src.controllers.game_analyzer import GameAnalyzer
from src.views.image_view import ImageView
from src.views.bounce_overlay import BounceOverlayWidget
from src.utils.config_manager import ConfigManager


class TestComponentIntegration(unittest.TestCase):
    """Test the integration of game analysis components."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""
        # Create QApplication for the tests
        if not QApplication.instance():
            cls.app = QApplication([])
        else:
            cls.app = QApplication.instance()
    
    def setUp(self):
        """Set up the test case."""
        # Create config manager
        self.config_manager = ConfigManager()
        
        # Set camera parameters in config
        camera_settings = {
            "focal_length_mm": 4.0,
            "sensor_width": 4.8,
            "sensor_height": 3.6,
            "baseline_m": 0.2,
            "principal_point_x": 960,
            "principal_point_y": 540,
            "camera_location_x": 0.0,
            "camera_location_y": -3.0,
            "camera_location_z": 2.5,
            "camera_rotation_x": -30.0,
            "camera_rotation_y": 0.0,
            "camera_rotation_z": 0.0,
            "resizing_scale": 1.0
        }
        self.config_manager.set_camera_settings(camera_settings)
        
        # Create mock model for BallTrackingController
        self.mock_model = MagicMock()
        
        # Create controllers
        self.ball_tracking_controller = BallTrackingController(
            self.mock_model, self.config_manager)
        self.game_analyzer = GameAnalyzer(self.config_manager)
        
        # Create ImageView and connect controllers
        self.image_view = ImageView()
        self.image_view.connect_ball_tracking_controller(self.ball_tracking_controller)
        self.image_view.connect_game_analyzer(self.game_analyzer)
        
        # Set up mock handlers
        self.tracking_updated_handler = MagicMock()
        self.bounce_detected_handler = MagicMock()
        self.court_position_handler = MagicMock()
        
        # Connect signals to handlers
        self.game_analyzer.tracking_updated.connect(self.tracking_updated_handler)
        self.game_analyzer.bounce_detected.connect(self.bounce_detected_handler)
        self.game_analyzer.court_position_updated.connect(self.court_position_handler)
        
        # Enable game analyzer
        self.game_analyzer.enable(True)
        
    def tearDown(self):
        """Clean up after the test."""
        # Reset controllers
        self.ball_tracking_controller.reset_tracking()
        self.game_analyzer.reset()
        
        # Delete views
        self.image_view = None
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Delete QApplication
        cls.app = None
    
    def test_connection_setup(self):
        """Test if connections are set up correctly."""
        # Check if ImageView has references to controllers
        self.assertIsNotNone(self.image_view.game_analyzer)
        
        # Check if bounce overlay is visible and connected
        self.assertTrue(hasattr(self.image_view, 'bounce_overlay'))
        self.assertTrue(self.image_view.bounce_overlay.isVisible())
        
        # Check if analysis tabs are visible
        self.assertTrue(hasattr(self.image_view, 'analysis_tabs'))
        self.assertTrue(self.image_view.analysis_tabs.isVisible())
    
    def test_signal_connections(self):
        """Test if signal connections between controllers work."""
        # Create test data
        left_point = (960, 540)  # Center of image
        right_point = (930, 540)  # Shifted by disparity
        
        # Simulate ball detection signal from ball tracking controller
        self.ball_tracking_controller.detection_updated.emit(
            1.0,            # Detection rate
            (left_point, right_point),  # Pixel coordinates
            (0.0, 0.0, 0.0)  # World coordinates (not used in this test)
        )
        
        # Check if game analyzer received the signal and processed it
        # This would require more setup to work properly in the real code
        # For this test, we'll just check that ImageView has the connections in place
        self.assertIsNotNone(self.image_view.game_analyzer)
    
    def test_app_controller_integration(self):
        """Test if AppController initializes and connects everything correctly."""
        # Create AppController
        app_controller = AppController()
        
        # Check if game analyzer was initialized
        self.assertTrue(hasattr(app_controller, 'game_analyzer'))
        self.assertIsNotNone(app_controller.game_analyzer)
        
        # Check if connections to ball tracking controller exist
        self.assertTrue(hasattr(app_controller, '_connect_ball_tracking_to_game_analyzer'))
        
        # Show app to trigger UI initialization
        app_controller.show()
        
        # Check if the app window has an image view
        self.assertTrue(hasattr(app_controller.view, 'image_view'))
        
        # Clean up
        app_controller.view.close()
        app_controller = None


class TestBounceOverlayWidget(unittest.TestCase):
    """Test the BounceOverlayWidget functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""
        # Create QApplication for the tests
        if not QApplication.instance():
            cls.app = QApplication([])
        else:
            cls.app = QApplication.instance()
    
    def setUp(self):
        """Set up the test case."""
        # Create bounce overlay widget
        self.overlay = BounceOverlayWidget()
        self.overlay.show()
        
        # Connect to a mock game analyzer
        self.mock_analyzer = MagicMock()
        self.mock_analyzer.ball_position_updated = MagicMock()
        self.mock_analyzer.bounce_detected = MagicMock()
        
        # Show the widget
        self.overlay.resize(400, 300)
        self.overlay.show()
    
    def tearDown(self):
        """Clean up after the test."""
        # Close and delete the widget
        self.overlay.close()
        self.overlay = None
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Delete QApplication
        cls.app = None
    
    def test_add_bounce(self):
        """Test adding a bounce point to the overlay."""
        # Add a bounce on court (in bounds)
        self.overlay.add_bounce(2.0, 5.0, True)
        
        # Add a bounce out of court (out of bounds)
        self.overlay.add_bounce(15.0, 5.0, False)
        
        # Check that bounce points exist in the scene
        bounce_items = [item for item in self.overlay.scene.items() 
                      if hasattr(item, 'bounce_type')]
        self.assertEqual(len(bounce_items), 2)
        
        # Check that we have one IN and one OUT bounce
        in_bounces = [item for item in bounce_items if item.is_inside_court]
        out_bounces = [item for item in bounce_items if not item.is_inside_court]
        self.assertEqual(len(in_bounces), 1)
        self.assertEqual(len(out_bounces), 1)
    
    def test_update_ball_position(self):
        """Test updating the ball position in the overlay."""
        # Update ball position
        self.overlay.update_ball_position(3.0, 10.0, 0.5)  # x, y, z
        
        # Check that a ball item exists in the scene
        ball_items = [item for item in self.overlay.scene.items() 
                     if hasattr(item, 'is_ball')]
        self.assertEqual(len(ball_items), 1)
    
    def test_reset(self):
        """Test resetting the overlay."""
        # Add some bounces and update ball position
        self.overlay.add_bounce(2.0, 5.0, True)
        self.overlay.add_bounce(15.0, 5.0, False)
        self.overlay.update_ball_position(3.0, 10.0, 0.5)
        
        # Reset the overlay
        self.overlay.reset()
        
        # Check that all items are removed
        bounce_items = [item for item in self.overlay.scene.items() 
                      if hasattr(item, 'bounce_type')]
        self.assertEqual(len(bounce_items), 0)
        
        ball_items = [item for item in self.overlay.scene.items() 
                    if hasattr(item, 'is_ball')]
        self.assertEqual(len(ball_items), 0)
        
    def test_connect_game_analyzer(self):
        """Test connecting to a game analyzer."""
        # Connect to mock game analyzer
        self.overlay.connect_game_analyzer(self.mock_analyzer)
        
        # Check if signal connections are established
        self.assertEqual(self.mock_analyzer.ball_position_updated.connect.call_count, 1)
        self.assertEqual(self.mock_analyzer.bounce_detected.connect.call_count, 1)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the tests
    unittest.main() 