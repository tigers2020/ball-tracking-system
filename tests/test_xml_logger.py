#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for xml_logger module.
"""

import unittest
import tempfile
import os
import shutil
import xml.etree.ElementTree as ET
from src.utils.xml_logger import XMLLogger


class TestXMLLogger(unittest.TestCase):
    """Test cases for XMLLogger class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create logger with small flush interval
        self.logger = XMLLogger(flush_interval=2)
    
    def tearDown(self):
        """Clean up after test."""
        # Close logger
        if self.logger.is_session_active:
            self.logger.close()
        
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_start_session(self):
        """Test start_session method."""
        # Start session
        result = self.logger.start_session("test_folder", self.temp_dir)
        
        # Check result
        self.assertTrue(result)
        self.assertTrue(self.logger.is_session_active)
        self.assertIsNotNone(self.logger.root)
        
        # Check file path
        expected_path = os.path.join(self.temp_dir, "tracking_data.xml")
        self.assertEqual(self.logger.file_path, expected_path)
    
    def test_log_frame(self):
        """Test log_frame method."""
        # Start session
        self.logger.start_session("test_folder", self.temp_dir)
        
        # Create test data
        frame_data = {
            "tracking_active": True,
            "left": {
                "hsv_center": {"x": 100.5, "y": 200.5},
                "hough_center": {"x": 110.5, "y": 210.5, "radius": 15.0},
                "kalman_prediction": {"x": 120.5, "y": 220.5, "vx": 1.0, "vy": 2.0},
                "fused_center": {"x": 115.0, "y": 215.0}
            },
            "right": {
                "hsv_center": {"x": 300.5, "y": 400.5},
                "hough_center": {"x": 310.5, "y": 410.5, "radius": 15.0},
                "kalman_prediction": {"x": 320.5, "y": 420.5, "vx": 1.0, "vy": 2.0},
                "fused_center": {"x": 315.0, "y": 415.0}
            }
        }
        
        # Log frame
        result = self.logger.log_frame(1, frame_data, "frame_001.png")
        
        # Check result
        self.assertTrue(result)
        self.assertEqual(self.logger.frame_count, 1)
        
        # Log another frame to trigger flush
        result = self.logger.log_frame(2, frame_data, "frame_002.png")
        
        # Check result
        self.assertTrue(result)
        self.assertEqual(self.logger.frame_count, 2)
        
        # Check that file exists
        self.assertTrue(os.path.exists(self.logger.file_path))
        
        # Parse XML file
        tree = ET.parse(self.logger.file_path)
        root = tree.getroot()
        
        # Check root attributes
        self.assertEqual(root.tag, "TrackingData")
        self.assertEqual(root.get("folder"), "test_folder")
        
        # Check number of images
        images = root.findall("Image")
        self.assertEqual(len(images), 2)
        
        # Check first image
        first_image = images[0]
        self.assertEqual(first_image.get("number"), "1")
        self.assertEqual(first_image.get("name"), "frame_001.png")
        self.assertEqual(first_image.get("tracking_active"), "True")
        
        # Check left data
        left = first_image.find("Left")
        self.assertIsNotNone(left)
        
        hsv = left.find("HSV")
        self.assertIsNotNone(hsv)
        self.assertEqual(hsv.get("x"), "100.5")
        self.assertEqual(hsv.get("y"), "200.5")
        
        hough = left.find("Hough")
        self.assertIsNotNone(hough)
        self.assertEqual(hough.get("x"), "110.5")
        self.assertEqual(hough.get("y"), "210.5")
        self.assertEqual(hough.get("radius"), "15.0")
        
        kalman = left.find("Kalman")
        self.assertIsNotNone(kalman)
        self.assertEqual(kalman.get("x"), "120.5")
        self.assertEqual(kalman.get("y"), "220.5")
        self.assertEqual(kalman.get("vx"), "1.0")
        self.assertEqual(kalman.get("vy"), "2.0")
        
        fused = left.find("Fused")
        self.assertIsNotNone(fused)
        self.assertEqual(fused.get("x"), "115.0")
        self.assertEqual(fused.get("y"), "215.0")
    
    def test_add_statistics(self):
        """Test add_statistics method."""
        # Start session
        self.logger.start_session("test_folder", self.temp_dir)
        
        # Log a frame
        frame_data = {"tracking_active": True, "left": {}, "right": {}}
        self.logger.log_frame(1, frame_data)
        
        # Add statistics
        stats = {
            "detection_rate": 0.75,
            "frames_total": 100,
            "detections_count": 75,
            "tracking_active": "True"
        }
        result = self.logger.add_statistics(stats)
        
        # Check result
        self.assertTrue(result)
        
        # Flush to disk
        self.logger.flush()
        
        # Parse XML file
        tree = ET.parse(self.logger.file_path)
        root = tree.getroot()
        
        # Check statistics
        stats_elem = root.find("Statistics")
        self.assertIsNotNone(stats_elem)
        self.assertEqual(stats_elem.get("detection_rate"), "0.75")
        self.assertEqual(stats_elem.get("frames_total"), "100")
        self.assertEqual(stats_elem.get("detections_count"), "75")
        self.assertEqual(stats_elem.get("tracking_active"), "True")
    
    def test_close(self):
        """Test close method."""
        # Start session
        self.logger.start_session("test_folder", self.temp_dir)
        
        # Log a frame
        frame_data = {"tracking_active": True, "left": {}, "right": {}}
        self.logger.log_frame(1, frame_data)
        
        # Close logger
        result = self.logger.close()
        
        # Check result
        self.assertTrue(result)
        self.assertFalse(self.logger.is_session_active)
        
        # Check that file exists
        self.assertTrue(os.path.exists(self.logger.file_path))
        
        # Parse XML file
        tree = ET.parse(self.logger.file_path)
        root = tree.getroot()
        
        # Check close time
        self.assertIsNotNone(root.get("closed"))
        self.assertIsNotNone(root.get("close_time"))
        self.assertEqual(root.get("total_frames"), "1")


if __name__ == "__main__":
    unittest.main() 