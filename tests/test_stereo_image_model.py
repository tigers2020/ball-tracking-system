#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the StereoImageModel class.
"""

import os
import sys
import shutil
import tempfile
import unittest
from pathlib import Path

import pytest
import numpy as np
import cv2
from PySide6.QtCore import QCoreApplication

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.stereo_image_model import StereoImageModel, StereoFrame


class TestStereoFrame(unittest.TestCase):
    """Tests for the StereoFrame class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory and test images
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test images
        self.left_image_path = os.path.join(self.temp_dir, "left.png")
        self.right_image_path = os.path.join(self.temp_dir, "right.png")
        
        # Create simple test images
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:, :, 0] = 255  # Red channel for left image
        cv2.imwrite(self.left_image_path, test_image)
        
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:, :, 2] = 255  # Blue channel for right image
        cv2.imwrite(self.right_image_path, test_image)
        
        # Create test frame
        self.frame = StereoFrame(0, self.left_image_path, self.right_image_path)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary directory and its contents
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.frame.index, 0)
        self.assertEqual(self.frame.left_image_path, self.left_image_path)
        self.assertEqual(self.frame.right_image_path, self.right_image_path)
        self.assertIsNone(self.frame._left_image)
        self.assertIsNone(self.frame._right_image)
    
    def test_load_images(self):
        """Test loading images."""
        left_image, right_image = self.frame.load_images()
        
        # Check that images were loaded
        self.assertIsNotNone(left_image)
        self.assertIsNotNone(right_image)
        
        # Check image dimensions
        self.assertEqual(left_image.shape, (100, 100, 3))
        self.assertEqual(right_image.shape, (100, 100, 3))
        
        # Check image content (simple check on color channels)
        self.assertEqual(left_image[50, 50, 0], 255)  # Red channel should be 255
        self.assertEqual(right_image[50, 50, 2], 255)  # Blue channel should be 255
    
    def test_get_left_image(self):
        """Test getting the left image."""
        # Image should be loaded on demand
        self.assertIsNone(self.frame._left_image)
        
        left_image = self.frame.get_left_image()
        
        # Check that image was loaded
        self.assertIsNotNone(left_image)
        self.assertIsNotNone(self.frame._left_image)
        
        # Check image dimensions
        self.assertEqual(left_image.shape, (100, 100, 3))
        
        # Check image content (simple check on color channels)
        self.assertEqual(left_image[50, 50, 0], 255)  # Red channel should be 255
    
    def test_get_right_image(self):
        """Test getting the right image."""
        # Image should be loaded on demand
        self.assertIsNone(self.frame._right_image)
        
        right_image = self.frame.get_right_image()
        
        # Check that image was loaded
        self.assertIsNotNone(right_image)
        self.assertIsNotNone(self.frame._right_image)
        
        # Check image dimensions
        self.assertEqual(right_image.shape, (100, 100, 3))
        
        # Check image content (simple check on color channels)
        self.assertEqual(right_image[50, 50, 2], 255)  # Blue channel should be 255
    
    def test_release_images(self):
        """Test releasing images."""
        # Load images
        self.frame.load_images()
        
        # Check that images are loaded
        self.assertIsNotNone(self.frame._left_image)
        self.assertIsNotNone(self.frame._right_image)
        
        # Release images
        self.frame.release_images()
        
        # Check that images are released
        self.assertIsNone(self.frame._left_image)
        self.assertIsNone(self.frame._right_image)


@pytest.fixture(scope="module")
def qapp():
    """Create a QApplication instance for tests."""
    app = QCoreApplication([])
    yield app
    app.quit()


class TestStereoImageModel:
    """Tests for the StereoImageModel class."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self, qapp):
        """Set up test fixtures."""
        # Create temporary directory and test images
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test directory structure
        os.makedirs(os.path.join(self.temp_dir, "images"))
        
        # Create test images
        self.create_test_images()
        
        # Create test model
        self.model = StereoImageModel()
        
        yield
        
        # Remove temporary directory and its contents
        shutil.rmtree(self.temp_dir)
    
    def create_test_images(self):
        """Create test images for the model."""
        images_dir = os.path.join(self.temp_dir, "images")
        
        # Create simple test images
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        for i in range(3):
            # Create left image (red)
            left_image = test_image.copy()
            left_image[:, :, 0] = 255
            cv2.imwrite(os.path.join(images_dir, f"image_{i}_L.png"), left_image)
            
            # Create right image (blue)
            right_image = test_image.copy()
            right_image[:, :, 2] = 255
            cv2.imwrite(os.path.join(images_dir, f"image_{i}_R.png"), right_image)
    
    def test_init(self):
        """Test initialization."""
        assert len(self.model.frames) == 0
        assert self.model.current_frame_index == 0
        assert self.model.total_frames == 0
        assert self.model.base_folder is None
        assert self.model.xml_path is None
        assert not self.model.is_playing
    
    def test_load_from_folder(self):
        """Test loading from a folder."""
        result = self.model.load_from_folder(os.path.join(self.temp_dir, "images"))
        
        # Check that loading was successful
        assert result is True
        
        # Check that frames were loaded
        assert len(self.model.frames) > 0
        
        # Check that the XML file was created
        assert os.path.exists(os.path.join(self.temp_dir, "images", "frames_info.xml"))
    
    def test_get_current_frame(self):
        """Test getting the current frame."""
        # Load frames
        self.model.load_from_folder(os.path.join(self.temp_dir, "images"))
        
        # Check getting the current frame
        frame = self.model.get_current_frame()
        assert frame is not None
        assert frame.index == 0
        
        # Check with invalid index
        self.model.current_frame_index = 999
        frame = self.model.get_current_frame()
        assert frame is None
    
    def test_set_current_frame_index(self):
        """Test setting the current frame index."""
        # Load frames
        self.model.load_from_folder(os.path.join(self.temp_dir, "images"))
        
        # Set valid index
        self.model.set_current_frame_index(1)
        assert self.model.current_frame_index == 1
        
        # Set invalid index (should not change)
        self.model.set_current_frame_index(999)
        assert self.model.current_frame_index == 1
    
    def test_next_frame(self):
        """Test moving to the next frame."""
        # Load frames
        self.model.load_from_folder(os.path.join(self.temp_dir, "images"))
        
        # Check initial state
        assert self.model.current_frame_index == 0
        
        # Move to next frame
        frame = self.model.next_frame()
        assert frame is not None
        assert self.model.current_frame_index == 1
        
        # Move to next frame
        frame = self.model.next_frame()
        assert frame is not None
        assert self.model.current_frame_index == 2
        
        # If there are exactly 3 frames, this should return None
        if len(self.model.frames) == 3:
            frame = self.model.next_frame()
            assert frame is None
            assert self.model.current_frame_index == 2
    
    def test_prev_frame(self):
        """Test moving to the previous frame."""
        # Load frames
        self.model.load_from_folder(os.path.join(self.temp_dir, "images"))
        
        # Set current frame to the last frame 
        last_index = len(self.model.frames) - 1
        self.model.set_current_frame_index(last_index)
        
        # Move to previous frame
        frame = self.model.prev_frame()
        assert frame is not None
        assert self.model.current_frame_index == last_index - 1
        
        # Move to previous frame
        frame = self.model.prev_frame()
        assert frame is not None
        assert self.model.current_frame_index == last_index - 2
        
        # Try to move before the first frame
        while self.model.current_frame_index > 0:
            self.model.prev_frame()
        
        frame = self.model.prev_frame()
        assert frame is None
        assert self.model.current_frame_index == 0 