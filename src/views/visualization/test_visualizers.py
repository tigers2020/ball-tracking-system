#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for visualization modules.
"""

import cv2
import numpy as np
import os
import sys
import logging

# Add the project root to sys.path if not already there
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import visualization modules using the proper interface
from src.views.visualization import OpenCVVisualizer

def create_test_image(width=640, height=480):
    """Create a test image with a gradient background."""
    # Create a gradient image
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Create RGB channels
    r = (X * 255).astype(np.uint8)
    g = (Y * 255).astype(np.uint8)
    b = ((1 - X) * 255).astype(np.uint8)
    
    # Combine channels
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = b  # OpenCV uses BGR
    img[:, :, 1] = g
    img[:, :, 2] = r
    
    return img

def create_test_mask(width=640, height=480):
    """Create a test mask with a circle and rectangle."""
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Add a circle
    cv2.circle(mask, (width // 3, height // 2), 80, 255, -1)
    
    # Add a rectangle
    cv2.rectangle(mask, (width // 2, height // 4), (width * 3 // 4, height * 3 // 4), 255, -1)
    
    return mask

def test_hsv_visualizer():
    """Test HSV visualization functions."""
    print("Testing HSV visualizer...")
    
    # Create test image and mask
    img = create_test_image()
    mask = create_test_mask()
    
    # Test mask overlay
    overlay_img = OpenCVVisualizer.apply_mask_overlay(img, mask)
    
    # Test centroid drawing
    centroid_img = OpenCVVisualizer.draw_centroid(img, (img.shape[1] // 2, img.shape[0] // 2), radius=10)
    
    # Save results
    cv2.imwrite("hsv_overlay_test.jpg", overlay_img)
    cv2.imwrite("centroid_test.jpg", centroid_img)
    
    print("HSV visualizer tests complete. Output saved to hsv_overlay_test.jpg and centroid_test.jpg")

def test_roi_visualizer():
    """Test ROI visualization functions."""
    print("Testing ROI visualizer...")
    
    # Create test image
    img = create_test_image()
    
    # Create test ROI
    roi = {
        'x': 100,
        'y': 100,
        'width': 200,
        'height': 150,
        'center_x': 200,
        'center_y': 175
    }
    
    # Test ROI drawing
    roi_img = OpenCVVisualizer.draw_roi(img, roi, color=(0, 255, 255))
    
    # Save result
    cv2.imwrite("roi_test.jpg", roi_img)
    
    print("ROI visualizer test complete. Output saved to roi_test.jpg")

def test_hough_visualizer():
    """Test Hough circle visualization functions."""
    print("Testing Hough visualizer...")
    
    # Create test image
    img = create_test_image()
    
    # Create test circles
    circles = [
        (150, 150, 50),  # Main circle
        (300, 200, 30),  # Secondary circle
        (400, 300, 40)   # Secondary circle
    ]
    
    # Test circle drawing
    circles_img = OpenCVVisualizer.draw_circles(img, circles)
    
    # Save result
    cv2.imwrite("circles_test.jpg", circles_img)
    
    print("Hough visualizer test complete. Output saved to circles_test.jpg")

def test_kalman_visualizer():
    """Test Kalman filter visualization functions."""
    print("Testing Kalman visualizer...")
    
    # Create test image
    img = create_test_image()
    
    # Test prediction arrow
    current_pos = (100, 100)
    predicted_pos = (200, 150)
    prediction_img = OpenCVVisualizer.draw_prediction(img, current_pos, predicted_pos)
    
    # Test trajectory
    positions = [
        (50, 50),
        (100, 100),
        (150, 130),
        (200, 150),
        (250, 180),
        (300, 200)
    ]
    trajectory_img = OpenCVVisualizer.draw_trajectory(img, positions)
    
    # Combine prediction and trajectory
    combined_img = OpenCVVisualizer.draw_prediction(trajectory_img, current_pos, predicted_pos)
    
    # Save results
    cv2.imwrite("prediction_test.jpg", prediction_img)
    cv2.imwrite("trajectory_test.jpg", trajectory_img)
    cv2.imwrite("combined_test.jpg", combined_img)
    
    print("Kalman visualizer tests complete. Output saved to prediction_test.jpg, trajectory_test.jpg, and combined_test.jpg")

def test_all():
    """Run all visualization tests."""
    test_hsv_visualizer()
    test_roi_visualizer()
    test_hough_visualizer()
    test_kalman_visualizer()
    
    print("\nAll visualization tests completed successfully.")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Run tests
    test_all() 