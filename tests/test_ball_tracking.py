#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ball Tracking Test Module.
This module tests the ball tracking functionality.
"""

import os
import sys
import cv2
import numpy as np
import time
import logging

# Add parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from src.controllers.ball_tracking_controller import BallTrackingController
from src.views.info_view import InfoView
from src.views.image_view_widget import StereoImageViewWidget, ImageViewWidget

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BallTrackingTest:
    """Test class for ball tracking functionality."""
    
    def __init__(self):
        """Initialize the test."""
        self.app = QApplication(sys.argv)
        
        # Create controllers
        self.ball_tracking_controller = BallTrackingController()
        
        # Create views
        self.stereo_view = StereoImageViewWidget()
        self.info_view = InfoView()
        
        # Connect views to controllers
        self.info_view.connect_tracking_controller(self.ball_tracking_controller)
        
        # Enable ball tracking
        self.ball_tracking_controller.enable(True)
        
        # Show views
        self.stereo_view.show()
        self.info_view.show()
        
        # Setup timer for test sequence
        self.test_timer = QTimer()
        self.test_timer.timeout.connect(self.run_next_test)
        self.current_test = 0
        self.tests = [
            self.test_roi_extraction,
            self.test_circle_detection,
            self.test_coordinate_tracking,
            self.test_kalman_prediction,
            self.test_detection_rate,
            self.test_info_view_integration
        ]
    
    def run_tests(self):
        """Run all tests sequentially."""
        logging.info("Starting ball tracking tests...")
        self.test_timer.start(100)  # Start first test after 100ms
        return self.app.exec()
    
    def run_next_test(self):
        """Run the next test in the sequence."""
        self.test_timer.stop()
        
        if self.current_test < len(self.tests):
            test_func = self.tests[self.current_test]
            logging.info(f"Running test {self.current_test + 1}/{len(self.tests)}: {test_func.__name__}")
            test_func()
            self.current_test += 1
            if self.current_test < len(self.tests):
                self.test_timer.start(1000)  # 1 second between tests
            else:
                logging.info("All tests completed!")
                self.app.quit()
        else:
            self.app.quit()
    
    def load_test_images(self):
        """Load test images for processing."""
        try:
            # Try to load sample images
            left_img = cv2.imread("test_data/left_sample.png")
            right_img = cv2.imread("test_data/right_sample.png")
            
            # If images don't exist, create synthetic test images
            if left_img is None or right_img is None:
                logging.info("Creating synthetic test images")
                # Create black images
                left_img = np.zeros((480, 640, 3), dtype=np.uint8)
                right_img = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Draw a red circle on each image
                cv2.circle(left_img, (320, 240), 30, (0, 0, 255), -1)
                cv2.circle(right_img, (350, 220), 35, (0, 0, 255), -1)
            
            return left_img, right_img
            
        except Exception as e:
            logging.error(f"Error loading test images: {e}")
            return None, None
    
    def test_roi_extraction(self):
        """Test ROI extraction and cropping."""
        logging.info("Testing ROI extraction and cropping...")
        
        # Load test images
        left_img, right_img = self.load_test_images()
        
        # Set images to controller
        self.ball_tracking_controller.set_images(left_img, right_img)
        
        # Display original images
        self.stereo_view.set_images(left_img, right_img)
        
        # Get ROI settings
        roi_settings = self.ball_tracking_controller.get_roi_settings()
        logging.info(f"ROI settings: {roi_settings}")
        
        # Get ROIs
        left_roi, right_roi = self.ball_tracking_controller.get_current_rois()
        logging.info(f"Left ROI: {left_roi}")
        logging.info(f"Right ROI: {right_roi}")
        
        # Get cropped ROI images
        left_cropped, right_cropped = self.ball_tracking_controller.get_cropped_roi_images()
        
        # Display cropped ROI images
        self.stereo_view.display_roi_images(left_cropped, right_cropped)
        
        # Save ROI images for verification
        if left_cropped is not None and right_cropped is not None:
            os.makedirs("test_results", exist_ok=True)
            cv2.imwrite("test_results/left_roi.png", left_cropped)
            cv2.imwrite("test_results/right_roi.png", right_cropped)
            logging.info("ROI images saved to test_results folder")
    
    def test_circle_detection(self):
        """Test circle detection using Hough transform."""
        logging.info("Testing circle detection...")
        
        # Load test images
        left_img, right_img = self.load_test_images()
        
        # Set images to controller
        self.ball_tracking_controller.set_images(left_img, right_img)
        
        # Detect circles in ROIs
        left_result, right_result = self.ball_tracking_controller.detect_circles_in_roi()
        
        # Extract results
        left_processed, left_circles = left_result
        right_processed, right_circles = right_result
        
        # Log circle detection results
        if left_circles:
            logging.info(f"Left circles detected: {len(left_circles)}")
            for i, circle in enumerate(left_circles):
                logging.info(f"  Circle {i+1}: x={circle[0]}, y={circle[1]}, r={circle[2]}")
        else:
            logging.warning("No circles detected in left image")
            
        if right_circles:
            logging.info(f"Right circles detected: {len(right_circles)}")
            for i, circle in enumerate(right_circles):
                logging.info(f"  Circle {i+1}: x={circle[0]}, y={circle[1]}, r={circle[2]}")
        else:
            logging.warning("No circles detected in right image")
        
        # Display processed images with circles
        if left_processed is not None and right_processed is not None:
            self.stereo_view.set_images(left_processed, right_processed)
            
            # Save processed images for verification
            os.makedirs("test_results", exist_ok=True)
            cv2.imwrite("test_results/left_circles.png", left_processed)
            cv2.imwrite("test_results/right_circles.png", right_processed)
            logging.info("Circle detection images saved to test_results folder")
    
    def test_coordinate_tracking(self):
        """Test coordinate tracking functionality."""
        logging.info("Testing coordinate tracking...")
        
        # Load test images
        left_img, right_img = self.load_test_images()
        
        # Clear previous coordinate history
        self.ball_tracking_controller.clear_coordinate_history()
        
        # Process multiple frames to build coordinate history
        for i in range(5):
            # Modify images slightly to simulate movement
            left_img_copy = left_img.copy()
            right_img_copy = right_img.copy()
            
            # Move circles
            cv2.circle(left_img_copy, (320 + i*10, 240 - i*5), 30, (0, 0, 0), -1)  # Clear previous
            cv2.circle(left_img_copy, (320 + i*10, 240 - i*5), 30, (0, 0, 255), -1)  # Draw new
            
            cv2.circle(right_img_copy, (350 + i*10, 220 - i*5), 35, (0, 0, 0), -1)  # Clear previous
            cv2.circle(right_img_copy, (350 + i*10, 220 - i*5), 35, (0, 0, 255), -1)  # Draw new
            
            # Set images to controller
            self.ball_tracking_controller.set_images(left_img_copy, right_img_copy)
            
            # Process images (detect circles and update coordinate history)
            self.ball_tracking_controller.detect_circles_in_roi()
            
            # Simulate frame processing delay
            time.sleep(0.1)
        
        # Get coordinate history
        left_history = self.ball_tracking_controller.get_coordinate_history("left")
        right_history = self.ball_tracking_controller.get_coordinate_history("right")
        
        # Log coordinate history
        logging.info(f"Left coordinate history: {len(left_history)} points")
        for i, record in enumerate(left_history):
            logging.info(f"  Point {i+1}: x={record[0]}, y={record[1]}, r={record[2]}, t={record[3]:.2f}")
            
        logging.info(f"Right coordinate history: {len(right_history)} points")
        for i, record in enumerate(right_history):
            logging.info(f"  Point {i+1}: x={record[0]}, y={record[1]}, r={record[2]}, t={record[3]:.2f}")
        
        # Get latest coordinates
        left_coords, right_coords = self.ball_tracking_controller.get_latest_coordinates()
        logging.info(f"Latest left coordinates: {left_coords}")
        logging.info(f"Latest right coordinates: {right_coords}")
    
    def test_kalman_prediction(self):
        """Test Kalman filter prediction."""
        logging.info("Testing Kalman prediction...")
        
        # Load test images
        left_img, right_img = self.load_test_images()
        
        # Clear previous coordinate history
        self.ball_tracking_controller.clear_coordinate_history()
        
        # Process multiple frames with movement to train Kalman filter
        for i in range(5):
            # Modify images slightly to simulate movement
            left_img_copy = left_img.copy()
            right_img_copy = right_img.copy()
            
            # Move circles with consistent velocity
            cv2.circle(left_img_copy, (320 + i*15, 240 - i*8), 30, (0, 0, 0), -1)  # Clear previous
            cv2.circle(left_img_copy, (320 + i*15, 240 - i*8), 30, (0, 0, 255), -1)  # Draw new
            
            cv2.circle(right_img_copy, (350 + i*15, 220 - i*8), 35, (0, 0, 0), -1)  # Clear previous
            cv2.circle(right_img_copy, (350 + i*15, 220 - i*8), 35, (0, 0, 255), -1)  # Draw new
            
            # Set images to controller
            self.ball_tracking_controller.set_images(left_img_copy, right_img_copy)
            
            # Process images (detect circles and update Kalman filter)
            self.ball_tracking_controller.detect_circles_in_roi()
            
            # Simulate frame processing delay
            time.sleep(0.1)
        
        # Get predictions
        left_pred, right_pred = self.ball_tracking_controller.get_predictions()
        
        # Log predictions
        logging.info(f"Left prediction: {left_pred}")
        logging.info(f"Right prediction: {right_pred}")
        
        if left_pred and right_pred:
            # Create an image to visualize predictions
            pred_img_left = left_img.copy()
            pred_img_right = right_img.copy()
            
            # Draw predicted position (blue) and velocity vector (green)
            if left_pred:
                px, py, vx, vy = left_pred
                cv2.circle(pred_img_left, (px, py), 30, (255, 0, 0), 2)  # Predicted position
                cv2.arrowedLine(pred_img_left, (px, py), (px + vx*3, py + vy*3), (0, 255, 0), 2)  # Velocity vector
                
            if right_pred:
                px, py, vx, vy = right_pred
                cv2.circle(pred_img_right, (px, py), 35, (255, 0, 0), 2)  # Predicted position
                cv2.arrowedLine(pred_img_right, (px, py), (px + vx*3, py + vy*3), (0, 255, 0), 2)  # Velocity vector
            
            # Display images with predictions
            self.stereo_view.set_images(pred_img_left, pred_img_right)
            
            # Save prediction images
            os.makedirs("test_results", exist_ok=True)
            cv2.imwrite("test_results/left_prediction.png", pred_img_left)
            cv2.imwrite("test_results/right_prediction.png", pred_img_right)
            logging.info("Prediction images saved to test_results folder")
    
    def test_detection_rate(self):
        """Test detection rate calculation."""
        logging.info("Testing detection rate calculation...")
        
        # Load test images
        left_img, right_img = self.load_test_images()
        
        # Reset tracking statistics
        self.ball_tracking_controller.detection_stats = {
            "first_detection_time": None,
            "detection_count": 0,
            "total_frames": 0,
            "is_tracking": False
        }
        
        # Process multiple frames with varied detection success
        for i in range(10):
            # Modify images to simulate movement
            left_img_copy = left_img.copy()
            right_img_copy = right_img.copy()
            
            # Every 3rd frame, simulate a missed detection in right image
            if i % 3 == 2:
                # Only draw circle in left image
                cv2.circle(left_img_copy, (320 + i*10, 240), 30, (0, 0, 255), -1)
                # Black out right image
                right_img_copy = np.zeros_like(right_img)
            else:
                # Draw circles in both images
                cv2.circle(left_img_copy, (320 + i*10, 240), 30, (0, 0, 255), -1)
                cv2.circle(right_img_copy, (350 + i*10, 220), 35, (0, 0, 255), -1)
            
            # Set images to controller
            self.ball_tracking_controller.set_images(left_img_copy, right_img_copy)
            
            # Process images
            self.ball_tracking_controller.detect_circles_in_roi()
            
            # Log current detection rate
            detection_rate = self.ball_tracking_controller.get_detection_rate()
            logging.info(f"Frame {i+1}: Detection rate = {detection_rate:.2%}")
            
            # Simulate frame processing delay
            time.sleep(0.1)
        
        # Final detection rate
        final_rate = self.ball_tracking_controller.get_detection_rate()
        logging.info(f"Final detection rate: {final_rate:.2%}")
        logging.info(f"Detection count: {self.ball_tracking_controller.detection_stats['detection_count']}")
        logging.info(f"Total frames: {self.ball_tracking_controller.detection_stats['total_frames']}")
    
    def test_info_view_integration(self):
        """Test info view integration with controller signals."""
        logging.info("Testing info view integration...")
        
        # Load test images
        left_img, right_img = self.load_test_images()
        
        # Clear previous tracking data
        self.ball_tracking_controller.clear_coordinate_history()
        
        # Reset tracking statistics
        self.ball_tracking_controller.detection_stats = {
            "first_detection_time": None,
            "detection_count": 0,
            "total_frames": 0,
            "is_tracking": False
        }
        
        # Process multiple frames
        for i in range(5):
            # Modify images to simulate movement
            left_img_copy = left_img.copy()
            right_img_copy = right_img.copy()
            
            cv2.circle(left_img_copy, (320 + i*10, 240 - i*5), 30, (0, 0, 255), -1)
            cv2.circle(right_img_copy, (350 + i*10, 220 - i*5), 35, (0, 0, 255), -1)
            
            # Set images to controller
            self.ball_tracking_controller.set_images(left_img_copy, right_img_copy)
            
            # Process images
            self.ball_tracking_controller.detect_circles_in_roi()
            
            # Display the processed images
            self.stereo_view.set_images(left_img_copy, right_img_copy)
            
            # The info view should update automatically via signal
            
            # Simulate frame processing delay
            time.sleep(0.5)  # Longer delay to observe info view updates
        
        # Log current info view state
        logging.info(f"Info view detection rate: {self.info_view.detection_rate:.2%}")
        logging.info(f"Info view left coordinates: {self.info_view.left_pixel_coords}")
        logging.info(f"Info view right coordinates: {self.info_view.right_pixel_coords}")


if __name__ == "__main__":
    test = BallTrackingTest()
    sys.exit(test.run_tests()) 