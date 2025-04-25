#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Thread-safe wrapper for BallTrackingController.
This module provides a thread-safe wrapper around the BallTrackingController class.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import time

from PySide6.QtCore import QObject, Signal, QMutex, QMutexLocker

from src.controllers.ball_tracking_controller import BallTrackingController
from src.controllers.ball_tracking_logger import BallTrackingXMLLogger


class ThreadSafeBallTracker(QObject):
    """
    Thread-safe wrapper around BallTrackingController.
    Provides the same interface but with thread safety via mutex locking.
    Also includes performance monitoring.
    """
    
    # Forward all signals from the wrapped controller
    mask_updated = Signal(np.ndarray, np.ndarray, dict)
    roi_updated = Signal(dict, dict)
    detection_updated = Signal(int, float, tuple, tuple)  # frame_idx, detection_rate, left_coords, right_coords
    circles_processed = Signal(np.ndarray, np.ndarray)
    
    def __init__(self, controller: Optional[BallTrackingController] = None):
        """
        Initialize thread-safe ball tracker.
        
        Args:
            controller: BallTrackingController instance or None to create a new one
        """
        super(ThreadSafeBallTracker, self).__init__()
        
        # Create controller if not provided
        self.controller = controller if controller is not None else BallTrackingController()
        
        # Create mutex for thread safety
        self.mutex = QMutex()
        
        # Connect signals from controller to our signals
        self.controller.mask_updated.connect(self.mask_updated)
        self.controller.roi_updated.connect(self.roi_updated)
        self.controller.detection_updated.connect(self.detection_updated)
        self.controller.circles_processed.connect(self.circles_processed)
        
        # Performance monitoring
        self.performance_stats = {
            "last_processing_time": 0.0,
            "avg_processing_time": 0.0,
            "max_processing_time": 0.0,
            "min_processing_time": float('inf'),
            "total_frames_processed": 0
        }
        
        # Create XML logger
        self.xml_logger = BallTrackingXMLLogger(self.controller)
    
    def _update_performance_stats(self, processing_time: float):
        """
        Update performance statistics.
        
        Args:
            processing_time: Processing time for the current frame in seconds
        """
        with QMutexLocker(self.mutex):
            # Update performance stats
            self.performance_stats["last_processing_time"] = processing_time
            self.performance_stats["total_frames_processed"] += 1
            
            # Update running average
            n = self.performance_stats["total_frames_processed"]
            prev_avg = self.performance_stats["avg_processing_time"]
            self.performance_stats["avg_processing_time"] = prev_avg + (processing_time - prev_avg) / n
            
            # Update min/max
            self.performance_stats["max_processing_time"] = max(
                self.performance_stats["max_processing_time"], processing_time)
            
            if processing_time > 0:  # Avoid division by zero or negative times
                self.performance_stats["min_processing_time"] = min(
                    self.performance_stats["min_processing_time"], processing_time)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dict containing performance statistics
        """
        with QMutexLocker(self.mutex):
            return self.performance_stats.copy()
    
    # Forward all BallTrackingController methods with thread safety
    
    def set_hsv_values(self, hsv_values):
        """Thread-safe wrapper for BallTrackingController.set_hsv_values"""
        with QMutexLocker(self.mutex):
            self.controller.set_hsv_values(hsv_values)
    
    def set_images(self, left_image, right_image):
        """Thread-safe wrapper for BallTrackingController.set_images"""
        start_time = time.time()
        
        with QMutexLocker(self.mutex):
            self.controller.set_images(left_image, right_image)
        
        # Update performance stats
        processing_time = time.time() - start_time
        self._update_performance_stats(processing_time)
    
    def enable(self, enabled=True):
        """Thread-safe wrapper for BallTrackingController.enable"""
        with QMutexLocker(self.mutex):
            self.controller.enable(enabled)
    
    def get_current_masks(self):
        """Thread-safe wrapper for BallTrackingController.get_current_masks"""
        with QMutexLocker(self.mutex):
            return self.controller.get_current_masks()
    
    def get_hsv_values(self):
        """Thread-safe wrapper for BallTrackingController.get_hsv_values"""
        with QMutexLocker(self.mutex):
            return self.controller.get_hsv_values()
    
    def set_roi_settings(self, roi_settings):
        """Thread-safe wrapper for BallTrackingController.set_roi_settings"""
        with QMutexLocker(self.mutex):
            # Validate ROI settings against image dimensions if available
            image_width = None
            image_height = None
            
            if self.controller.left_image is not None:
                image_height, image_width = self.controller.left_image.shape[:2]
            elif self.controller.right_image is not None:
                image_height, image_width = self.controller.right_image.shape[:2]
                
            # Use validated settings
            validated_settings = self.controller.config_manager.validate_roi(
                roi_settings, image_width, image_height)
            
            self.controller.set_roi_settings(validated_settings)
            return validated_settings
    
    def get_roi_settings(self):
        """Thread-safe wrapper for BallTrackingController.get_roi_settings"""
        with QMutexLocker(self.mutex):
            return self.controller.get_roi_settings()
    
    def get_current_rois(self):
        """Thread-safe wrapper for BallTrackingController.get_current_rois"""
        with QMutexLocker(self.mutex):
            return self.controller.get_current_rois()
    
    def get_cropped_roi_images(self):
        """Thread-safe wrapper for BallTrackingController.get_cropped_roi_images"""
        with QMutexLocker(self.mutex):
            return self.controller.get_cropped_roi_images()
    
    def detect_circles_in_roi(self):
        """Thread-safe wrapper for BallTrackingController.detect_circles_in_roi"""
        start_time = time.time()
        
        with QMutexLocker(self.mutex):
            result = self.controller.detect_circles_in_roi()
        
        # Update performance stats
        processing_time = time.time() - start_time
        self._update_performance_stats(processing_time)
        
        return result
    
    def get_detection_rate(self):
        """Thread-safe wrapper for BallTrackingController.get_detection_rate"""
        with QMutexLocker(self.mutex):
            return self.controller.get_detection_rate()
    
    def get_predictions(self):
        """Thread-safe wrapper for BallTrackingController.get_predictions"""
        with QMutexLocker(self.mutex):
            return self.controller.get_predictions()
    
    def get_latest_coordinates(self):
        """Thread-safe wrapper for BallTrackingController.get_latest_coordinates"""
        with QMutexLocker(self.mutex):
            return self.controller.get_latest_coordinates()
    
    def clear_coordinate_history(self):
        """Thread-safe wrapper for BallTrackingController.clear_coordinate_history"""
        with QMutexLocker(self.mutex):
            self.controller.clear_coordinate_history()
    
    def reset_tracking(self):
        """Thread-safe wrapper for BallTrackingController.reset_tracking"""
        with QMutexLocker(self.mutex):
            self.controller.reset_tracking()
    
    def get_coordinate_history(self, side="both", count=None):
        """Thread-safe wrapper for BallTrackingController.get_coordinate_history"""
        with QMutexLocker(self.mutex):
            return self.controller.get_coordinate_history(side, count)
    
    def save_coordinate_history(self, filename):
        """Thread-safe wrapper for BallTrackingController.save_coordinate_history"""
        with QMutexLocker(self.mutex):
            return self.controller.save_coordinate_history(filename)
    
    def save_tracking_data_to_json(self, folder_path=None, filename=None):
        """Thread-safe wrapper for BallTrackingController.save_tracking_data_to_json"""
        with QMutexLocker(self.mutex):
            return self.controller.save_tracking_data_to_json(folder_path, filename)
    
    def save_tracking_data_for_frame(self, frame_number, folder_path=None):
        """
        Save tracking data for the current frame.
        Uses XML logger instead of JSON.
        
        Args:
            frame_number: Frame number
            folder_path: Output folder path
            
        Returns:
            str: Path to the XML file or None if failed
        """
        with QMutexLocker(self.mutex):
            # Use XML logger
            return self.xml_logger.save_tracking_data_for_frame(frame_number, folder_path)
    
    def initialize_xml_tracking(self, folder_name):
        """Thread-safe wrapper for initializing XML tracking"""
        with QMutexLocker(self.mutex):
            self.xml_logger.logger.start_session(folder_name)
            return True
    
    def save_xml_tracking_data(self, folder_path=None):
        """Thread-safe wrapper for finalizing XML tracking"""
        with QMutexLocker(self.mutex):
            # Add statistics and close the XML logger
            self.xml_logger.save_statistics()
            return self.xml_logger.logger.close() 