#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ball Tracking Logger module.
This module provides an XML logger adapter for the BallTrackingController.
"""

import logging
from typing import Dict, Any, Optional, Tuple

from src.utils.xml_logger import XMLLogger, XMLLoggerAdapter


class BallTrackingXMLLogger(XMLLoggerAdapter):
    """
    XML Logger adapter for BallTrackingController.
    Handles conversion of ball tracking data to XML format.
    """
    
    def __init__(self, ball_tracker=None, xml_logger: XMLLogger = None):
        """
        Initialize the ball tracking XML logger.
        
        Args:
            ball_tracker: Reference to the BallTrackingController
            xml_logger: XMLLogger instance to use, or create a new one if None
        """
        super().__init__(xml_logger)
        self.ball_tracker = ball_tracker
    
    def set_ball_tracker(self, ball_tracker):
        """
        Set the reference to the BallTrackingController.
        
        Args:
            ball_tracker: Reference to the BallTrackingController
        """
        self.ball_tracker = ball_tracker
    
    def _build_frame_data(self, frame_number: int) -> Dict[str, Any]:
        """
        Build frame data dictionary from the current state of the ball tracker.
        
        Args:
            frame_number: Frame number
            
        Returns:
            Dict containing frame data
        """
        if self.ball_tracker is None:
            logging.warning("No ball tracker reference set. Returning empty frame data.")
            return super()._build_frame_data(frame_number)
        
        # Create frame data structure
        frame_data = {
            "tracking_active": self.ball_tracker.detection_stats["is_tracking"],
            "left": {},
            "right": {}
        }
        
        # Process left side
        if self.ball_tracker.left_mask is not None:
            # Get HSV mask centroid
            left_hsv_center = self.ball_tracker._compute_mask_centroid(self.ball_tracker.left_mask)
            if left_hsv_center:
                frame_data["left"]["hsv_center"] = {
                    "x": float(left_hsv_center[0]),
                    "y": float(left_hsv_center[1])
                }
            
            # Get latest Hough circle center if available
            if self.ball_tracker.left_circles:
                frame_data["left"]["hough_center"] = {
                    "x": float(self.ball_tracker.left_circles[0][0]),
                    "y": float(self.ball_tracker.left_circles[0][1]),
                    "radius": float(self.ball_tracker.left_circles[0][2])
                }
            
            # Get Kalman prediction if available
            if self.ball_tracker.prediction["left"] is not None:
                px, py, vx, vy = self.ball_tracker.prediction["left"]
                frame_data["left"]["kalman_prediction"] = {
                    "x": float(px),
                    "y": float(py),
                    "vx": float(vx),
                    "vy": float(vy)
                }
                
            # Get fused coordinates
            left_hsv_center = self.ball_tracker._compute_mask_centroid(self.ball_tracker.left_mask) if self.ball_tracker.left_mask is not None else None
            left_hough_center = (self.ball_tracker.left_circles[0][0], self.ball_tracker.left_circles[0][1]) if self.ball_tracker.left_circles else None
            
            # Get coordinates for fusion
            coords = []
            if left_hsv_center:
                coords.append(left_hsv_center)
            if left_hough_center:
                coords.append(left_hough_center)
                
            # Add kalman if not first frame and tracking is active
            if (self.ball_tracker.prediction["left"] is not None and 
                self.ball_tracker.detection_stats["is_tracking"] and 
                self.ball_tracker.detection_stats["total_frames"] > 0):
                coords.append((self.ball_tracker.prediction["left"][0], self.ball_tracker.prediction["left"][1]))
                
            # Compute fused coordinates
            from src.utils.coord_utils import fuse_coordinates
            left_fused = fuse_coordinates(coords)
            
            if left_fused:
                frame_data["left"]["fused_center"] = {
                    "x": float(left_fused[0]),
                    "y": float(left_fused[1])
                }
        
        # Process right side (similar to left)
        if self.ball_tracker.right_mask is not None:
            # Get HSV mask centroid
            right_hsv_center = self.ball_tracker._compute_mask_centroid(self.ball_tracker.right_mask)
            if right_hsv_center:
                frame_data["right"]["hsv_center"] = {
                    "x": float(right_hsv_center[0]),
                    "y": float(right_hsv_center[1])
                }
            
            # Get latest Hough circle center if available
            if self.ball_tracker.right_circles:
                frame_data["right"]["hough_center"] = {
                    "x": float(self.ball_tracker.right_circles[0][0]),
                    "y": float(self.ball_tracker.right_circles[0][1]),
                    "radius": float(self.ball_tracker.right_circles[0][2])
                }
            
            # Get Kalman prediction if available
            if self.ball_tracker.prediction["right"] is not None:
                px, py, vx, vy = self.ball_tracker.prediction["right"]
                frame_data["right"]["kalman_prediction"] = {
                    "x": float(px),
                    "y": float(py),
                    "vx": float(vx),
                    "vy": float(vy)
                }
                
            # Get fused coordinates
            right_hsv_center = self.ball_tracker._compute_mask_centroid(self.ball_tracker.right_mask) if self.ball_tracker.right_mask is not None else None
            right_hough_center = (self.ball_tracker.right_circles[0][0], self.ball_tracker.right_circles[0][1]) if self.ball_tracker.right_circles else None
            
            # Get coordinates for fusion
            coords = []
            if right_hsv_center:
                coords.append(right_hsv_center)
            if right_hough_center:
                coords.append(right_hough_center)
                
            # Add kalman if not first frame and tracking is active
            if (self.ball_tracker.prediction["right"] is not None and 
                self.ball_tracker.detection_stats["is_tracking"] and 
                self.ball_tracker.detection_stats["total_frames"] > 0):
                coords.append((self.ball_tracker.prediction["right"][0], self.ball_tracker.prediction["right"][1]))
                
            # Compute fused coordinates
            from src.utils.coord_utils import fuse_coordinates
            right_fused = fuse_coordinates(coords)
            
            if right_fused:
                frame_data["right"]["fused_center"] = {
                    "x": float(right_fused[0]),
                    "y": float(right_fused[1])
                }
        
        return frame_data
        
    def save_statistics(self):
        """
        Save summary statistics to the XML.
        
        Returns:
            bool: Success or failure
        """
        if self.ball_tracker is None:
            logging.warning("No ball tracker reference set. Cannot save statistics.")
            return False
            
        stats = {
            "detection_rate": self.ball_tracker.get_detection_rate(),
            "frames_total": self.ball_tracker.detection_stats["total_frames"],
            "detections_count": self.ball_tracker.detection_stats["detection_count"],
            "tracking_active": str(self.ball_tracker.detection_stats["is_tracking"])
        }
        
        return self.logger.add_statistics(stats)
        
    def close(self):
        """
        Close the session and save final statistics.
        
        Returns:
            bool: Success or failure
        """
        self.save_statistics()
        return self.logger.close() 