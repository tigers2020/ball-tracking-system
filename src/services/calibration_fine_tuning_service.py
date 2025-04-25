#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calibration Fine-Tuning Service.
This module contains services for fine-tuning calibration points using computer vision techniques.
"""

import logging
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional, Any
import time

from src.services.roi_cropper import crop_roi, crop_roi_with_padding, adjust_point_from_roi
from src.services.skeletonizer import skeletonize_roi, enhance_intersections
from src.services.intersection_finder import find_and_sort_intersections

logger = logging.getLogger(__name__)


class CalibrationFineTuningService:
    """
    Service for fine-tuning calibration points using computer vision techniques.
    Extracts regions of interest (ROIs) around calibration points, skeletonizes them, 
    finds intersections, and identifies the best intersection points.
    """
    
    def __init__(self, debug_mode: bool = False):
        """
        Initialize the calibration fine-tuning service.
        
        Args:
            debug_mode (bool): If True, intermediate processing results will be saved
        """
        self.debug_mode = debug_mode
        
    def _save_debug_image(self, image: np.ndarray, name: str):
        """
        Save an image for debugging purposes if debug_mode is enabled.
        
        Args:
            image (np.ndarray): Image to save
            name (str): Image name
        """
        if self.debug_mode:
            try:
                timestamp = int(time.time())
                filename = f"debug_{name}_{timestamp}.png"
                cv2.imwrite(filename, image)
                logger.debug(f"Saved debug image: {filename}")
            except Exception as e:
                logger.error(f"Failed to save debug image: {e}")
    
    def fine_tune_points(self, 
                       side: str, 
                       image: np.ndarray, 
                       points: List[Tuple[float, float]]) -> Dict[int, Dict[str, Any]]:
        """
        Fine-tune a list of points for a specific side using computer vision.
        
        Args:
            side (str): 'left' or 'right' to identify which side the points belong to
            image (np.ndarray): Image for the specified side
            points (List[Tuple[float, float]]): List of points to fine-tune as (x, y) coordinates
            
        Returns:
            Dict[int, Dict[str, Any]]: Dictionary mapping point index to fine-tuned information:
                {
                    point_index: {
                        'original': (original_x, original_y),
                        'adjusted': (adjusted_x, adjusted_y),
                        'distance': adjustment_distance,
                        'success': True/False
                    }
                }
        """
        # Check if inputs are valid
        if image is None or len(points) == 0:
            logger.error("Invalid inputs for fine-tuning points")
            return {}
        
        # Get image dimensions
        img_height, img_width = image.shape[:2]
        
        # Calculate dynamic ROI radius based on image size (approximately 2.5% of image width)
        roi_radius = max(int(img_width * 0.025), 15)  # Min 15 pixels
        logger.info(f"Using dynamic ROI radius of {roi_radius} pixels for {side} image ({img_width}x{img_height})")
        
        # Dictionary to store results
        results = {}
        
        # Process each point
        for index, point in enumerate(points):
            try:
                # Store original point
                results[index] = {
                    'original': point,
                    'adjusted': point,  # Default to original if fine-tuning fails
                    'distance': 0.0,
                    'success': False
                }
                
                # Skip if point is None or invalid
                if point is None or not isinstance(point, tuple) or len(point) < 2:
                    logger.warning(f"Skipping invalid point at index {index}: {point}")
                    continue
                
                # Extract ROI around the point (using model coordinates - original pixels)
                roi = crop_roi(image, point, radius=roi_radius)
                
                if roi is None:
                    logger.warning(f"Failed to crop ROI for {side} point {index}")
                    continue
                
                # Save ROI for debugging if enabled
                self._save_debug_image(roi, f"{side}_roi_{index}")
                
                # Skeletonize ROI
                skeleton = skeletonize_roi(roi)
                
                if skeleton is None or skeleton.size == 0:
                    logger.warning(f"Failed to skeletonize ROI for {side} point {index}")
                    continue
                    
                # Save skeleton for debugging
                self._save_debug_image(skeleton, f"{side}_skeleton_{index}")
                
                # Enhance intersections to make them more distinct
                enhanced_skeleton = enhance_intersections(skeleton)
                self._save_debug_image(enhanced_skeleton, f"{side}_enhanced_{index}")
                
                # Find intersections in skeletonized ROI
                # We'll use the ROI center (half of width/height) as the origin reference 
                # for sorting intersections by proximity
                roi_height, roi_width = roi.shape[:2]
                roi_center = (roi_width // 2, roi_height // 2)
                
                intersections = find_and_sort_intersections(enhanced_skeleton, roi_center, max_points=5)
                
                # If intersections found, use the closest one
                if intersections:
                    # Get the closest intersection (the first in the sorted list)
                    best_x, best_y = intersections[0]
                    
                    # Create a visualization of the intersections for debugging
                    if self.debug_mode:
                        viz = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
                        # Draw all intersections
                        for i, (ix, iy) in enumerate(intersections):
                            color = (0, 0, 255) if i == 0 else (0, 255, 0)
                            cv2.circle(viz, (ix, iy), 2, color, -1)
                        self._save_debug_image(viz, f"{side}_intersections_{index}")
                    
                    # Calculate the offset to convert ROI coordinates to image coordinates
                    roi_with_padding, (offset_x, offset_y) = crop_roi_with_padding(image, point, radius=roi_radius)
                    
                    # Adjust intersection coordinates to image coordinates (original pixel space)
                    adjusted_x = best_x + offset_x
                    adjusted_y = best_y + offset_y
                    
                    # Calculate adjustment distance for logging
                    adjustment_distance = ((adjusted_x - point[0])**2 + (adjusted_y - point[1])**2)**0.5
                    
                    # Only accept adjustment if the distance is reasonable (not too far from original)
                    max_adjustment = roi_radius * 0.75
                    if adjustment_distance > max_adjustment:
                        logger.warning(f"Adjustment distance too large for {side} point {index}: "
                                      f"{adjustment_distance:.2f} > {max_adjustment:.2f} pixels, using original point")
                        continue
                    
                    # Even if adjustment_distance is 0, consider it a success
                    # This happens when the point is already at an intersection
                    logger.info(f"Fine-tuned {side} point {index} from {point} to ({adjusted_x:.1f}, {adjusted_y:.1f}), "
                               f"adjustment distance: {adjustment_distance:.2f} pixels")
                    
                    # Update result
                    results[index] = {
                        'original': point,
                        'adjusted': (adjusted_x, adjusted_y),
                        'distance': adjustment_distance,
                        'success': True
                    }
                else:
                    logger.warning(f"No intersections found for {side} point {index}")
                
            except Exception as e:
                logger.error(f"Error fine-tuning {side} point {index}: {e}")
                # Continue to next point rather than aborting
                continue
                
        logger.info(f"Completed fine-tuning {len(points)} {side} points")
        return results
    
    def fine_tune_calibration_points(self, 
                                   left_image: Optional[np.ndarray], 
                                   right_image: Optional[np.ndarray],
                                   left_points: List[Tuple[float, float]],
                                   right_points: List[Tuple[float, float]]) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """
        Fine-tune calibration points for both left and right images.
        
        Args:
            left_image (Optional[np.ndarray]): Left image
            right_image (Optional[np.ndarray]): Right image
            left_points (List[Tuple[float, float]]): List of left points as (x, y) coordinates
            right_points (List[Tuple[float, float]]): List of right points as (x, y) coordinates
            
        Returns:
            Dict[str, Dict[int, Dict[str, Any]]]: Dictionary with results for both sides:
                {
                    'left': {point_index: {point_data}},
                    'right': {point_index: {point_data}}
                }
        """
        results = {
            'left': {},
            'right': {}
        }
        
        # Process left points if image is available
        if left_image is not None and len(left_points) > 0:
            results['left'] = self.fine_tune_points('left', left_image, left_points)
        
        # Process right points if image is available
        if right_image is not None and len(right_points) > 0:
            results['right'] = self.fine_tune_points('right', right_image, right_points)
        
        return results
        
    def set_debug_mode(self, enabled: bool) -> None:
        """
        Enable or disable debug mode.
        
        Args:
            enabled (bool): True to enable debug mode, False to disable
        """
        self.debug_mode = enabled
        logger.info(f"Debug mode {'enabled' if enabled else 'disabled'}")
        
    def get_success_rate(self, results: Dict[str, Dict[int, Dict[str, Any]]]) -> Dict[str, float]:
        """
        Calculate the success rate for fine-tuning on both sides.
        
        Args:
            results (Dict[str, Dict[int, Dict[str, Any]]]): Results from fine_tune_calibration_points
            
        Returns:
            Dict[str, float]: Success rate for each side (0.0 to 1.0)
        """
        success_rates = {}
        
        for side in ['left', 'right']:
            side_results = results.get(side, {})
            if not side_results:
                success_rates[side] = 0.0
                continue
                
            success_count = sum(1 for res in side_results.values() if res.get('success', False))
            total_count = len(side_results)
            
            success_rates[side] = success_count / total_count if total_count > 0 else 0.0
            
        # Overall success rate
        all_results = list(results.get('left', {}).values()) + list(results.get('right', {}).values())
        if all_results:
            success_count = sum(1 for res in all_results if res.get('success', False))
            success_rates['overall'] = success_count / len(all_results)
        else:
            success_rates['overall'] = 0.0
            
        return success_rates 