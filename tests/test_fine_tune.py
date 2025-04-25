#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tuning test script.
This script demonstrates the calibration fine-tuning process with visualizations.
"""

import os
import cv2
import numpy as np
import logging

from src.services.roi_cropper import crop_roi, crop_roi_with_padding
from src.services.skeletonizer import skeletonize_roi, enhance_intersections
from src.services.intersection_finder import find_and_sort_intersections
from src.services.calibration_fine_tuning_service import CalibrationFineTuningService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_image(width=800, height=600, grid_spacing=100):
    """Create a test image with a grid pattern."""
    # Create an empty image
    image = np.ones((height, width), dtype=np.uint8) * 255
    
    # Draw grid lines
    for i in range(0, width, grid_spacing):
        cv2.line(image, (i, 0), (i, height), (0, 0, 0), 2)
    
    for i in range(0, height, grid_spacing):
        cv2.line(image, (0, i), (width, i), (0, 0, 0), 2)
    
    # Save the image
    cv2.imwrite('test_image.png', image)
    
    return image

def test_fine_tune_point():
    """Test fine-tuning a point."""
    # Create a test image
    image = create_test_image()
    
    # Create test points that are slightly off from grid intersections
    # The points are not exactly at intersections to simulate real-world scenarios
    test_points = [
        (103, 102),  # slightly off from (100, 100)
        (198, 199),  # slightly off from (200, 200)
        (302, 297),  # slightly off from (300, 300)
        (398, 401)   # slightly off from (400, 400)
    ]
    
    # Initialize the ROI cropper
    roi_radius = 20  # pixels
    
    # For each test point
    for i, point in enumerate(test_points):
        print(f"Testing point {i+1}: {point}")
        
        # Extract ROI around the point
        roi, offset_x, offset_y = crop_roi(image, point, roi_radius)
        
        # Skeletonize the ROI
        skeleton = skeletonize_roi(roi)
        
        # Find intersections in the skeleton
        intersections = find_and_sort_intersections(skeleton, (roi.shape[1] // 2, roi.shape[0] // 2), max_points=5)
        
        # If intersections were found
        if len(intersections) > 0:
            # Get the best intersection (closest to the center of the ROI)
            center_x, center_y = roi.shape[1] // 2, roi.shape[0] // 2
            best_intersection = min(intersections, key=lambda p: (p[0] - center_x)**2 + (p[1] - center_y)**2)
            
            # Calculate the adjusted coordinates in the original image
            adjusted_x = best_intersection[0] + offset_x
            adjusted_y = best_intersection[1] + offset_y
            
            # Calculate adjustment distance
            adjustment_distance = ((adjusted_x - point[0])**2 + (adjusted_y - point[1])**2)**0.5
            
            print(f"Original: {point}, Adjusted: ({adjusted_x}, {adjusted_y}), "
                  f"Adjustment distance: {adjustment_distance:.2f} pixels")
            
            # Visualize the results
            result_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
            
            # Draw original point (red)
            cv2.circle(result_image, point, 5, (0, 0, 255), -1)
            
            # Draw adjusted point (green)
            cv2.circle(result_image, (int(adjusted_x), int(adjusted_y)), 5, (0, 255, 0), -1)
            
            # Draw ROI (blue)
            cv2.rectangle(result_image, 
                          (point[0] - roi_radius, point[1] - roi_radius),
                          (point[0] + roi_radius, point[1] + roi_radius),
                          (255, 0, 0), 2)
            
            # Save the result
            cv2.imwrite(f'test_result_{i+1}.png', result_image)
        else:
            print(f"No intersections found for point {point}")

def main():
    """Run the test."""
    print("Testing fine-tuning service...")
    
    # Create a test image
    image = create_test_image()
    
    # Create test points that are slightly off from grid intersections
    test_points = [
        (103, 102),  # slightly off from (100, 100)
        (198, 199),  # slightly off from (200, 200)
        (302, 297),  # slightly off from (300, 300)
        (398, 401)   # slightly off from (400, 400)
    ]
    
    # Initialize the test service
    fine_tuning_service = CalibrationFineTuningService(debug_mode=True)
    
    # Dynamic ROI radius based on image size
    roi_radius = int(min(image.shape) * 0.025)  # 2.5% of the smallest dimension
    print(f"Using dynamic ROI radius of {roi_radius} pixels for image size {image.shape}")
    
    # Test the service
    all_results = fine_tuning_service.fine_tune_points('test', image, test_points)
    
    # Print results
    for idx, result in all_results.items():
        if result.get('success', False):
            original = result['original']
            adjusted = result['adjusted']
            distance = result['distance']
            print(f"Point {idx}: Original {original} -> Adjusted {adjusted}, Distance: {distance:.2f} pixels")
        else:
            print(f"Point {idx}: Fine-tuning failed")
    
    print("Test completed successfully.")

if __name__ == "__main__":
    main() 