#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the fine-tune functionality.
This directly tests the _fine_tune_points method of the CalibrationController.
"""

import sys
import numpy as np
import cv2
import logging

from src.models.calibration_model import CalibrationModel
from src.services.roi_cropper import crop_roi, crop_roi_with_padding
from src.services.skeletonizer import skeletonize_roi
from src.services.intersection_finder import find_and_sort_intersections

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_test_image(width=400, height=300):
    """Create a test image with a grid pattern for testing."""
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Draw horizontal lines
    for y in range(50, height, 50):
        cv2.line(image, (0, y), (width, y), 255, 1)
    
    # Draw vertical lines
    for x in range(50, width, 50):
        cv2.line(image, (x, 0), (x, height), 255, 1)
    
    return image

def test_fine_tune_point(image, point, radius=25.0):
    """
    Test fine-tuning a single calibration point.
    
    Args:
        image (np.ndarray): Test image
        point (tuple): Initial point (x, y)
        radius (float): ROI radius
        
    Returns:
        tuple: Fine-tuned point or None if fine-tuning failed
    """
    try:
        logger.info(f"Testing fine-tuning point {point}")
        
        # Extract ROI around the point
        roi = crop_roi(image, point, radius=radius)
        
        if roi is None:
            logger.warning(f"Failed to crop ROI for point {point}")
            return None
        
        # Save the ROI image for inspection
        cv2.imwrite('test_roi.png', roi)
        logger.info(f"Saved ROI image to test_roi.png")
        
        # Skeletonize ROI
        skeleton = skeletonize_roi(roi)
        
        # Save the skeleton image for inspection
        cv2.imwrite('test_skeleton.png', skeleton)
        logger.info(f"Saved skeleton image to test_skeleton.png")
        
        # Find intersections in skeletonized ROI
        roi_height, roi_width = roi.shape[:2]
        roi_center = (roi_width // 2, roi_height // 2)
        
        intersections = find_and_sort_intersections(skeleton, roi_center, max_points=3)
        logger.info(f"Found {len(intersections)} intersections")
        
        # If intersections found, use the closest one
        if intersections:
            # Get the closest intersection (the first in the sorted list)
            best_x, best_y = intersections[0]
            logger.info(f"Best intersection: ({best_x}, {best_y})")
            
            # Calculate the offset to convert ROI coordinates to image coordinates
            roi_with_padding, (offset_x, offset_y) = crop_roi_with_padding(image, point, radius=radius)
            
            # Adjust intersection coordinates to image coordinates
            adjusted_x = best_x + offset_x
            adjusted_y = best_y + offset_y
            
            logger.info(f"Adjusted coordinates: ({adjusted_x}, {adjusted_y})")
            
            # Calculate improvement
            dx = adjusted_x - point[0]
            dy = adjusted_y - point[1]
            distance = np.sqrt(dx*dx + dy*dy)
            
            logger.info(f"Adjustment distance: {distance:.2f} pixels")
            
            return (adjusted_x, adjusted_y)
        else:
            logger.warning("No intersections found")
            return None
            
    except Exception as e:
        logger.error(f"Error fine-tuning point: {e}")
        return None

def main():
    """Main test function."""
    # Create a test image
    image = create_test_image(400, 300)
    
    # Save the test image for inspection
    cv2.imwrite('test_image.png', image)
    logger.info("Saved test image to test_image.png")
    
    # Define test points (slightly off from the actual intersections)
    test_points = [
        (52, 52),    # Slightly off from intersection at (50, 50)
        (150, 102),  # Slightly off from intersection at (150, 100)
        (248, 198)   # Slightly off from intersection at (250, 200)
    ]
    
    # Test fine-tuning for each point
    for i, point in enumerate(test_points):
        logger.info(f"\nTesting point {i+1}: {point}")
        result = test_fine_tune_point(image, point)
        
        if result:
            logger.info(f"Original point: {point}")
            logger.info(f"Fine-tuned point: {result}")
            
            # Draw the points on the image (original in red, fine-tuned in green)
            test_image_color = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
            cv2.circle(test_image_color, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)  # Red
            cv2.circle(test_image_color, (int(result[0]), int(result[1])), 5, (0, 255, 0), -1)  # Green
            
            # Save the result image
            cv2.imwrite(f'test_result_{i+1}.png', test_image_color)
            logger.info(f"Saved result image to test_result_{i+1}.png")
        else:
            logger.warning(f"Fine-tuning failed for point {point}")

if __name__ == '__main__':
    main() 