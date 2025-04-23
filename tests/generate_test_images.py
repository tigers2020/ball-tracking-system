#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate Test Images.
This script generates test images for the ball tracking tests.
"""

import os
import cv2
import numpy as np

def generate_test_images():
    """Generate test images for ball tracking."""
    print("Generating test images...")
    
    # Create output directory
    os.makedirs("test_data", exist_ok=True)
    
    # Create image dimensions
    width, height = 640, 480
    
    # Generate left image
    left_img = np.ones((height, width, 3), dtype=np.uint8) * 240  # Gray background
    
    # Add some noise to make it more realistic
    noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
    left_img = cv2.add(left_img, noise)
    
    # Add a grid pattern
    for i in range(0, width, 50):
        cv2.line(left_img, (i, 0), (i, height), (200, 200, 200), 1)
    for i in range(0, height, 50):
        cv2.line(left_img, (0, i), (width, i), (200, 200, 200), 1)
    
    # Draw a red ball
    ball_center = (320, 240)
    ball_radius = 30
    cv2.circle(left_img, ball_center, ball_radius, (0, 0, 255), -1)
    
    # Add highlight to make it look more realistic
    highlight_center = (ball_center[0] - 10, ball_center[1] - 10)
    highlight_radius = 10
    cv2.circle(left_img, highlight_center, highlight_radius, (50, 50, 255), -1)
    
    # Generate right image (similar but with slight differences)
    right_img = np.ones((height, width, 3), dtype=np.uint8) * 240  # Gray background
    
    # Add some noise to make it more realistic
    noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
    right_img = cv2.add(right_img, noise)
    
    # Add a grid pattern
    for i in range(0, width, 50):
        cv2.line(right_img, (i, 0), (i, height), (200, 200, 200), 1)
    for i in range(0, height, 50):
        cv2.line(right_img, (0, i), (width, i), (200, 200, 200), 1)
    
    # Draw a red ball (with slight position difference to simulate stereo)
    ball_center_right = (350, 220)  # Offset from left image
    ball_radius_right = 35  # Slightly different size
    cv2.circle(right_img, ball_center_right, ball_radius_right, (0, 0, 255), -1)
    
    # Add highlight to make it look more realistic
    highlight_center_right = (ball_center_right[0] - 10, ball_center_right[1] - 10)
    highlight_radius_right = 12
    cv2.circle(right_img, highlight_center_right, highlight_radius_right, (50, 50, 255), -1)
    
    # Save images
    cv2.imwrite("test_data/left_sample.png", left_img)
    cv2.imwrite("test_data/right_sample.png", right_img)
    
    # Generate sequence of images with movement
    for i in range(5):
        # Create copies of the base images
        left_seq = left_img.copy()
        right_seq = right_img.copy()
        
        # Calculate new ball positions
        new_left_center = (ball_center[0] + i*20, ball_center[1] - i*10)
        new_right_center = (ball_center_right[0] + i*20, ball_center_right[1] - i*10)
        
        # Clear previous ball position
        cv2.circle(left_seq, ball_center, ball_radius + 5, (240, 240, 240), -1)
        cv2.circle(right_seq, ball_center_right, ball_radius_right + 5, (240, 240, 240), -1)
        
        # Redraw the grid lines that were cleared
        for j in range(0, width, 50):
            cv2.line(left_seq, (j, 0), (j, height), (200, 200, 200), 1)
            cv2.line(right_seq, (j, 0), (j, height), (200, 200, 200), 1)
        for j in range(0, height, 50):
            cv2.line(left_seq, (0, j), (width, j), (200, 200, 200), 1)
            cv2.line(right_seq, (0, j), (width, j), (200, 200, 200), 1)
        
        # Draw new ball positions
        cv2.circle(left_seq, new_left_center, ball_radius, (0, 0, 255), -1)
        cv2.circle(right_seq, new_right_center, ball_radius_right, (0, 0, 255), -1)
        
        # Add highlights
        new_left_highlight = (new_left_center[0] - 10, new_left_center[1] - 10)
        new_right_highlight = (new_right_center[0] - 10, new_right_center[1] - 10)
        cv2.circle(left_seq, new_left_highlight, highlight_radius, (50, 50, 255), -1)
        cv2.circle(right_seq, new_right_highlight, highlight_radius_right, (50, 50, 255), -1)
        
        # Save sequence images
        cv2.imwrite(f"test_data/left_seq_{i+1}.png", left_seq)
        cv2.imwrite(f"test_data/right_seq_{i+1}.png", right_seq)
    
    print("Test images generated successfully:")
    print("- test_data/left_sample.png")
    print("- test_data/right_sample.png")
    print("- test_data/left_seq_1.png to test_data/left_seq_5.png")
    print("- test_data/right_seq_1.png to test_data/right_seq_5.png")

if __name__ == "__main__":
    generate_test_images() 