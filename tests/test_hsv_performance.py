#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HSV Mask Performance Test Script.
This script tests the performance of CPU vs GPU-based HSV mask creation.
"""

import cv2
import numpy as np
import time
import torch
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.controllers.ball_tracking_controller import BallTrackingController
from src.utils.gpu_utils import create_hsv_mask_gpu, upload_tensor, download_tensor, IS_GPU_AVAILABLE, DEVICE
from src.utils.config_manager import ConfigManager
from src.utils.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def test_hsv_performance(image_path=None, num_iterations=100, batch_test=False):
    """
    Test the performance of CPU vs GPU-based HSV mask creation.
    
    Args:
        image_path (str): Path to test image file
        num_iterations (int): Number of iterations for timing
        batch_test (bool): Whether to test batch processing performance
    """
    logger.info("Starting HSV mask performance test")
    logger.info(f"GPU available: {IS_GPU_AVAILABLE}")
    
    # Create test controller
    controller = BallTrackingController()
    hsv_values = controller.get_hsv_values()
    
    # Load test image or create one
    if image_path and os.path.exists(image_path):
        logger.info(f"Loading test image from {image_path}")
        image = cv2.imread(image_path)
    else:
        # Create a larger test image (Full HD)
        logger.info("Creating test image (1920x1080)")
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Create multiple colored objects to test various HSV ranges
        # Create a red ball (main target)
        cv2.circle(image, (960, 540), 80, (0, 0, 255), -1)
        
        # Create some additional colored objects
        # Blue circle
        cv2.circle(image, (500, 300), 60, (255, 0, 0), -1)
        # Green circle
        cv2.circle(image, (1400, 700), 70, (0, 255, 0), -1)
        # Yellow circle
        cv2.circle(image, (700, 800), 50, (0, 255, 255), -1)
        
        # Add some noise and gradients for realism
        noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
        image = cv2.add(image, noise)
        
        # Add a background gradient
        for i in range(image.shape[1]):
            value = int(255 * i / image.shape[1])
            cv2.line(image, (i, 0), (i, image.shape[0]), (value//3, value//3, value//3), 1)
    
    # Ensure we have a valid image
    if image is None or image.size == 0:
        logger.error("Failed to load/create test image")
        return
    
    # Convert to HSV for CPU testing
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Upload to GPU for GPU testing
    if IS_GPU_AVAILABLE:
        # Convert BGR to RGB for kornia
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Upload directly to BCHW format
        image_tensor = upload_tensor(rgb_img, to_bchw=True)
        logger.debug(f"Uploaded tensor shape: {image_tensor.shape}")
    
    # Test CPU implementation
    logger.info("Testing CPU implementation...")
    cpu_times = []
    for i in range(num_iterations):
        start_time = time.time()
        mask = controller._create_hsv_mask(hsv_img, hsv_values)
        elapsed = time.time() - start_time
        cpu_times.append(elapsed)
        if i % max(1, num_iterations // 5) == 0:
            logger.info(f"CPU Iteration {i}: {elapsed*1000:.2f} ms")
    
    avg_cpu_time = sum(cpu_times) / len(cpu_times)
    logger.info(f"CPU Average time: {avg_cpu_time*1000:.2f} ms")
    
    # Test GPU implementation (if available)
    if IS_GPU_AVAILABLE:
        logger.info("Testing GPU implementation...")
        gpu_times = []
        for i in range(num_iterations):
            # Warm up GPU for first few iterations
            if i > 5:  # Skip first 5 iterations for warm-up
                start_time = time.time()
                mask_tensor = create_hsv_mask_gpu(image_tensor, hsv_values)
                # Force synchronization to get accurate timing
                torch.cuda.synchronize()
                elapsed = time.time() - start_time
                gpu_times.append(elapsed)
                if i % max(1, num_iterations // 5) == 0:
                    logger.info(f"GPU Iteration {i}: {elapsed*1000:.2f} ms")
            else:
                # Warm-up iterations (not timed)
                mask_tensor = create_hsv_mask_gpu(image_tensor, hsv_values)
                torch.cuda.synchronize()
                logger.info(f"GPU Warm-up iteration {i}")
        
        avg_gpu_time = sum(gpu_times) / len(gpu_times)
        logger.info(f"GPU Average time: {avg_gpu_time*1000:.2f} ms")
        
        # Calculate speedup
        speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else float('inf')
        logger.info(f"GPU speedup: {speedup:.2f}x")
        
        # Test batch processing if requested
        if batch_test and IS_GPU_AVAILABLE:
            logger.info("Testing GPU batch processing performance...")
            
            batch_sizes = [1, 2, 4, 8, 16]
            batch_times = {}
            
            for batch_size in batch_sizes:
                if batch_size == 1:
                    # Already tested above
                    batch_times[batch_size] = avg_gpu_time
                    continue
                    
                logger.info(f"Testing batch size {batch_size}...")
                
                # Create a batch by repeating the image
                batch_tensor = image_tensor.repeat(batch_size, 1, 1, 1)
                
                # Warm up with this batch size
                for _ in range(3):
                    mask_batch = create_hsv_mask_gpu(batch_tensor, hsv_values)
                    torch.cuda.synchronize()
                
                # Time the batch processing
                batch_iteration_times = []
                for _ in range(10):  # 10 iterations for each batch size
                    start_time = time.time()
                    mask_batch = create_hsv_mask_gpu(batch_tensor, hsv_values)
                    torch.cuda.synchronize()
                    elapsed = time.time() - start_time
                    batch_iteration_times.append(elapsed)
                
                avg_batch_time = sum(batch_iteration_times) / len(batch_iteration_times)
                batch_times[batch_size] = avg_batch_time
                
                # Calculate per-image time
                per_image_time = avg_batch_time / batch_size
                speedup_vs_single = avg_gpu_time / per_image_time if per_image_time > 0 else float('inf')
                
                logger.info(f"Batch size {batch_size}:")
                logger.info(f"  Total batch time: {avg_batch_time*1000:.2f} ms")
                logger.info(f"  Per-image time: {per_image_time*1000:.2f} ms")
                logger.info(f"  Speedup vs. single image: {speedup_vs_single:.2f}x")
            
            # Report batch processing summary
            logger.info("Batch processing summary:")
            for batch_size, batch_time in batch_times.items():
                per_image_time = batch_time / batch_size
                logger.info(f"  Batch size {batch_size}: {per_image_time*1000:.2f} ms per image")
    
    # Save results for visual comparison
    try:
        # Save original image
        cv2.imwrite("test_original.jpg", image)
        
        # Save CPU mask
        mask = controller._create_hsv_mask(hsv_img, hsv_values)
        if mask is not None:
            cv2.imwrite("test_cpu_mask.jpg", mask)
        
        # Save GPU mask if available
        if IS_GPU_AVAILABLE:
            mask_tensor = create_hsv_mask_gpu(image_tensor, hsv_values)
            if mask_tensor is not None:
                gpu_mask = download_tensor(mask_tensor.squeeze())
                cv2.imwrite("test_gpu_mask.jpg", gpu_mask)
                
            # Create a visual comparison
            if mask is not None and gpu_mask is not None:
                diff = cv2.absdiff(mask, gpu_mask)
                cv2.imwrite("test_mask_diff.jpg", diff)
                
                # Overlay masks on original image
                original_with_cpu = image.copy()
                original_with_gpu = image.copy()
                
                # Apply colored overlays
                original_with_cpu[mask > 0] = [0, 0, 255]  # Red for CPU mask
                original_with_gpu[gpu_mask > 0] = [0, 255, 0]  # Green for GPU mask
                
                cv2.imwrite("test_overlay_cpu.jpg", original_with_cpu)
                cv2.imwrite("test_overlay_gpu.jpg", original_with_gpu)
        
        logger.info("Saved test images and masks")
    except Exception as e:
        logger.error(f"Error saving test images: {e}")
    
    return {
        "cpu_avg_time": avg_cpu_time,
        "gpu_avg_time": avg_gpu_time if IS_GPU_AVAILABLE else None,
        "speedup": speedup if IS_GPU_AVAILABLE else None
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test HSV mask performance")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--batch-test", action="store_true", help="Test batch processing performance")
    args = parser.parse_args()
    
    test_hsv_performance(args.image, args.iterations, args.batch_test) 