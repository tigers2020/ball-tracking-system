#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image View Widget module.
This module contains the ImageViewWidget class for displaying images in the Stereo Image Player.
"""

import cv2
import numpy as np
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QSizePolicy
import logging

from src.utils.ui_constants import Layout, ROI
from src.views.visualization import OpenCVVisualizer


class ImageViewWidget(QWidget):
    """
    Widget for displaying an image with title.
    """
    
    def __init__(self, title="Image", parent=None):
        """
        Initialize the image view widget.
        
        Args:
            title (str): Title to display above the image
            parent (QWidget, optional): Parent widget
        """
        super(ImageViewWidget, self).__init__(parent)
        
        # Set up UI
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(Layout.MARGIN, Layout.MARGIN, Layout.MARGIN, Layout.MARGIN)
        self.layout.setSpacing(Layout.SPACING)
        
        # Title label
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title_label)
        
        # Image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMinimumSize(320, 240)
        self.layout.addWidget(self.image_label)
        
        # Store current CV image
        self.current_cv_image = None
        self.current_mask = None
        self.current_roi = None
        self.mask_enabled = True  # Default to True for backward compatibility
        
        # Store HSV settings for dynamic color mask
        self.hsv_settings = None
        
        # Cache for optimizing performance
        self._prev_mask = None
        self._cached_overlay = None
        self._frame_id = 0
        self._log_sample_rate = 30  # Only log once every 30 frames
        
        # Empty pixmap as placeholder
        self.clear_image()
    
    def set_image_from_cv(self, cv_image):
        """
        Set the image from an OpenCV image (numpy array).
        
        Args:
            cv_image (numpy.ndarray): OpenCV image
            
        Returns:
            bool: True if successful, False otherwise
        """
        if cv_image is None:
            self.clear_image()
            return False
        
        # Increment frame counter for sampling logs
        self._frame_id += 1
        
        # Store the original CV image
        self.current_cv_image = cv_image.copy()
        
        # Create a working copy of the image
        display_image = cv_image.copy()
        
        # Draw ROI first if available (so it's visible underneath the mask)
        if self.current_roi is not None:
            display_image = self._draw_roi(display_image, self.current_roi)
        
        # Apply mask last if available (with transparency so other elements show through)
        if self.current_mask is not None:
            display_image = self._apply_mask(display_image, self.current_mask)
        
        # Convert from BGR to RGB
        if len(display_image.shape) == 3 and display_image.shape[2] == 3:
            display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        
        # Create QImage from numpy array
        height, width = display_image.shape[:2]
        bytes_per_line = 3 * width
        
        q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Convert to pixmap and display
        pixmap = QPixmap.fromImage(q_image)
        self.set_image(pixmap)
        return True
    
    def set_mask(self, mask, hsv_settings=None):
        """
        Set a binary mask to overlay on the image.
        
        Args:
            mask (numpy.ndarray): Binary mask (0-255)
            hsv_settings (dict, optional): HSV settings for dynamic color visualization
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Reset cached mask when receiving a new one
        self._prev_mask = None
        self._cached_overlay = None
        self.current_mask = mask
        
        # Update HSV settings if provided
        if hsv_settings is not None:
            self.hsv_settings = hsv_settings
        
        # Reapply image with mask if we have an image
        if self.current_cv_image is not None:
            return self.set_image_from_cv(self.current_cv_image)
        
        return False
    
    def set_roi(self, roi):
        """
        Set a ROI to display on the image.
        
        Args:
            roi (dict): ROI information with x, y, width, height, center_x, center_y
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.current_roi = roi
        
        # Reapply image with ROI if we have an image
        if self.current_cv_image is not None:
            return self.set_image_from_cv(self.current_cv_image)
        
        return False
    
    def _apply_mask(self, image, mask):
        """
        Apply a mask overlay to the image.
        
        Args:
            image (numpy.ndarray): Original image
            mask (numpy.ndarray): Binary mask
            
        Returns:
            numpy.ndarray: Image with mask overlay
        """
        if image is None or mask is None:
            return image
        
        # If mask overlay is disabled, return original image
        if hasattr(self, 'mask_enabled') and not self.mask_enabled:
            return image
        
        # Fast-exit if mask unchanged - use cached result
        if self._prev_mask is not None and mask.size == self._prev_mask.size:
            if np.array_equal(mask, self._prev_mask) and self._cached_overlay is not None:
                return self._cached_overlay
        
        # Count non-zero pixels for performance logging (but only periodically)
        if self._frame_id % self._log_sample_rate == 0:
            non_zero = cv2.countNonZero(mask)
            mask_ratio = non_zero / mask.size if mask.size > 0 else 0
            logging.debug(f"Mask: {mask.shape}, non-zero: {non_zero}, ratio: {mask_ratio:.4f}")
        
        try:
            # Apply mask overlay with dynamic color if HSV settings are available
            result = OpenCVVisualizer.apply_mask_overlay(
                image, 
                mask, 
                alpha=0.3,  # Semi-transparent
                hsv_settings=self.hsv_settings
            )
            
            # Cache for future reuse
            self._prev_mask = mask.copy()
            self._cached_overlay = result.copy()
            
            return result
            
        except Exception as e:
            logging.error(f"Error applying mask overlay: {e}")
            
            # Fallback method if visualization module import fails
            colored_mask = np.zeros_like(image)
            colored_mask[mask > 0] = (0, 255, 0)  # Default to green if visualization fails
            
            # Apply colored mask with alpha blending
            result = image.copy()
            cv2.addWeighted(colored_mask, 0.3, result, 1.0, 0, result)
            
            # Cache for future reuse
            self._prev_mask = mask.copy()
            self._cached_overlay = result.copy()
            
            return result
    
    def _draw_roi(self, image, roi):
        """
        Draw ROI rectangle and center point on the image.
        
        Args:
            image (numpy.ndarray): Original image
            roi (dict): ROI information with x, y, width, height, center_x, center_y
            
        Returns:
            numpy.ndarray: Image with ROI visualization
        """
        if image is None or roi is None:
            return image
        
        try:
            # Apply ROI visualization with dynamic color if HSV settings are available
            result = OpenCVVisualizer.draw_roi(
                image, 
                roi, 
                color=ROI.BORDER_COLOR, 
                thickness=ROI.BORDER_THICKNESS, 
                show_center=True, 
                center_color=ROI.CENTER_MARKER_COLOR
            )
            
            logging.debug(f"ROI drawn: x={roi['x']}, y={roi['y']}, w={roi['width']}, h={roi['height']}, center=({roi['center_x']}, {roi['center_y']})")
            return result
            
        except Exception as e:
            logging.error(f"Error drawing ROI: {e}")
            return image
    
    def set_image(self, pixmap):
        """
        Set the image from a QPixmap.
        
        Args:
            pixmap (QPixmap): Image pixmap
        """
        if pixmap.isNull():
            self.clear_image()
            return
        
        # Scale pixmap to fit the label while preserving aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.image_label.setPixmap(scaled_pixmap)
    
    def clear_image(self):
        """Clear the image by setting an empty pixmap."""
        empty_pixmap = QPixmap(QSize(320, 240))
        empty_pixmap.fill(Qt.black)
        self.image_label.setPixmap(empty_pixmap)
        self.current_cv_image = None
    
    def enable_mask_overlay(self, enable: bool = True):
        """
        Enable or disable mask overlay on images.
        
        Args:
            enable (bool): Whether to enable mask overlay
        """
        self.mask_enabled = enable
        # Refresh display if we have an image
        if self.current_cv_image is not None:
            self.set_image_from_cv(self.current_cv_image)
    
    def set_title(self, title):
        """
        Set the title displayed above the image.
        
        Args:
            title (str): Title text
        """
        self.title_label.setText(title)
    
    def resizeEvent(self, event):
        """
        Handle resize events to scale the image.
        
        Args:
            event (QResizeEvent): Resize event
        """
        pixmap = self.image_label.pixmap()
        if pixmap and not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
        
        super(ImageViewWidget, self).resizeEvent(event)


class StereoImageViewWidget(QWidget):
    """
    Widget for displaying stereo images (left and right) side by side.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the stereo image view widget.
        
        Args:
            parent (QWidget, optional): Parent widget
        """
        super(StereoImageViewWidget, self).__init__(parent)
        
        # Set up UI
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(Layout.MARGIN, Layout.MARGIN, Layout.MARGIN, Layout.MARGIN)
        self.layout.setSpacing(Layout.SPACING)
        
        # Left image view
        self.left_image_view = ImageViewWidget("Left Image")
        self.layout.addWidget(self.left_image_view)
        
        # Right image view
        self.right_image_view = ImageViewWidget("Right Image")
        self.layout.addWidget(self.right_image_view)
        
        # Flag to prevent duplicate signal connections
        self._controller_connected = False
        
        # Set size policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    
    def set_images(self, left_image, right_image):
        """
        Set the left and right images.
        
        Args:
            left_image (numpy.ndarray): Left OpenCV image
            right_image (numpy.ndarray): Right OpenCV image
            
        Returns:
            tuple: (left_success, right_success) indicating if each image was successfully set
        """
        left_success = self.left_image_view.set_image_from_cv(left_image)
        right_success = self.right_image_view.set_image_from_cv(right_image)
        return left_success, right_success
    
    def clear_images(self):
        """Clear both the left and right images."""
        self.left_image_view.clear_image()
        self.right_image_view.clear_image()
    
    def set_masks(self, left_mask, right_mask, hsv_settings=None):
        """
        Set masks for the left and right images.
        
        Args:
            left_mask (numpy.ndarray): Binary mask for left image
            right_mask (numpy.ndarray): Binary mask for right image
            hsv_settings (dict, optional): HSV settings for dynamic color visualization
            
        Returns:
            tuple: (left_success, right_success) indicating if each mask was successfully set
        """
        left_success = self.left_image_view.set_mask(left_mask, hsv_settings)
        right_success = self.right_image_view.set_mask(right_mask, hsv_settings)
        return left_success, right_success
    
    def set_rois(self, left_roi, right_roi):
        """
        Set ROIs for the left and right images.
        
        Args:
            left_roi (dict): ROI information for left image
            right_roi (dict): ROI information for right image
            
        Returns:
            tuple: (left_success, right_success) indicating if each ROI was successfully set
        """
        left_success = self.left_image_view.set_roi(left_roi)
        right_success = self.right_image_view.set_roi(right_roi)
        return left_success, right_success
    
    def set_titles(self, left_title="Left Image", right_title="Right Image"):
        """
        Set titles for both image views.
        
        Args:
            left_title (str): Title for left image
            right_title (str): Title for right image
        """
        self.left_image_view.set_title(left_title)
        self.right_image_view.set_title(right_title)
    
    def enable_mask_overlay(self, enable: bool = True):
        """
        Enable or disable mask overlay on both images.
        
        Args:
            enable (bool): Whether to enable mask overlay
        """
        self.left_image_view.enable_mask_overlay(enable)
        self.right_image_view.enable_mask_overlay(enable)
    
    def display_roi_images(self, left_roi_image, right_roi_image):
        """
        Display cropped ROI images in separate windows.
        
        Args:
            left_roi_image (numpy.ndarray): Cropped ROI image from left camera
            right_roi_image (numpy.ndarray): Cropped ROI image from right camera
            
        Returns:
            bool: True if at least one ROI image was displayed successfully
        """
        success = False
        
        # Create ROI display windows if needed
        if not hasattr(self, 'left_roi_view'):
            self.left_roi_view = ImageViewWidget("Left ROI")
            self.left_roi_view.setWindowTitle("Left ROI")
            self.left_roi_view.resize(320, 240)
            
        if not hasattr(self, 'right_roi_view'):
            self.right_roi_view = ImageViewWidget("Right ROI")
            self.right_roi_view.setWindowTitle("Right ROI")
            self.right_roi_view.resize(320, 240)
            
        # Display left ROI image if available
        if left_roi_image is not None:
            self.left_roi_view.set_image_from_cv(left_roi_image)
            self.left_roi_view.show()
            success = True
        else:
            self.left_roi_view.hide()
            
        # Display right ROI image if available
        if right_roi_image is not None:
            self.right_roi_view.set_image_from_cv(right_roi_image)
            self.right_roi_view.show()
            success = True
        else:
            self.right_roi_view.hide()
            
        return success
    
    def save_roi_images(self, left_roi_image, right_roi_image, left_path=None, right_path=None):
        """
        Save ROI images to disk.
        
        Args:
            left_roi_image (numpy.ndarray): Cropped ROI image from left camera
            right_roi_image (numpy.ndarray): Cropped ROI image from right camera
            left_path (str, optional): Path to save left ROI image. If None, a default path is used.
            right_path (str, optional): Path to save right ROI image. If None, a default path is used.
            
        Returns:
            tuple: (left_success, right_success) indicating if each image was successfully saved
        """
        import os
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(os.getcwd(), "roi_images")
        
        # Create directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Set default paths if not provided
        if left_path is None:
            left_path = os.path.join(save_dir, f"left_roi_{timestamp}.png")
        if right_path is None:
            right_path = os.path.join(save_dir, f"right_roi_{timestamp}.png")
            
        left_success = False
        right_success = False
        
        # Save left ROI image
        if left_roi_image is not None:
            try:
                cv2.imwrite(left_path, left_roi_image)
                left_success = True
                logging.info(f"Left ROI image saved to {left_path}")
            except Exception as e:
                logging.error(f"Error saving left ROI image: {e}")
                
        # Save right ROI image
        if right_roi_image is not None:
            try:
                cv2.imwrite(right_path, right_roi_image)
                right_success = True
                logging.info(f"Right ROI image saved to {right_path}")
            except Exception as e:
                logging.error(f"Error saving right ROI image: {e}")
                
        return left_success, right_success
    
    def connect_ball_tracking_controller(self, controller):
        """
        Connect the ball tracking controller to the image view.
        
        Args:
            controller: BallTrackingController instance
        """
        # Prevent duplicate connections
        if hasattr(self, '_controller_connected') and self._controller_connected:
            logging.info("Ball tracking controller already connected, skipping reconnection")
            return
            
        # Connect controller signals to view
        controller.mask_updated.connect(self.set_masks)
        controller.roi_updated.connect(self.set_rois)
        controller.circles_processed.connect(self.set_images)
        
        # Mark as connected
        self._controller_connected = True
        logging.info("Connected to ball tracking controller") 