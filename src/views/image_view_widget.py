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
        
        # Store the original CV image
        self.current_cv_image = cv_image.copy()
        
        # Create a working copy of the image
        display_image = cv_image.copy()
        
        # Apply mask if available
        if self.current_mask is not None:
            display_image = self._apply_mask(display_image, self.current_mask)
        
        # Draw ROI if available
        if self.current_roi is not None:
            display_image = self._draw_roi(display_image, self.current_roi)
        
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
    
    def set_mask(self, mask):
        """
        Set a binary mask to overlay on the image.
        
        Args:
            mask (numpy.ndarray): Binary mask (0-255)
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.current_mask = mask
        
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
        
        try:
            # Make a copy of the original image
            result = image.copy()
            
            # Ensure mask is the same size as the image
            if mask.shape[:2] != image.shape[:2]:
                logging.debug(f"Resizing mask from {mask.shape} to match image size {image.shape[:2]}")
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # 1. Apply contour highlighting (red boundary around detected area)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, (0, 0, 255), 2)  # Red contour with thickness 2
            
            # 2. Apply colored mask overlay
            # Create a colored overlay (red with transparency)
            overlay = np.zeros_like(image, dtype=np.uint8)
            
            # Create a binary mask that matches the image size
            binary_mask = (mask > 0)
            if binary_mask.shape[:2] != overlay.shape[:2]:
                logging.error(f"Mask shape {binary_mask.shape} doesn't match overlay shape {overlay.shape[:2]}")
                # Safety check - don't apply mismatched mask
                return result
                
            # Apply red color where mask is non-zero
            overlay[binary_mask] = (0, 0, 255)
            
            # Blend the overlay with the contour-highlighted image
            alpha = 0.3  # Transparency factor
            cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
            
            # 3. Draw a small square at the center of each contour
            for contour in contours:
                if contour.size > 5:  # Only if contour has enough points
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        # Draw a small white square at the center
                        square_size = 5
                        cv2.rectangle(result, 
                                     (cX - square_size, cY - square_size), 
                                     (cX + square_size, cY + square_size), 
                                     (255, 255, 255), -1)  # White filled square
            
            # Log successful mask application
            logging.debug(f"Mask applied successfully - mask shape: {mask.shape}, non-zero pixels: {np.count_nonzero(mask)}")
            
            return result
            
        except Exception as e:
            logging.error(f"Error applying mask: {e}")
            return image
    
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
            # Make a copy to avoid modifying the original
            result = image.copy()
            
            # Safely extract ROI information with default values
            try:
                x = int(roi.get("x", 0))
                y = int(roi.get("y", 0))
                w = int(roi.get("width", 100))
                h = int(roi.get("height", 100))
                center_x = int(roi.get("center_x", x + w // 2))
                center_y = int(roi.get("center_y", y + h // 2))
                
                # Ensure all values are valid
                if w <= 0 or h <= 0:
                    logging.warning(f"Invalid ROI dimensions: w={w}, h={h}, using defaults")
                    w = max(1, w)
                    h = max(1, h)
                
                # Ensure coordinates are within image bounds
                img_h, img_w = image.shape[:2]
                x = max(0, min(x, img_w - 1))
                y = max(0, min(y, img_h - 1))
                w = min(w, img_w - x)
                h = min(h, img_h - y)
                center_x = max(0, min(center_x, img_w - 1))
                center_y = max(0, min(center_y, img_h - 1))
                
            except (ValueError, TypeError) as e:
                logging.error(f"ROI value conversion error: {e}")
                return image
            
            # Draw ROI rectangle
            cv2.rectangle(
                result,
                (x, y),
                (x + w, y + h),
                ROI.BORDER_COLOR,
                ROI.BORDER_THICKNESS
            )
            
            # Draw center marker
            cv2.drawMarker(
                result,
                (center_x, center_y),
                ROI.CENTER_MARKER_COLOR,
                cv2.MARKER_CROSS,
                ROI.CENTER_MARKER_SIZE * 2,
                ROI.BORDER_THICKNESS
            )
            
            logging.debug(f"ROI drawn: x={x}, y={y}, w={w}, h={h}, center=({center_x}, {center_y})")
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
        """Clear the image display."""
        # Create an empty pixmap
        empty_pixmap = QPixmap(640, 480)
        empty_pixmap.fill(Qt.black)
        self.image_label.setPixmap(empty_pixmap)
        
        # Clear stored images and masks
        self.current_cv_image = None
        self.current_mask = None
        self.current_roi = None
    
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
    
    def set_masks(self, left_mask, right_mask):
        """
        Set masks for the left and right images.
        
        Args:
            left_mask (numpy.ndarray): Binary mask for left image
            right_mask (numpy.ndarray): Binary mask for right image
            
        Returns:
            tuple: (left_success, right_success) indicating if each mask was successfully set
        """
        left_success = self.left_image_view.set_mask(left_mask)
        right_success = self.right_image_view.set_mask(right_mask)
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
        Set the titles for the left and right image views.
        
        Args:
            left_title (str): Left image title
            right_title (str): Right image title
        """
        self.left_image_view.set_title(left_title)
        self.right_image_view.set_title(right_title)
        
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