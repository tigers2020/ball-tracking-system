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

from src.utils.ui_constants import Layout


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
        
        # Convert from BGR to RGB
        if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Create QImage from numpy array
        height, width = cv_image.shape[:2]
        bytes_per_line = 3 * width
        
        q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Convert to pixmap and display
        pixmap = QPixmap.fromImage(q_image)
        self.set_image(pixmap)
        return True
    
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
    
    def set_titles(self, left_title="Left Image", right_title="Right Image"):
        """
        Set the titles for the left and right image views.
        
        Args:
            left_title (str): Left image title
            right_title (str): Right image title
        """
        self.left_image_view.set_title(left_title)
        self.right_image_view.set_title(right_title) 