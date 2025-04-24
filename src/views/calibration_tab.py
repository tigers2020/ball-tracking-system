#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Court Calibration View module.
This module contains the CourtCalibrationView class for the court calibration tab.
"""

import logging
import numpy as np
import cv2
from PySide6.QtCore import Qt, Signal, Slot, QPointF, QRectF
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QIcon, QBrush, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsEllipseItem, QGraphicsItem,
    QFileDialog, QMessageBox, QProgressDialog, QSpinBox, QCheckBox, QGroupBox
)

from src.utils.ui_constants import Layout, WindowSize, Messages, Icons
from src.controllers.calibration import CourtCalibrationController


class CalibrationGraphicsView(QGraphicsView):
    """
    Custom graphics view for court calibration with point selection capability.
    """
    
    point_clicked = Signal(tuple, str)  # Emitted when user clicks to add a point (coords, side)
    point_moved = Signal(int, tuple, str)  # Emitted when point is moved (index, new_pos, side)
    
    def __init__(self, side="left", parent=None):
        """
        Initialize the calibration graphics view.
        
        Args:
            side (str): "left" or "right" to indicate which side this view represents
            parent (QWidget, optional): Parent widget
        """
        super(CalibrationGraphicsView, self).__init__(parent)
        
        # Set side
        self.side = side
        
        # Set up scene
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # Set up view properties
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)  # Changed from ScrollHandDrag to NoDrag
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        
        # Flag to track if we're in drag mode
        self.is_dragging = False
        self.current_dragged_point = None
        self.current_dragged_index = -1
        
        # Image item
        self.image_item = None
        
        # Point markers
        self.point_markers = []
        self.point_labels = []
        
        # ROI overlay
        self.roi_overlay = None
        self.hover_point = None
        
        # State tracking
        self.image_loaded = False
        self.points_visible = True
        self.roi_visible = True
        
        # Point appearance
        self.point_size = 12  # Increased from 6 to 12
        self.point_color = QColor(255, 0, 0)  # Red for raw points
        self.fine_point_color = QColor(0, 255, 0)  # Green for fine-tuned points
        self.roi_color = QColor(0, 0, 255, 100)  # Semi-transparent blue for ROI
        
        # Maximum number of points per side
        self.max_points = 14
        
        # Set viewport cursor to normal arrow
        self.viewport().setCursor(Qt.ArrowCursor)
    
    def wheelEvent(self, event):
        """
        Handle mouse wheel events for zooming.
        
        Args:
            event (QWheelEvent): Wheel event
        """
        zoom_factor = 1.15
        
        if event.angleDelta().y() > 0:
            # Zoom in
            self.scale(zoom_factor, zoom_factor)
        else:
            # Zoom out
            self.scale(1.0 / zoom_factor, 1.0 / zoom_factor)
    
    def mousePressEvent(self, event):
        """
        Handle mouse press events for adding points.
        
        Args:
            event (QMouseEvent): Mouse event
        """
        if event.button() == Qt.LeftButton and self.image_loaded:
            # Get the item under the cursor
            item = self.itemAt(event.pos())
            
            # Check if we're clicking on a point marker or on empty space
            if item is None or (item not in self.point_markers and item not in self.point_labels):
                # We're clicking on empty space, add a new point
                scene_pos = self.mapToScene(event.pos())
                
                # Check if we've reached the maximum number of points
                if len(self.point_markers) >= self.max_points:
                    return
                    
                # Emit signal with point coordinates and side
                self.point_clicked.emit((int(scene_pos.x()), int(scene_pos.y())), self.side)
            else:
                # We're clicking on an existing point
                if item in self.point_markers:
                    # Find the index of the clicked point marker
                    self.current_dragged_index = self.point_markers.index(item)
                    self.current_dragged_point = item
                elif item in self.point_labels:
                    # Find the index of the clicked point label
                    self.current_dragged_index = self.point_labels.index(item)
                    # Get the corresponding marker
                    self.current_dragged_point = self.point_markers[self.current_dragged_index]
                
                # Start drag mode
                self.setDragMode(QGraphicsView.NoDrag)
                self.is_dragging = True
                # Change cursor to hand
                self.viewport().setCursor(Qt.ClosedHandCursor)
                # Store initial scene position
                self.drag_start_pos = self.mapToScene(event.pos())
        
        # Pass event to parent for handling
        super(CalibrationGraphicsView, self).mousePressEvent(event)
    
    def mouseReleaseEvent(self, event):
        """
        Handle mouse release events to end drag mode.
        
        Args:
            event (QMouseEvent): Mouse event
        """
        if self.is_dragging and self.current_dragged_point is not None:
            # Calculate new position
            scene_pos = self.mapToScene(event.pos())
            new_pos = (int(scene_pos.x()), int(scene_pos.y()))
            
            # Emit signal with point index and new position
            self.point_moved.emit(self.current_dragged_index, new_pos, self.side)
            
            # Reset dragging state
            self.current_dragged_point = None
            self.current_dragged_index = -1
            
            # End drag mode
            self.setDragMode(QGraphicsView.NoDrag)
            self.is_dragging = False
            # Change cursor back to arrow
            self.viewport().setCursor(Qt.ArrowCursor)
        
        super(CalibrationGraphicsView, self).mouseReleaseEvent(event)
    
    def mouseMoveEvent(self, event):
        """
        Handle mouse move events for showing ROI overlay and dragging points.
        
        Args:
            event (QMouseEvent): Mouse event
        """
        # If dragging a point, update its position
        if self.is_dragging and self.current_dragged_point is not None:
            scene_pos = self.mapToScene(event.pos())
            x, y = int(scene_pos.x()), int(scene_pos.y())
            
            # Update the position of the point marker
            radius = self.point_size
            self.current_dragged_point.setRect(x - radius/2, y - radius/2, radius, radius)
            
            # Update the position of the corresponding label
            if self.current_dragged_index >= 0 and self.current_dragged_index < len(self.point_labels):
                label = self.point_labels[self.current_dragged_index]
                label.setPos(x + radius/2, y - radius/2)
        else:
            # Change cursor if hovering over point markers
            item = self.itemAt(event.pos())
            if item in self.point_markers or item in self.point_labels:
                if not self.is_dragging:
                    self.viewport().setCursor(Qt.OpenHandCursor)
            elif not self.is_dragging:
                self.viewport().setCursor(Qt.ArrowCursor)
            
            if self.image_loaded and self.roi_visible and not self.is_dragging:
                # Get scene position
                scene_pos = self.mapToScene(event.pos())
                self.hover_point = (int(scene_pos.x()), int(scene_pos.y()))
                
                # Update ROI overlay
                self._update_roi_overlay()
        
        # Pass event to parent for handling drag
        super(CalibrationGraphicsView, self).mouseMoveEvent(event)
    
    def leaveEvent(self, event):
        """
        Handle mouse leave events to hide ROI overlay.
        
        Args:
            event (QEvent): Leave event
        """
        self.hover_point = None
        self._update_roi_overlay()
        super(CalibrationGraphicsView, self).leaveEvent(event)
    
    def _update_roi_overlay(self):
        """Update the ROI overlay based on hover position."""
        # Remove existing overlay
        if self.roi_overlay:
            self.scene.removeItem(self.roi_overlay)
            self.roi_overlay = None
            
        # Add new overlay if we have a hover point
        if self.hover_point and self.image_loaded:
            x, y = self.hover_point
            roi_size = self.parent().parent().roi_size_spinner.value()
            
            # Create rectangle item for ROI
            roi_rect = QRectF(x - roi_size, y - roi_size, roi_size * 2, roi_size * 2)
            self.roi_overlay = self.scene.addRect(roi_rect, QPen(self.roi_color), QBrush(self.roi_color))
    
    def set_image(self, cv_img):
        """
        Set the image to display.
        
        Args:
            cv_img (numpy.ndarray): OpenCV image
            
        Returns:
            bool: True if image was set successfully
        """
        if cv_img is None or not isinstance(cv_img, np.ndarray):
            return False
            
        # Convert OpenCV image to QImage
        height, width, channels = cv_img.shape
        bytes_per_line = channels * width
        
        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        qimg = QImage(rgb_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        # Clear existing image
        if self.image_item:
            self.scene.removeItem(self.image_item)
            
        # Add new image
        self.image_item = self.scene.addPixmap(pixmap)
        
        # Adjust scene
        self.scene.setSceneRect(0, 0, width, height)
        
        # Reset view
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        
        self.image_loaded = True
        return True
    
    def add_point_marker(self, point, is_fine_point=False, index=None):
        """
        Add a point marker at the specified coordinates.
        
        Args:
            point (tuple): (x, y) coordinates
            is_fine_point (bool): True if this is a fine-tuned point
            index (int, optional): Index of the point for labeling
        """
        if not self.image_loaded:
            return
            
        x, y = point
        
        # Create ellipse item
        color = self.fine_point_color if is_fine_point else self.point_color
        radius = self.point_size
        
        ellipse = QGraphicsEllipseItem(x - radius/2, y - radius/2, radius, radius)
        ellipse.setPen(QPen(color, 2))
        ellipse.setBrush(QBrush(color, Qt.SolidPattern if is_fine_point else Qt.SolidPattern))
        
        # Make the ellipse selectable and movable (for drag detection)
        ellipse.setFlag(QGraphicsItem.ItemIsSelectable, True)
        
        # Add to scene and list
        self.scene.addItem(ellipse)
        self.point_markers.append(ellipse)
        
        # Add text label with point number
        if index is not None:
            font = QFont()
            font.setBold(True)
            font.setPointSize(10)
            
            text_color = QColor(255, 255, 255) if is_fine_point else QColor(255, 255, 255)
            text_item = self.scene.addText(str(index + 1), font)
            text_item.setDefaultTextColor(text_color)
            text_item.setPos(x + radius/2, y - radius/2)
            
            # Make the text selectable (for drag detection)
            text_item.setFlag(QGraphicsItem.ItemIsSelectable, True)
            
            self.point_labels.append(text_item)
            
            # Make sure it's visible
            text_item.setVisible(self.points_visible)
        
        # Make sure it's visible
        ellipse.setVisible(self.points_visible)
    
    def clear_points(self):
        """Clear all point markers and labels."""
        for marker in self.point_markers:
            self.scene.removeItem(marker)
        self.point_markers = []
        
        for label in self.point_labels:
            self.scene.removeItem(label)
        self.point_labels = []
    
    def set_points_visibility(self, visible):
        """
        Set visibility of point markers and labels.
        
        Args:
            visible (bool): True to show points, False to hide
        """
        self.points_visible = visible
        for marker in self.point_markers:
            marker.setVisible(visible)
        for label in self.point_labels:
            label.setVisible(visible)
    
    def set_roi_visibility(self, visible):
        """
        Set visibility of ROI overlay.
        
        Args:
            visible (bool): True to show ROI, False to hide
        """
        self.roi_visible = visible
        if not visible and self.roi_overlay:
            self.scene.removeItem(self.roi_overlay)
            self.roi_overlay = None


class CourtCalibrationView(QWidget):
    """
    Widget for court calibration tab.
    Allows loading images and calibrating court points.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the court calibration view.
        
        Args:
            parent (QWidget, optional): Parent widget
        """
        super(CourtCalibrationView, self).__init__(parent)
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        
        # Set up controller
        self.controller = CourtCalibrationController()
        
        # Connect controller signals
        self.controller.calibration_updated.connect(self.update_display)
        self.controller.error_occurred.connect(self.show_error)
        self.controller.processing_started.connect(self.processing_started)
        self.controller.processing_completed.connect(self.processing_completed)
        self.controller.tuning_completed.connect(self.tuning_completed)
        
        # Set up UI
        self._setup_ui()
        
        # Initialize progress dialog
        self.progress_dialog = None
        
        # Initialize the display
        self.update_display()
    
    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(Layout.MARGIN, Layout.MARGIN, Layout.MARGIN, Layout.MARGIN)
        main_layout.setSpacing(Layout.SPACING)
        
        # Top controls
        top_layout = QHBoxLayout()
        
        # Left controls (file operations)
        file_group = QGroupBox("Image Loading")
        file_layout = QVBoxLayout(file_group)
        
        self.load_frame_btn = QPushButton("Load Current Frame")
        self.load_frame_btn.setIcon(QIcon(Icons.OPEN))
        self.load_frame_btn.clicked.connect(self._on_load_current_frame)
        file_layout.addWidget(self.load_frame_btn)
        
        # Add save and load configuration buttons
        self.save_config_btn = QPushButton("Save Calibration")
        self.save_config_btn.clicked.connect(self._on_save_config)
        file_layout.addWidget(self.save_config_btn)
        
        self.load_config_btn = QPushButton("Load Calibration")
        self.load_config_btn.clicked.connect(self._on_load_config)
        file_layout.addWidget(self.load_config_btn)
        
        top_layout.addWidget(file_group)
        
        # Right controls (calibration operations)
        calib_group = QGroupBox("Calibration Controls")
        calib_layout = QVBoxLayout(calib_group)
        
        # Points controls
        points_layout = QHBoxLayout()
        
        self.clear_points_btn = QPushButton("Clear Points")
        self.clear_points_btn.clicked.connect(self._on_clear_points)
        points_layout.addWidget(self.clear_points_btn)
        
        self.show_points_cb = QCheckBox("Show Points")
        self.show_points_cb.setChecked(True)
        self.show_points_cb.stateChanged.connect(self._on_show_points_changed)
        points_layout.addWidget(self.show_points_cb)
        
        self.show_roi_cb = QCheckBox("Show ROI")
        self.show_roi_cb.setChecked(True)
        self.show_roi_cb.stateChanged.connect(self._on_show_roi_changed)
        points_layout.addWidget(self.show_roi_cb)
        
        calib_layout.addLayout(points_layout)
        
        # Point counters
        point_counters_layout = QHBoxLayout()
        point_counters_layout.addWidget(QLabel("Left Points:"))
        self.left_count_label = QLabel("0/14")
        point_counters_layout.addWidget(self.left_count_label)
        
        point_counters_layout.addWidget(QLabel("Right Points:"))
        self.right_count_label = QLabel("0/14")
        point_counters_layout.addWidget(self.right_count_label)
        
        calib_layout.addLayout(point_counters_layout)
        
        # ROI size spinner
        roi_layout = QHBoxLayout()
        roi_layout.addWidget(QLabel("ROI Size:"))
        
        self.roi_size_spinner = QSpinBox()
        self.roi_size_spinner.setRange(5, 50)
        self.roi_size_spinner.setValue(20)
        self.roi_size_spinner.valueChanged.connect(self._on_roi_size_changed)
        roi_layout.addWidget(self.roi_size_spinner)
        
        calib_layout.addLayout(roi_layout)
        
        # Fine-tune button
        self.tune_btn = QPushButton("Fine-Tune Points")
        self.tune_btn.clicked.connect(self._on_tune_points)
        calib_layout.addWidget(self.tune_btn)
        
        top_layout.addWidget(calib_group)
        
        main_layout.addLayout(top_layout)
        
        # Image views
        image_views_layout = QHBoxLayout()
        
        # Left image view
        left_group = QGroupBox("Left Image")
        left_layout = QVBoxLayout(left_group)
        self.left_image_view = CalibrationGraphicsView(side="left")
        self.left_image_view.point_clicked.connect(self._on_point_clicked)
        self.left_image_view.point_moved.connect(self._on_point_moved)
        left_layout.addWidget(self.left_image_view)
        image_views_layout.addWidget(left_group)
        
        # Right image view
        right_group = QGroupBox("Right Image")
        right_layout = QVBoxLayout(right_group)
        self.right_image_view = CalibrationGraphicsView(side="right")
        self.right_image_view.point_clicked.connect(self._on_point_clicked)
        self.right_image_view.point_moved.connect(self._on_point_moved)
        right_layout.addWidget(self.right_image_view)
        image_views_layout.addWidget(right_group)
        
        main_layout.addLayout(image_views_layout, 1)  # Stretch factor 1
        
        # Status
        self.status_label = QLabel(Messages.READY)
        main_layout.addWidget(self.status_label)
    
    def _on_load_current_frame(self):
        """Handle load current frame button click."""
        # Get main window reference
        main_window = None
        parent = self.parent()
        while parent:
            if hasattr(parent, 'image_view'):
                main_window = parent
                break
            parent = parent.parent()
        
        if not main_window or not hasattr(main_window, 'image_view'):
            self.show_error("Unable to access current frame images")
            return
        
        try:
            # Get current frame images from the image view
            image_view = main_window.image_view
            
            # Access the stereo view to get left and right images
            stereo_view = getattr(image_view, 'stereo_view', None)
            if not stereo_view:
                self.show_error("Stereo view not available")
                return
                
            # Get the current OpenCV images from stereo view
            # Note: This assumes the stereo view has a way to get the current images
            # If not directly available, we might need to capture from the QImage
            left_img = getattr(stereo_view, 'left_cv_image', None)
            right_img = getattr(stereo_view, 'right_cv_image', None)
            
            if left_img is None or right_img is None:
                self.show_error("No images currently loaded in the main view")
                return
                
            # Set images in the controller
            success = self.controller.set_images(left_img, right_img)
            
            if success:
                self.update_status("Current frame images loaded successfully")
            else:
                self.show_error("Failed to load current frame images")
                
        except Exception as e:
            self.show_error(f"Error loading current frame images: {str(e)}")
            self.logger.error(f"Error loading current frame images: {str(e)}", exc_info=True)
    
    def _on_save_config(self):
        """Handle save configuration button click."""
        # Get config manager
        try:
            from src.utils.config_manager import ConfigManager
            config_manager = ConfigManager()
            
            # Get calibration points from controller
            points = {
                "left_points": self.controller.get_model().left_pts,
                "right_points": self.controller.get_model().right_pts,
                "left_fine_points": self.controller.get_model().left_fine_pts,
                "right_fine_points": self.controller.get_model().right_fine_pts
            }
            
            # Save to config
            config_manager.set("court_calibration", points)
            config_manager.save_config(force=True)
            
            self.update_status("Calibration points saved to config.json")
        except Exception as e:
            self.show_error(f"Error saving configuration: {str(e)}")
            self.logger.error(f"Error saving configuration: {str(e)}", exc_info=True)
    
    def _on_load_config(self):
        """Handle load configuration button click."""
        try:
            from src.utils.config_manager import ConfigManager
            config_manager = ConfigManager()
            
            # Get calibration data from config
            calibration_data = config_manager.get("court_calibration")
            
            if not calibration_data:
                self.show_warning("No calibration data found in configuration")
                return
                
            # Update the controller with the loaded data
            model = self.controller.get_model()
            
            # Ensure we have images loaded before setting points
            if model.left_img is None or model.right_img is None:
                self.show_warning("Please load images first before loading calibration data")
                return
            
            # Clear existing points
            self.controller.clear_points()
            
            # Add points from config
            left_points = calibration_data.get("left_points", [])
            right_points = calibration_data.get("right_points", [])
            
            for point in left_points:
                self.controller.add_point(point, "left")
                
            for point in right_points:
                self.controller.add_point(point, "right")
            
            # Set fine points if available
            left_fine_points = calibration_data.get("left_fine_points", [])
            right_fine_points = calibration_data.get("right_fine_points", [])
            
            if left_fine_points or right_fine_points:
                model.update_fine_points(left_fine_points, right_fine_points)
            
            self.update_status("Calibration data loaded successfully")
            self.update_display()
            
        except Exception as e:
            self.show_error(f"Error loading configuration: {str(e)}")
            self.logger.error(f"Error loading configuration: {str(e)}", exc_info=True)
    
    def _on_clear_points(self):
        """Handle clear points button click."""
        self.controller.clear_points()
        
        # The controller.clear_points will emit calibration_updated which will
        # trigger update_display, so we don't need to manually update UI here
        
        self.update_status("Points cleared")
    
    def _on_show_points_changed(self, state):
        """
        Handle show points checkbox state change.
        
        Args:
            state (int): Checkbox state
        """
        visible = state == Qt.Checked
        self.left_image_view.set_points_visibility(visible)
        self.right_image_view.set_points_visibility(visible)
    
    def _on_show_roi_changed(self, state):
        """
        Handle show ROI checkbox state change.
        
        Args:
            state (int): Checkbox state
        """
        visible = state == Qt.Checked
        self.left_image_view.set_roi_visibility(visible)
        self.right_image_view.set_roi_visibility(visible)
    
    def _on_roi_size_changed(self, value):
        """
        Handle ROI size spinner value change.
        
        Args:
            value (int): New ROI size
        """
        self.controller.set_roi_size(value)
        # Update ROI overlay on both views
        self.left_image_view._update_roi_overlay()
        self.right_image_view._update_roi_overlay()
    
    def _on_tune_points(self):
        """Handle fine-tune button click."""
        # Check if we have enough points
        model = self.controller.get_model()
        left_count = len(model.left_pts)
        right_count = len(model.right_pts)
        
        if left_count < 4 or right_count < 4:
            self.show_warning("At least 4 points per side are required for calibration")
            return
        
        # Start the fine-tuning process
        try:
            # Request tuning from controller
            self.controller.request_tuning()
            self.update_status("Fine-tuning process initiated. Please wait...")
        except Exception as e:
            self.show_error(f"Error during fine-tuning: {str(e)}")
            self.logger.error(f"Error during fine-tuning: {str(e)}", exc_info=True)
    
    def _on_point_clicked(self, point, side):
        """
        Handle point clicked in image view.
        
        Args:
            point (tuple): (x, y) coordinates
            side (str): "left" or "right" indicating which image was clicked
        """
        # Get current point counts from the model
        model = self.controller.get_model()
        left_count = len(model.left_pts)
        right_count = len(model.right_pts)
        
        # Check if we've reached the max points
        if side == "left" and left_count >= 14:
            self.update_status(f"Maximum points (14) reached for left image")
            return
        elif side == "right" and right_count >= 14:
            self.update_status(f"Maximum points (14) reached for right image")
            return
            
        # Add point to controller with side information
        self.controller.add_point(point, side)
        
        # Update status - do not manually add markers here
        # Let update_display handle the display of markers
        self.update_status(f"Point added at {point} ({side})")
        
        # Note: We don't need to manually add markers or update point counters
        # because the controller.add_point will emit calibration_updated 
        # which will trigger update_display
    
    def _on_point_moved(self, index, new_pos, side):
        """
        Handle point moved in image view.
        
        Args:
            index (int): Index of the moved point
            new_pos (tuple): New (x, y) coordinates
            side (str): "left" or "right" indicating which side the point belongs to
        """
        model = self.controller.get_model()
        
        # Update the point in the model
        if side == "left" and 0 <= index < len(model.left_pts):
            model.left_pts[index] = new_pos
        elif side == "right" and 0 <= index < len(model.right_pts):
            model.right_pts[index] = new_pos
        
        # Notify controller of the update
        self.controller.calibration_updated.emit()
        
        self.update_status(f"Point {index + 1} moved to {new_pos} ({side})")
    
    @Slot()
    def update_display(self):
        """Update the display with current model data."""
        model = self.controller.get_model()
        
        # Update left image if available
        left_img = model.left_img
        if left_img is not None:
            self.left_image_view.set_image(left_img)
            
        # Update right image if available
        right_img = model.right_img
        if right_img is not None:
            self.right_image_view.set_image(right_img)
            
        # Update points
        self.left_image_view.clear_points()
        self.right_image_view.clear_points()
        
        # Count points
        left_count = len(model.left_pts)
        right_count = len(model.right_pts)
        
        # Add points
        for i, pt in enumerate(model.left_pts):
            self.left_image_view.add_point_marker(pt, is_fine_point=False, index=i)
            
        for i, pt in enumerate(model.right_pts):
            self.right_image_view.add_point_marker(pt, is_fine_point=False, index=i)
            
        # Update point counters directly from model data
        self.left_count_label.setText(f"{left_count}/14")
        self.right_count_label.setText(f"{right_count}/14")
    
    @Slot(str)
    def show_error(self, message):
        """
        Show an error message.
        
        Args:
            message (str): Error message
        """
        QMessageBox.critical(self, "Error", message)
        self.status_label.setText(f"Error: {message}")
    
    @Slot(str)
    def show_warning(self, message):
        """
        Show a warning message.
        
        Args:
            message (str): Warning message
        """
        QMessageBox.warning(self, "Warning", message)
        self.status_label.setText(f"Warning: {message}")
    
    def update_status(self, message):
        """
        Update the status label.
        
        Args:
            message (str): Status message
        """
        self.status_label.setText(message)
    
    @Slot()
    def processing_started(self):
        """Handle processing started signal."""
        # Create and show progress dialog
        self.progress_dialog = QProgressDialog("Processing points...", "Cancel", 0, 0, self)
        self.progress_dialog.setWindowTitle("Fine-Tuning")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.show()
        
        # Disable controls
        self._set_controls_enabled(False)
        
        self.update_status("Processing points...")
    
    @Slot()
    def processing_completed(self):
        """Handle processing completed signal."""
        # Close progress dialog
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
            
        # Enable controls
        self._set_controls_enabled(True)
    
    @Slot(list)
    def tuning_completed(self, fine_pts):
        """
        Handle tuning completed signal.
        
        Args:
            fine_pts (list): Fine-tuned points
        """
        self.update_status(f"Fine-tuning completed: {len(fine_pts)} points processed")
    
    def _set_controls_enabled(self, enabled):
        """
        Enable or disable controls.
        
        Args:
            enabled (bool): True to enable, False to disable
        """
        self.load_frame_btn.setEnabled(enabled)
        self.save_config_btn.setEnabled(enabled)
        self.load_config_btn.setEnabled(enabled)
        self.clear_points_btn.setEnabled(enabled)
        self.tune_btn.setEnabled(enabled)
        self.roi_size_spinner.setEnabled(enabled) 