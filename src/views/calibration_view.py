#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calibration View module.
This module contains the CalibrationView class which provides the UI for court calibration.
"""

import logging
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, Signal, Slot, QPointF
from PySide6.QtGui import QPen, QBrush, QColor, QPainter
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGraphicsScene, QGraphicsView,
    QGraphicsEllipseItem, QGraphicsPathItem, QPushButton, QLabel,
    QFileDialog, QMessageBox
)

from src.utils.ui_constants import CalibrationTab, Layout


class CalibrationView(QWidget):
    """
    View for court calibration featuring dual image views and point calibration.
    """
    
    # Signals
    point_added = Signal(str, QPointF)  # side, position
    point_moved = Signal(str, int, QPointF)  # side, index, new_position
    fine_tune_requested = Signal()
    save_calibration_requested = Signal(str)  # file_path
    load_calibration_requested = Signal(str)  # file_path
    clear_points_requested = Signal()
    load_images_requested = Signal(str, str)  # left_image_path, right_image_path
    load_current_frame_requested = Signal()  # Request to load the current frame
    
    def __init__(self, parent=None):
        """
        Initialize the calibration view.
        
        Args:
            parent (QWidget, optional): Parent widget
        """
        super(CalibrationView, self).__init__(parent)
        
        # Initialize scenes and views
        self.left_scene = QGraphicsScene(self)
        self.right_scene = QGraphicsScene(self)
        
        # Initialize points storage - will be managed by controller but kept here for UI
        self.left_points: List[QGraphicsEllipseItem] = []
        self.right_points: List[QGraphicsEllipseItem] = []
        
        # Initialize connection lines
        self.left_path_item: Optional[QGraphicsPathItem] = None
        self.right_path_item: Optional[QGraphicsPathItem] = None
        
        # Set up UI
        self._setup_ui()
        
    def _setup_ui(self):
        """Set up the user interface."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(Layout.MARGIN, Layout.MARGIN, Layout.MARGIN, Layout.MARGIN)
        main_layout.setSpacing(Layout.SPACING)
        
        # Image views
        views_layout = QHBoxLayout()
        
        # Left view with label
        left_view_layout = QVBoxLayout()
        left_label = QLabel("Left Image")
        left_label.setAlignment(Qt.AlignCenter)
        self.left_view = QGraphicsView(self.left_scene)
        self.left_view.setRenderHint(QPainter.Antialiasing)
        left_view_layout.addWidget(left_label)
        left_view_layout.addWidget(self.left_view)
        
        # Right view with label
        right_view_layout = QVBoxLayout()
        right_label = QLabel("Right Image")
        right_label.setAlignment(Qt.AlignCenter)
        self.right_view = QGraphicsView(self.right_scene)
        self.right_view.setRenderHint(QPainter.Antialiasing)
        right_view_layout.addWidget(right_label)
        right_view_layout.addWidget(self.right_view)
        
        views_layout.addLayout(left_view_layout)
        views_layout.addLayout(right_view_layout)
        
        main_layout.addLayout(views_layout)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(Layout.SPACING)
        
        # Load Images button
        self.load_images_button = QPushButton("Load Current Frame")
        self.load_images_button.setFixedSize(CalibrationTab.BUTTON_WIDTH, CalibrationTab.BUTTON_HEIGHT)
        self.load_images_button.clicked.connect(self._on_load_current_frame)
        buttons_layout.addWidget(self.load_images_button)
        
        # Save Calibration button
        self.save_calib_button = QPushButton("Save Calibration")
        self.save_calib_button.setFixedSize(CalibrationTab.BUTTON_WIDTH, CalibrationTab.BUTTON_HEIGHT)
        self.save_calib_button.clicked.connect(self._on_save_calibration)
        buttons_layout.addWidget(self.save_calib_button)
        
        # Load Calibration button
        self.load_calib_button = QPushButton("Load Calibration")
        self.load_calib_button.setFixedSize(CalibrationTab.BUTTON_WIDTH, CalibrationTab.BUTTON_HEIGHT)
        self.load_calib_button.clicked.connect(self._on_load_calibration)
        buttons_layout.addWidget(self.load_calib_button)
        
        # Clear Points button
        self.clear_button = QPushButton("Clear Points")
        self.clear_button.setFixedSize(CalibrationTab.BUTTON_WIDTH, CalibrationTab.BUTTON_HEIGHT)
        self.clear_button.clicked.connect(self._on_clear_points)
        buttons_layout.addWidget(self.clear_button)
        
        # Fine Tune button
        self.fine_tune_button = QPushButton("Fine Tune")
        self.fine_tune_button.setFixedSize(CalibrationTab.BUTTON_WIDTH, CalibrationTab.BUTTON_HEIGHT)
        self.fine_tune_button.clicked.connect(self._on_fine_tune)
        buttons_layout.addWidget(self.fine_tune_button)
        
        main_layout.addLayout(buttons_layout)
        
        # Connect scene signals
        self.left_scene.mouseReleaseEvent = lambda event: self._on_scene_click(event, "left")
        self.right_scene.mouseReleaseEvent = lambda event: self._on_scene_click(event, "right")
    
    def _on_scene_click(self, event, side):
        """
        Handle mouse click on scene to add calibration point.
        
        Args:
            event (QGraphicsSceneMouseEvent): Mouse event
            side (str): 'left' or 'right' to indicate which scene was clicked
        """
        scene = self.left_scene if side == "left" else self.right_scene
        points = self.left_points if side == "left" else self.right_points
        
        # Check if we've reached the maximum number of points
        if len(points) >= CalibrationTab.MAX_POINTS:
            QMessageBox.warning(
                self, 
                "Maximum Points Reached", 
                f"Cannot add more than {CalibrationTab.MAX_POINTS} calibration points."
            )
            return
        
        # Get position in scene coordinates
        pos = event.scenePos()
        
        # Add the point visually
        self._add_point_to_scene(side, pos)
        
        # Emit signal for controller
        self.point_added.emit(side, pos)
    
    def _add_point_to_scene(self, side, pos):
        """
        Add a point to the scene.
        
        Args:
            side (str): 'left' or 'right' to indicate which scene
            pos (QPointF): Position in scene coordinates
        """
        scene = self.left_scene if side == "left" else self.right_scene
        points = self.left_points if side == "left" else self.right_points
        
        # Create ellipse item
        radius = CalibrationTab.POINT_RADIUS
        ellipse = QGraphicsEllipseItem(pos.x() - radius, pos.y() - radius, radius * 2, radius * 2)
        ellipse.setBrush(QBrush(QColor(*CalibrationTab.POINT_COLOR_ORIGINAL)))
        ellipse.setPen(QPen(QColor(*CalibrationTab.POINT_COLOR_ORIGINAL)))
        ellipse.setZValue(CalibrationTab.Z_VALUE_ORIGINAL)
        ellipse.setFlag(QGraphicsEllipseItem.ItemIsMovable)
        ellipse.setFlag(QGraphicsEllipseItem.ItemSendsGeometryChanges)
        
        # Store point index for reference
        ellipse.setData(0, len(points))
        ellipse.setData(1, side)
        
        # Add to scene and store
        scene.addItem(ellipse)
        points.append(ellipse)
        
        # Connect item movement
        ellipse.itemChange = lambda change, value: self._on_item_moved(ellipse, change, value)
        
        # Update connection lines
        self._update_connection_lines(side)
    
    def _on_item_moved(self, item, change, value):
        """
        Handle item movement.
        
        Args:
            item (QGraphicsEllipseItem): Item being moved
            change (GraphicsItemChange): Change type
            value: Change value
            
        Returns:
            Value to be used by the item
        """
        if change == QGraphicsEllipseItem.ItemPositionChange:
            # Item is being moved, get index and side
            index = item.data(0)
            side = item.data(1)
            
            # Emit signal for controller
            self.point_moved.emit(side, index, value)
            
            # Update connection lines
            self._update_connection_lines(side)
        
        return value
    
    def _update_connection_lines(self, side):
        """
        Update connection lines between points.
        
        Args:
            side (str): 'left' or 'right' to indicate which scene
        """
        points = self.left_points if side == "left" else self.right_points
        scene = self.left_scene if side == "left" else self.right_scene
        path_item = self.left_path_item if side == "left" else self.right_path_item
        
        # Need at least 2 points to draw lines
        if len(points) < 2:
            if path_item:
                scene.removeItem(path_item)
                if side == "left":
                    self.left_path_item = None
                else:
                    self.right_path_item = None
            return
        
        # Create path connecting points
        # Implementation will depend on specific connection rules
        # For now, simple connection in order added
        # TODO: Implement proper court line connections
        pass
    
    def _on_load_current_frame(self):
        """Handle load current frame button click."""
        # Simply emit signal to request loading the current frame
        self.load_current_frame_requested.emit()
    
    def _on_save_calibration(self):
        """Handle save calibration button click."""
        # Use config directly or prompt for file path
        # We'll emit signal either way and let controller decide
        self.save_calibration_requested.emit("")
    
    def _on_load_calibration(self):
        """Handle load calibration button click."""
        # Use config directly or prompt for file path
        # We'll emit signal either way and let controller decide
        self.load_calibration_requested.emit("")
    
    def _on_clear_points(self):
        """Handle clear points button click."""
        # Confirm with user
        response = QMessageBox.question(
            self, 
            "Clear Points", 
            "Are you sure you want to clear all calibration points?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if response == QMessageBox.Yes:
            # Emit signal for controller
            self.clear_points_requested.emit()
    
    def _on_fine_tune(self):
        """Handle fine tune button click."""
        # Emit signal for controller
        self.fine_tune_requested.emit()
    
    def set_left_image(self, pixmap):
        """
        Set the left image.
        
        Args:
            pixmap (QPixmap): Image pixmap
        """
        self.left_scene.clear()
        self.left_scene.addPixmap(pixmap)
        self.left_view.fitInView(self.left_scene.sceneRect(), Qt.KeepAspectRatio)
        self._rebuild_points("left")
    
    def set_right_image(self, pixmap):
        """
        Set the right image.
        
        Args:
            pixmap (QPixmap): Image pixmap
        """
        self.right_scene.clear()
        self.right_scene.addPixmap(pixmap)
        self.right_view.fitInView(self.right_scene.sceneRect(), Qt.KeepAspectRatio)
        self._rebuild_points("right")
    
    def update_point(self, side, index, position, is_fine_tuned=False):
        """
        Update point position and appearance.
        
        Args:
            side (str): 'left' or 'right'
            index (int): Point index
            position (QPointF): New position
            is_fine_tuned (bool): Whether point is fine-tuned
        """
        points = self.left_points if side == "left" else self.right_points
        
        if index < len(points):
            point = points[index]
            radius = CalibrationTab.POINT_RADIUS
            
            # Update position
            point.setRect(position.x() - radius, position.y() - radius, radius * 2, radius * 2)
            
            # Update appearance based on fine-tuned status
            color = CalibrationTab.POINT_COLOR_FINE_TUNED if is_fine_tuned else CalibrationTab.POINT_COLOR_ORIGINAL
            point.setBrush(QBrush(QColor(*color)))
            point.setPen(QPen(QColor(*color)))
            point.setZValue(CalibrationTab.Z_VALUE_FINE_TUNED if is_fine_tuned else CalibrationTab.Z_VALUE_ORIGINAL)
            
            # Update connection lines
            self._update_connection_lines(side)
    
    def clear_points(self):
        """Clear all calibration points."""
        # Left scene
        for point in self.left_points:
            self.left_scene.removeItem(point)
        self.left_points.clear()
        
        if self.left_path_item:
            self.left_scene.removeItem(self.left_path_item)
            self.left_path_item = None
        
        # Right scene
        for point in self.right_points:
            self.right_scene.removeItem(point)
        self.right_points.clear()
        
        if self.right_path_item:
            self.right_scene.removeItem(self.right_path_item)
            self.right_path_item = None
    
    def _rebuild_points(self, side):
        """
        Rebuild points after clearing the scene.
        
        Args:
            side (str): 'left' or 'right'
        """
        # Get the scene and points list based on side
        scene = self.left_scene if side == "left" else self.right_scene
        points_list = self.left_points if side == "left" else self.right_points
        
        # Check if there's a controller reference
        controller = getattr(self, "controller", None)
        if not controller or not hasattr(controller, "model"):
            logging.warning(f"Cannot rebuild points for {side} side: No controller or model available")
            return
        
        # Get points from model
        model_points = controller.model.get_points(side)
        
        # Clear existing points from the list
        for point in points_list[:]:
            scene.removeItem(point)
        points_list.clear()
        
        # Add points from model data
        for idx, point_data in enumerate(model_points):
            position = point_data['position']
            is_fine_tuned = point_data.get('is_fine_tuned', False)
            
            # Create visual representation
            radius = CalibrationTab.POINT_RADIUS
            ellipse = QGraphicsEllipseItem(position.x() - radius, position.y() - radius, radius * 2, radius * 2)
            
            # Set appearance
            color = CalibrationTab.POINT_COLOR_FINE_TUNED if is_fine_tuned else CalibrationTab.POINT_COLOR_ORIGINAL
            ellipse.setBrush(QBrush(QColor(*color)))
            ellipse.setPen(QPen(QColor(*color)))
            ellipse.setZValue(CalibrationTab.Z_VALUE_FINE_TUNED if is_fine_tuned else CalibrationTab.Z_VALUE_ORIGINAL)
            ellipse.setFlag(QGraphicsEllipseItem.ItemIsMovable)
            ellipse.setFlag(QGraphicsEllipseItem.ItemSendsGeometryChanges)
            
            # Store metadata
            ellipse.setData(0, idx)
            ellipse.setData(1, side)
            
            # Add to scene and list
            scene.addItem(ellipse)
            points_list.append(ellipse)
            
            # Connect move event
            ellipse.itemChange = lambda change, value: self._on_item_moved(ellipse, change, value)
        
        # Update connection lines
        self._update_connection_lines(side)
    
    def resizeEvent(self, event):
        """
        Handle resize event to maintain view fitting.
        
        Args:
            event (QResizeEvent): Resize event
        """
        super(CalibrationView, self).resizeEvent(event)
        
        # Fit images in view while maintaining aspect ratio
        self.left_view.fitInView(self.left_scene.sceneRect(), Qt.KeepAspectRatio)
        self.right_view.fitInView(self.right_scene.sceneRect(), Qt.KeepAspectRatio) 