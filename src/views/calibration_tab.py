#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calibration Tab module.
This module contains the CalibrationTab class which provides the UI for court calibration.
"""

import logging
from typing import Dict, List, Tuple

from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QPen, QBrush, QColor, QPainterPath, QPainter, QCursor, QPixmap, QImage
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsRectItem,
    QGraphicsPathItem, QSplitter
)

from src.utils.ui_constants import Layout, WindowSize
from src.utils.ui_theme import Colors, StyleManager
from src.views.visualization import VisualizerFactory

logger = logging.getLogger(__name__)

class CalibrationPoint(QGraphicsEllipseItem):
    """
    Custom QGraphicsEllipseItem for calibration points.
    Supports moving and tracks its index.
    """
    
    def __init__(self, x: float, y: float, index: int, radius: float = 10.0):
        """
        Initialize a calibration point.
        
        Args:
            x (float): X-coordinate
            y (float): Y-coordinate
            index (int): Point index
            radius (float): Point radius (default increased from 5.0 to 10.0)
        """
        super().__init__(x - radius, y - radius, radius * 2, radius * 2)
        self.setFlag(QGraphicsEllipseItem.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsEllipseItem.ItemIsSelectable, True)
        self.setBrush(QBrush(Colors.ACCENT))
        self.setPen(QPen(Colors.ACCENT.darker(120), 2))  # Increased pen width for better visibility
        self.setZValue(1.0)  # Make points appear above the image
        self.index = index
        self.radius = radius
        
        # Set cursor to pointing hand
        self.setCursor(QCursor(Qt.PointingHandCursor))
        
    def itemChange(self, change, value):
        """
        Handle position change events to keep track of the point's position.
        
        Args:
            change: The type of change
            value: The new value
            
        Returns:
            The processed value
        """
        if change == QGraphicsEllipseItem.ItemPositionChange and self.scene():
            # Get the parent view for this item's scene
            for view in self.scene().views():
                if isinstance(view.parent().parent(), CalibrationTab):
                    parent = view.parent().parent()
                    x = value.x() + self.radius
                    y = value.y() + self.radius
                    
                    # Determine which view this point belongs to
                    if view == parent.left_view:
                        side = 'left'
                    else:
                        side = 'right'
                    
                    # Emit the point moved signal
                    parent.point_moved.emit(side, self.index, x, y)
            
        return super().itemChange(change, value)


class CalibrationTab(QWidget):
    """
    Tab for court calibration with left and right image views.
    """
    
    # Signal emitted when a point is added (side, x, y)
    point_added = Signal(str, float, float)
    
    # Signal emitted when a point is moved (side, index, x, y)
    point_moved = Signal(str, int, float, float)
    
    def __init__(self, parent=None):
        """
        Initialize the calibration tab.
        
        Args:
            parent (QWidget, optional): Parent widget
        """
        super().__init__(parent)
        
        # Set up the UI
        self._setup_ui()
        
        # Initialize dictionaries to store point items
        self.left_points: Dict[int, CalibrationPoint] = {}
        self.right_points: Dict[int, CalibrationPoint] = {}
        
        # Initialize ROI overlays
        self.left_roi_overlay: QGraphicsRectItem = None
        self.right_roi_overlay: QGraphicsRectItem = None
        
        # Initialize grid lines
        self.left_grid_lines: List[QGraphicsPathItem] = []
        self.right_grid_lines: List[QGraphicsPathItem] = []
        
        # Initialize image pixmaps
        self.left_pixmap = None
        self.right_pixmap = None
        
        # Initialize image dimensions
        self.left_image_width = 0
        self.left_image_height = 0
        self.right_image_width = 0
        self.right_image_height = 0
    
    def _setup_ui(self):
        """Set up the user interface."""
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(Layout.MARGIN, Layout.MARGIN, Layout.MARGIN, Layout.MARGIN)
        main_layout.setSpacing(Layout.SPACING)
        
        # Create splitter for left and right views
        splitter = QSplitter(Qt.Horizontal)
        
        # Create left view container
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        left_label = QLabel("Left Image")
        left_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(left_label)
        
        self.left_scene = QGraphicsScene()
        self.left_view = QGraphicsView(self.left_scene)
        self.left_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.left_view.setDragMode(QGraphicsView.NoDrag)
        left_layout.addWidget(self.left_view)
        
        # Create right view container
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        right_label = QLabel("Right Image")
        right_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(right_label)
        
        self.right_scene = QGraphicsScene()
        self.right_view = QGraphicsView(self.right_scene)
        self.right_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.right_view.setDragMode(QGraphicsView.NoDrag)
        right_layout.addWidget(self.right_view)
        
        # Add containers to splitter
        splitter.addWidget(left_container)
        splitter.addWidget(right_container)
        splitter.setSizes([WindowSize.DEFAULT_WIDTH // 2, WindowSize.DEFAULT_WIDTH // 2])
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
        # Add button layout
        button_layout = QHBoxLayout()
        button_layout.setSpacing(Layout.SPACING)
        
        # Create buttons
        self.clear_button = QPushButton("Clear Points")
        self.clear_button.setToolTip("Clear all calibration points")
        
        self.fine_tune_button = QPushButton("Fine-Tune Points")
        self.fine_tune_button.setToolTip("Automatically fine-tune point positions")
        
        self.save_button = QPushButton("Save Configuration")
        self.save_button.setToolTip("Save calibration points to configuration file")
        
        self.load_button = QPushButton("Load Configuration")
        self.load_button.setToolTip("Load calibration points from configuration file")
        
        self.load_current_frame_button = QPushButton("Load Current Frame")
        self.load_current_frame_button.setToolTip("Load current frame images from the player")
        
        # Add buttons to layout
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.fine_tune_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.load_current_frame_button)
        
        # Add button layout to main layout
        main_layout.addLayout(button_layout)
        
        # Set styles
        self.setStyleSheet("")  # Using global application styles instead of custom stylesheet
        
        # Connect signals for mouse clicks
        self.left_view.mousePressEvent = lambda event: self._handle_mouse_press(event, self.left_view, 'left')
        self.right_view.mousePressEvent = lambda event: self._handle_mouse_press(event, self.right_view, 'right')
    
    def _handle_mouse_press(self, event, view, side):
        """
        Handle mouse press events in the graphics views.
        
        Args:
            event: Mouse event
            view: The view that received the event
            side: 'left' or 'right'
        """
        # Call the parent class's mousePressEvent to maintain default behavior
        QGraphicsView.mousePressEvent(view, event)
        
        # Get the position in the view
        pos = event.position().toPoint()
        
        # Convert view coordinates to scene coordinates
        scene_pos = view.mapToScene(pos)
        x, y = scene_pos.x(), scene_pos.y()
        
        # Check if there's an item at this position
        item = view.scene().itemAt(scene_pos, view.transform())
        
        # If not clicking on a movable item (like a calibration point), add a new point
        if not item or not item.flags() & QGraphicsEllipseItem.ItemIsMovable:
            # Emit signal for the new point
            logger.debug(f"Emitting point_added signal: {side}, {x}, {y}")
            self.point_added.emit(side, x, y)
    
    def add_point(self, side: str, point_id: int, scene_x: float, scene_y: float):
        """
        Add a calibration point to the specified view.
        
        Args:
            side (str): 'left' or 'right'
            point_id (int): Point identifier
            scene_x (float): X-coordinate in scene coordinates
            scene_y (float): Y-coordinate in scene coordinates
        """
        point = CalibrationPoint(scene_x, scene_y, point_id)
        
        if side == 'left':
            self.left_scene.addItem(point)
            self.left_points[point_id] = point
        elif side == 'right':
            self.right_scene.addItem(point)
            self.right_points[point_id] = point
        else:
            logger.error(f"Invalid side: {side}")
            return

    def update_point(self, side: str, point_id: int, scene_x: float, scene_y: float):
        """
        Update an existing calibration point's position.
        
        Args:
            side (str): 'left' or 'right'
            point_id (int): Point identifier
            scene_x (float): New X-coordinate in scene coordinates
            scene_y (float): New Y-coordinate in scene coordinates
        """
        if side == 'left' and point_id in self.left_points:
            point = self.left_points[point_id]
            # Update position, accounting for point radius
            point.setPos(scene_x - point.radius, scene_y - point.radius)
        elif side == 'right' and point_id in self.right_points:
            point = self.right_points[point_id]
            # Update position, accounting for point radius
            point.setPos(scene_x - point.radius, scene_y - point.radius)
        else:
            logger.error(f"Invalid side or point_id not found: {side}, {point_id}")
    
    def add_point_item(self, side: str, x: float, y: float, index: int):
        """
        Add a calibration point item to the specified view.
        
        Args:
            side (str): 'left' or 'right'
            x (float): X-coordinate
            y (float): Y-coordinate
            index (int): Point index
        """
        point = CalibrationPoint(x, y, index)
        
        if side == 'left':
            self.left_scene.addItem(point)
            self.left_points[index] = point
        elif side == 'right':
            self.right_scene.addItem(point)
            self.right_points[index] = point
        else:
            logger.error(f"Invalid side: {side}")
            return
    
    def update_point_item(self, side: str, index: int, x: float, y: float):
        """
        Update an existing calibration point item's position.
        
        Args:
            side (str): 'left' or 'right'
            index (int): Point index
            x (float): New X-coordinate
            y (float): New Y-coordinate
        """
        if side == 'left' and index in self.left_points:
            point = self.left_points[index]
            # Block signals to prevent recursion
            point.setPos(x - point.radius, y - point.radius)
        elif side == 'right' and index in self.right_points:
            point = self.right_points[index]
            # Block signals to prevent recursion
            point.setPos(x - point.radius, y - point.radius)
        else:
            logger.error(f"Invalid side or index not found: {side}, {index}")
    
    def clear_points(self):
        """Clear all calibration points from both views."""
        # Remove all point items from the scenes
        for point in self.left_points.values():
            self.left_scene.removeItem(point)
        for point in self.right_points.values():
            self.right_scene.removeItem(point)
        
        # Clear the point dictionaries
        self.left_points.clear()
        self.right_points.clear()
        
        # Clear grid lines
        self._clear_grid_lines()
    
    def _clear_grid_lines(self):
        """Clear grid lines from both views."""
        # Remove all grid lines from the scenes
        for line in self.left_grid_lines:
            self.left_scene.removeItem(line)
        for line in self.right_grid_lines:
            self.right_scene.removeItem(line)
        
        # Clear the grid line lists
        self.left_grid_lines.clear()
        self.right_grid_lines.clear()
    
    def show_roi(self, side: str, center: Tuple[float, float], radius: float):
        """
        Show ROI overlay around a point.
        
        Args:
            side (str): 'left' or 'right'
            center (Tuple[float, float]): Center coordinates (x, y)
            radius (float): ROI radius
        """
        from src.views.visualization import VisualizerFactory
        
        # Calculate ROI coordinates
        x, y = center
        roi = (x - radius, y - radius, radius * 2, radius * 2)
        
        # Remove existing overlay if any
        if side == 'left':
            if self.left_roi_overlay:
                self.left_scene.removeItem(self.left_roi_overlay)
                self.left_roi_overlay = None
            
            # Create visualizer and draw ROI
            visualizer = VisualizerFactory.create(backend="qt", scene=self.left_scene)
            self.left_scene = visualizer.draw_roi(
                self.left_scene, 
                roi, 
                color=Colors.ACCENT, 
                show_center=True
            )
            
            # Find and store the created ROI item
            for item in self.left_scene.items():
                if isinstance(item, QGraphicsRectItem) and item != self.left_roi_overlay:
                    self.left_roi_overlay = item
                    break
        elif side == 'right':
            if self.right_roi_overlay:
                self.right_scene.removeItem(self.right_roi_overlay)
                self.right_roi_overlay = None
            
            # Create visualizer and draw ROI
            visualizer = VisualizerFactory.create(backend="qt", scene=self.right_scene)
            self.right_scene = visualizer.draw_roi(
                self.right_scene, 
                roi, 
                color=Colors.ACCENT, 
                show_center=True
            )
            
            # Find and store the created ROI item
            for item in self.right_scene.items():
                if isinstance(item, QGraphicsRectItem) and item != self.right_roi_overlay:
                    self.right_roi_overlay = item
                    break
    
    def hide_roi(self, side: str):
        """
        Hide ROI overlay.
        
        Args:
            side (str): 'left' or 'right'
        """
        if side == 'left' and self.left_roi_overlay:
            self.left_scene.removeItem(self.left_roi_overlay)
            self.left_roi_overlay = None
        elif side == 'right' and self.right_roi_overlay:
            self.right_scene.removeItem(self.right_roi_overlay)
            self.right_roi_overlay = None
    
    def draw_grid_lines(self, side: str, points: List[Tuple[float, float]], rows: int, cols: int):
        """
        Draw grid lines connecting calibration points in a grid pattern.
        
        Args:
            side (str): 'left' or 'right'
            points (List[Tuple[float, float]]): List of (x, y) point coordinates
            rows (int): Number of rows in the grid
            cols (int): Number of columns in the grid
        """
        from src.views.visualization import VisualizerFactory
        
        # Clear existing grid lines for this side
        if side == 'left':
            for line in self.left_grid_lines:
                self.left_scene.removeItem(line)
            self.left_grid_lines.clear()
            scene = self.left_scene
        else:
            for line in self.right_grid_lines:
                self.right_scene.removeItem(line)
            self.right_grid_lines.clear()
            scene = self.right_scene
        
        # Create Qt visualizer using the factory
        visualizer = VisualizerFactory.create(backend="qt", scene=scene)
        
        # Draw grid lines using the visualizer
        # Note: The visualizer returns the scene, but also stores created items internally
        scene = visualizer.draw_grid_lines(scene, points, rows, cols, 
                                          color=Colors.WARNING, 
                                          thickness=1, 
                                          dashed=True)
        
        # Add references to all lines created by the visualizer
        if side == 'left':
            # We need to find the newly added items in the scene
            all_items = scene.items()
            existing_items = set(self.left_points.values())
            if self.left_roi_overlay:
                existing_items.add(self.left_roi_overlay)
            
            # Add newly created line items to our tracking list
            for item in all_items:
                if isinstance(item, QGraphicsPathItem) and item not in existing_items and item not in self.left_grid_lines:
                    self.left_grid_lines.append(item)
        else:
            # Similar logic for right side
            all_items = scene.items()
            existing_items = set(self.right_points.values())
            if self.right_roi_overlay:
                existing_items.add(self.right_roi_overlay)
            
            for item in all_items:
                if isinstance(item, QGraphicsPathItem) and item not in existing_items and item not in self.right_grid_lines:
                    self.right_grid_lines.append(item)
    
    def set_images(self, left_image, right_image):
        """
        Set the images for the left and right views.
        
        Args:
            left_image: QPixmap or QImage for the left view
            right_image: QPixmap or QImage for the right view
        """
        # Clear scenes first (but keep points)
        self.left_scene.clear()
        self.right_scene.clear()
        
        # Reset point items and grid lines (they were removed when clearing the scenes)
        self.left_points.clear()
        self.right_points.clear()
        self.left_grid_lines.clear()
        self.right_grid_lines.clear()
        self.left_roi_overlay = None
        self.right_roi_overlay = None
        
        # Convert to QPixmap if needed
        if isinstance(left_image, QImage):
            left_image = QPixmap.fromImage(left_image)
        if isinstance(right_image, QImage):
            right_image = QPixmap.fromImage(right_image)
        
        # Store pixmaps for later reference
        self.left_pixmap = left_image
        self.right_pixmap = right_image
        
        # Store image dimensions
        self.left_image_width = left_image.width()
        self.left_image_height = left_image.height()
        self.right_image_width = right_image.width()
        self.right_image_height = right_image.height()
        
        logger.info(f"Set images - Left: {self.left_image_width}x{self.left_image_height}, "
                   f"Right: {self.right_image_width}x{self.right_image_height}")
        
        # Add images to scenes
        self.left_scene.addPixmap(left_image)
        self.right_scene.addPixmap(right_image)
        
        # Fit views to content
        self.left_view.fitInView(self.left_scene.sceneRect(), Qt.KeepAspectRatio)
        self.right_view.fitInView(self.right_scene.sceneRect(), Qt.KeepAspectRatio)
    
    def get_image_dimension(self, side: str, dimension: str) -> int:
        """
        Get the dimension of an image.
        
        Args:
            side (str): 'left' or 'right'
            dimension (str): 'width' or 'height'
            
        Returns:
            int: The requested dimension
        """
        if side == 'left':
            if dimension == 'width':
                return self.left_image_width
            elif dimension == 'height':
                return self.left_image_height
        elif side == 'right':
            if dimension == 'width':
                return self.right_image_width
            elif dimension == 'height':
                return self.right_image_height
        
        logger.error(f"Invalid side or dimension: {side}, {dimension}")
        return 0
    
    def get_view_width(self, side: str) -> float:
        """
        Get the width of a view.
        
        Args:
            side (str): 'left' or 'right'
            
        Returns:
            float: The view width
        """
        if side == 'left':
            return self.left_scene.width()
        elif side == 'right':
            return self.right_scene.width()
        
        logger.error(f"Invalid side: {side}")
        return 0.0
    
    def get_view_height(self, side: str) -> float:
        """
        Get the height of a view.
        
        Args:
            side (str): 'left' or 'right'
            
        Returns:
            float: The view height
        """
        if side == 'left':
            return self.left_scene.height()
        elif side == 'right':
            return self.right_scene.height()
        
        logger.error(f"Invalid side: {side}")
        return 0.0
    
    def update_grid_lines(self):
        """Update grid lines for both views."""
        # This is a placeholder for compatibility with the controller
        # Actual implementation will be in the controller
        pass
    
    def resizeEvent(self, event):
        """Handle resize events to maintain aspect ratio."""
        super().resizeEvent(event)
        
        # Maintain aspect ratio when resizing
        if self.left_pixmap:
            self.left_view.fitInView(self.left_scene.sceneRect(), Qt.KeepAspectRatio)
        if self.right_pixmap:
            self.right_view.fitInView(self.right_scene.sceneRect(), Qt.KeepAspectRatio) 