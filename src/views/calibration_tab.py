#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calibration Tab module.
This module contains the CalibrationTab class which provides the UI for court calibration.
"""

import logging
from typing import Dict, List, Tuple

from PySide6.QtCore import Qt, Signal, QRectF, QPointF, QObject, QEvent, QLine
from PySide6.QtGui import QPen, QBrush, QColor, QPainterPath, QPainter, QPixmap, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsRectItem,
    QGraphicsPathItem, QSplitter, QGraphicsItem, QGraphicsObject, QGraphicsLineItem, QGraphicsSimpleTextItem
)

from src.utils.ui_constants import Layout, WindowSize
from src.utils.ui_theme import Colors, StyleManager

logger = logging.getLogger(__name__)

class CalibrationPoint(QGraphicsEllipseItem):
    """
    Represents a calibration point that can be dragged on the scene.
    """
    # Constants
    RADIUS = 12  # 포인터 크기 증가
    HOVER_RADIUS = 15  # hover 시 크기

    def __init__(self, x, y, index, side, radius=None, parent=None):
        """
        Initialize a calibration point.

        Args:
            x (float): X-coordinate
            y (float): Y-coordinate
            index (int): Index of the point
            side (str): Side ('left' or 'right')
            radius (float): Radius of the point
            parent (QGraphicsItem): Parent item
        """
        radius = radius or self.RADIUS
        super().__init__(x - radius, y - radius, 2 * radius, 2 * radius, parent)
        
        self.index = index
        self.side = side
        self.radius = radius
        self.point_id = None  # Will be set externally (p00, p01, etc.)
        
        # Make it draggable and selectable
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        
        # Set appearance
        if self.side == 'left':
            brush_color = Qt.green
        else:
            brush_color = Qt.blue
        
        self.setBrush(QBrush(brush_color))
        self.setPen(QPen(Qt.black, 2))
        
        # Set cursor to hand when hovering
        self.setCursor(Qt.PointingHandCursor)

        # 호버 효과 활성화
        self.setAcceptHoverEvents(True)

        # Add text label for point ID
        self.text_item = QGraphicsSimpleTextItem(self)
        font = QFont()
        font.setBold(True)
        self.text_item.setFont(font)
        if self.point_id:
            self.text_item.setText(self.point_id)
        self.text_item.setPos(radius, -radius * 2)
        self.text_item.setBrush(QBrush(Qt.black))

    def hoverEnterEvent(self, event):
        """
        마우스가 포인터 위에 올라왔을 때의 이벤트 처리
        """
        # 호버 상태에서 크기 증가
        curr_x, curr_y = self.pos().x(), self.pos().y()
        self.setRect(-self.HOVER_RADIUS, -self.HOVER_RADIUS, 
                     self.HOVER_RADIUS * 2, self.HOVER_RADIUS * 2)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """
        마우스가 포인터에서 벗어났을 때의 이벤트 처리
        """
        # 원래 크기로 복원
        curr_x, curr_y = self.pos().x(), self.pos().y()
        self.setRect(-self.radius, -self.radius, self.radius * 2, self.radius * 2)
        super().hoverLeaveEvent(event)

    def center(self):
        """
        Get the center point.
        
        Returns:
            QPointF: Center point
        """
        return QPointF(self.rect().center())
    
    def itemChange(self, change, value):
        """
        Handle item changes.
        
        Args:
            change: Type of change
            value: New value
            
        Returns:
            The adjusted value
        """
        if change == QGraphicsItem.ItemPositionChange and self.scene():
            # Update the text position
            self.text_item.setPos(self.radius, -self.radius * 2)
            
            # Notify the parent class
            self.scene().parent().point_moved.emit(self.side, self.index, value.x() + self.radius, value.y() + self.radius)
        
        return super().itemChange(change, value)
    
    def setPointId(self, point_id):
        """
        Set the point ID and update the text label.
        
        Args:
            point_id (str): Point ID (e.g., 'p00')
        """
        self.point_id = point_id
        if self.text_item:
            self.text_item.setText(point_id)


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
    
    def add_point_item(self, side: str, x: float, y: float, index: int, point_id: str = None):
        """
        Add a point to the specified scene.
        
        Args:
            side (str): 'left' or 'right'
            x (float): X-coordinate
            y (float): Y-coordinate
            index (int): Point index
            point_id (str, optional): Point ID (e.g., 'p00')
        """
        point = CalibrationPoint(x, y, index, side)
        
        if point_id:
            point.setPointId(point_id)
        
        if side == 'left':
            self.left_scene.addItem(point)
            self.left_points[index] = point
        else:
            self.right_scene.addItem(point)
            self.right_points[index] = point
        
        return point
    
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
        x, y = center
        rect = QRectF(x - radius, y - radius, radius * 2, radius * 2)
        
        # Create a semi-transparent rectangle
        color = QColor(Colors.ACCENT)
        color.setAlpha(64)  # 25% opacity
        
        # Remove existing overlay if any
        if side == 'left':
            if self.left_roi_overlay:
                self.left_scene.removeItem(self.left_roi_overlay)
            
            self.left_roi_overlay = QGraphicsRectItem(rect)
            self.left_roi_overlay.setPen(QPen(Colors.ACCENT, 1))
            self.left_roi_overlay.setBrush(QBrush(color))
            self.left_roi_overlay.setZValue(0.5)  # Between image and points
            self.left_scene.addItem(self.left_roi_overlay)
        elif side == 'right':
            if self.right_roi_overlay:
                self.right_scene.removeItem(self.right_roi_overlay)
            
            self.right_roi_overlay = QGraphicsRectItem(rect)
            self.right_roi_overlay.setPen(QPen(Colors.ACCENT, 1))
            self.right_roi_overlay.setBrush(QBrush(color))
            self.right_roi_overlay.setZValue(0.5)  # Between image and points
            self.right_scene.addItem(self.right_roi_overlay)
    
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
        # Clear existing grid lines for this side
        if side == 'left':
            for line in self.left_grid_lines:
                self.left_scene.removeItem(line)
            self.left_grid_lines.clear()
        else:
            for line in self.right_grid_lines:
                self.right_scene.removeItem(line)
            self.right_grid_lines.clear()
        
        # Create path for horizontal lines
        for row in range(rows):
            path = QPainterPath()
            for col in range(cols):
                idx = row * cols + col
                if idx >= len(points):
                    break
                
                x, y = points[idx]
                if col == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            
            # Create path item
            path_item = QGraphicsPathItem(path)
            path_item.setPen(QPen(Colors.WARNING, 1, Qt.DashLine))
            path_item.setZValue(0.75)  # Between ROI overlay and points
            
            # Add to scene and store reference
            if side == 'left':
                self.left_scene.addItem(path_item)
                self.left_grid_lines.append(path_item)
            else:
                self.right_scene.addItem(path_item)
                self.right_grid_lines.append(path_item)
        
        # Create path for vertical lines
        for col in range(cols):
            path = QPainterPath()
            for row in range(rows):
                idx = row * cols + col
                if idx >= len(points):
                    break
                
                x, y = points[idx]
                if row == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            
            # Create path item
            path_item = QGraphicsPathItem(path)
            path_item.setPen(QPen(Colors.WARNING, 1, Qt.DashLine))
            path_item.setZValue(0.75)  # Between ROI overlay and points
            
            # Add to scene and store reference
            if side == 'left':
                self.left_scene.addItem(path_item)
                self.left_grid_lines.append(path_item)
            else:
                self.right_scene.addItem(path_item)
                self.right_grid_lines.append(path_item)
    
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
        
        # Add images to scenes
        self.left_scene.addPixmap(left_image)
        self.right_scene.addPixmap(right_image)
        
        # Fit views to content
        self.left_view.fitInView(self.left_scene.sceneRect(), Qt.KeepAspectRatio)
        self.right_view.fitInView(self.right_scene.sceneRect(), Qt.KeepAspectRatio) 