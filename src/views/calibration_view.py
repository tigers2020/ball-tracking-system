#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calibration View module.
This module provides a UI component for court calibration.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import math
import numpy as np

from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap, QBrush, QCursor, QPainterPath, QColor
from PySide6.QtWidgets import (QGraphicsPathItem, QGraphicsRectItem, QGraphicsScene,
                           QGraphicsView, QHBoxLayout, QLabel, QPushButton,
                           QSplitter, QVBoxLayout, QWidget, QGraphicsEllipseItem, QGraphicsSimpleTextItem)

from src.utils.ui_constants import Layout, WindowSize
from src.utils.ui_theme import Colors
from src.utils.geometry import (pixel_to_scene, scene_to_pixel, 
                              get_scale_factors, pixel_to_normalized, 
                              normalized_to_pixel)

logger = logging.getLogger(__name__)


class CalibrationPoint(QGraphicsEllipseItem):
    """
    Custom QGraphicsEllipseItem for calibration points.
    Supports moving and tracks its index.
    """
    
    def __init__(self, index: int, radius: float = 12.0, parent_view=None, side=None):
        """
        Initialize a calibration point.
        
        Args:
            index (int): Point index
            radius (float): Point radius (default: 12.0)
            parent_view: Reference to the parent CalibrationView
            side: 'left' or 'right' indicating which side the point belongs to
        """
        super().__init__(0, 0, radius * 2, radius * 2)
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsSelectable, True)
        
        # 포인트 색상을 빨간색으로 변경
        self.setBrush(QBrush(QColor(255, 0, 0, 180)))  # 반투명 빨간색
        self.setPen(QPen(QColor(200, 0, 0), 2))  # 진한 빨간색 테두리
        
        self.setZValue(2.0)  # Make points appear above other items
        self.index = index
        self.radius = radius
        self.is_moving = False
        self.parent_view = parent_view
        self.side = side
        
        # Set cursor to pointing hand
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        
        # 텍스트 라벨 추가
        self.label = QGraphicsSimpleTextItem(str(index), self)
        self.label.setBrush(QBrush(Qt.white))
        
        # 중앙에 라벨 배치
        label_width = self.label.boundingRect().width()
        label_height = self.label.boundingRect().height()
        self.label.setPos(radius - label_width / 2, radius - label_height / 2)
        
    def itemChange(self, change, value):
        """
        Handle position change events to keep track of the point's position.
        
        Args:
            change: The type of change
            value: The new value
            
        Returns:
            The processed value
        """
        # Handle position change event - 드래그 중에 호출됨
        if change == QGraphicsEllipseItem.GraphicsItemChange.ItemPositionChange and self.scene():
            if self.parent_view and self.side:
                # Get center coordinates of the point (add radius to top-left position)
                new_pos = value  # 새 위치 (top-left 좌표)
                x = new_pos.x() + self.radius  # center x
                y = new_pos.y() + self.radius  # center y
                
                logger.debug(f"Point {self.index} moving on {self.side} side to scene coordinates: ({x:.2f}, {y:.2f})")
                
                # Set moving state
                if not self.is_moving:
                    self.is_moving = True
                    self.setBrush(QBrush(QColor(255, 165, 0, 200)))  # 드래그 중에는 주황색으로 변경
                    self.setPen(QPen(QColor(255, 140, 0), 3))
        
        # Handle position changed event (after move is complete or during drag)
        elif change == QGraphicsEllipseItem.GraphicsItemChange.ItemPositionHasChanged and self.scene():
            if self.parent_view and self.side:
                # 현재 위치 계산 (top-left에서 center로 변환)
                current_pos = self.pos()
                center_x = current_pos.x() + self.radius
                center_y = current_pos.y() + self.radius
                
                # Emit the point moved signal with center coordinates
                self.parent_view.point_moved.emit(self.side, self.index, center_x, center_y)
                
                logger.debug(f"Point {self.index} moved on {self.side} side to scene coords: ({center_x:.2f}, {center_y:.2f})")
            
            # Reset moving state if this is end of movement
            if self.is_moving and not self.scene().mouseGrabberItem() == self:
                self.is_moving = False
                self.setBrush(QBrush(QColor(255, 0, 0, 180)))  # 빨간색으로 복귀
                self.setPen(QPen(QColor(200, 0, 0), 2))
        
        # Handle selection change
        elif change == QGraphicsEllipseItem.GraphicsItemChange.ItemSelectedChange:
            # Change appearance when selected
            if value:
                self.setBrush(QBrush(QColor(0, 255, 255, 200)))  # 선택시 청록색
                self.setPen(QPen(QColor(0, 200, 200), 3))
            else:
                self.setBrush(QBrush(QColor(255, 0, 0, 180)))  # 빨간색으로 복귀
                self.setPen(QPen(QColor(200, 0, 0), 2))
            
        return super().itemChange(change, value)
        
    def mousePressEvent(self, event):
        """Handle mouse press events to provide visual feedback."""
        self.setBrush(QBrush(QColor(255, 165, 0, 200)))  # 주황색으로 변경
        self.setPen(QPen(QColor(255, 140, 0), 3))
        super().mousePressEvent(event)
        
    def mouseReleaseEvent(self, event):
        """Handle mouse release events to provide visual feedback."""
        self.setBrush(QBrush(QColor(255, 0, 0, 180)))  # 빨간색으로 복귀
        self.setPen(QPen(QColor(200, 0, 0), 2))
        super().mouseReleaseEvent(event)


class CalibrationView(QWidget):
    """
    Widget for court calibration with left and right image views.
    """
    
    # Signal emitted when a point is added (side, x, y)
    point_added = Signal(str, float, float)
    
    # Signal emitted when a point is moved (side, index, x, y)
    point_moved = Signal(str, int, float, float)
    
    def __init__(self, parent=None):
        """
        Initialize the calibration view.
        
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
        self.left_roi_overlay: Optional[QGraphicsRectItem] = None
        self.right_roi_overlay: Optional[QGraphicsRectItem] = None
        
        # Initialize grid lines
        self.left_grid_lines: List[QGraphicsPathItem] = []
        self.right_grid_lines: List[QGraphicsPathItem] = []
        
        # Initialize image pixmaps
        self.left_pixmap: Optional[QPixmap] = None
        self.right_pixmap: Optional[QPixmap] = None
        
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
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Create left view container
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        left_label = QLabel("Left Image")
        left_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(left_label)
        
        self.left_scene = QGraphicsScene()
        self.left_view = QGraphicsView(self.left_scene)
        self.left_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.left_view.setDragMode(QGraphicsView.DragMode.NoDrag)
        left_layout.addWidget(self.left_view)
        
        # Create right view container
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        right_label = QLabel("Right Image")
        right_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(right_label)
        
        self.right_scene = QGraphicsScene()
        self.right_view = QGraphicsView(self.right_scene)
        self.right_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.right_view.setDragMode(QGraphicsView.DragMode.NoDrag)
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
        Handle mouse press events for calibration point placement.
        
        Args:
            event: Mouse event object
            view: The graphics view that received the event
            side: 'left' or 'right' to indicate which view
        """
        # Check if we clicked on an existing item
        scene_pos = view.mapToScene(event.pos())
        items_at_pos = view.scene().items(scene_pos)
        
        # If we clicked on a CalibrationPoint, don't add a new point
        for item in items_at_pos:
            if isinstance(item, CalibrationPoint):
                # Let the original handler handle selection/dragging
                super(view.__class__, view).mousePressEvent(event)
                return
        
        # If we reached here, we didn't click on a point, so try to add a new one
        scene_x, scene_y = scene_pos.x(), scene_pos.y()
        
        # Emit the signal with scene coordinates
        self.point_added.emit(side, scene_x, scene_y)
        
        # Call the original handler to maintain other functionality
        super(view.__class__, view).mousePressEvent(event)
    
    def add_point(self, side: str, index: int, scene_x: float, scene_y: float):
        """
        Add a calibration point to the specified view.
        
        Args:
            side (str): 'left' or 'right'
            index (int): Point index
            scene_x (float): X-coordinate in scene coordinates
            scene_y (float): Y-coordinate in scene coordinates
        """
        if side not in ['left', 'right']:
            logger.error(f"Invalid side specified: {side}")
            return
        
        # Create a calibration point item (with +/- offset to center point on coordinates)
        point = CalibrationPoint(index, radius=7, parent_view=self, side=side)
        point.setPos(scene_x - point.radius, scene_y - point.radius)
        
        # Add to scene
        if side == 'left':
            self.left_scene.addItem(point)
            self.left_points[index] = point
        else:
            self.right_scene.addItem(point)
            self.right_points[index] = point
        
        logger.debug(f"Added {side} calibration point {index} at ({scene_x:.1f}, {scene_y:.1f})")
    
    def add_point_item(self, side: str, x: float, y: float, index: int):
        """
        Add a calibration point item to the specified view.
        This method is used for backward compatibility with CalibrationTab.
        
        Args:
            side (str): 'left' or 'right'
            x (float): X-coordinate in scene coordinates
            y (float): Y-coordinate in scene coordinates
            index (int): Point index
        """
        self.add_point(side, index, x, y)
    
    def update_point(self, side: str, index: int, scene_x: float, scene_y: float):
        """
        Update an existing calibration point's position.
        
        Args:
            side (str): 'left' or 'right'
            index (int): Point index
            scene_x (float): New X-coordinate in scene coordinates
            scene_y (float): New Y-coordinate in scene coordinates
        """
        if side == 'left' and index in self.left_points:
            point = self.left_points[index]
            point.setPos(scene_x - point.radius, scene_y - point.radius)
        elif side == 'right' and index in self.right_points:
            point = self.right_points[index]
            point.setPos(scene_x - point.radius, scene_y - point.radius)
        else:
            logger.error(f"Invalid side or index not found: {side}, {index}")
    
    def update_point_item(self, side: str, index: int, x: float, y: float):
        """
        Update an existing calibration point item's position.
        This method is used for backward compatibility with CalibrationTab.
        
        Args:
            side (str): 'left' or 'right'
            index (int): Point index
            x (float): New X-coordinate in scene coordinates
            y (float): New Y-coordinate in scene coordinates
        """
        self.update_point(side, index, x, y)
    
    def _on_point_moved(self, side: str, index: int, scene_x: float, scene_y: float):
        """
        Handle point movement events from CalibrationPoint items.
        
        Args:
            side (str): 'left' or 'right'
            index (int): Point index
            scene_x (float): New X-coordinate in scene coordinates
            scene_y (float): New Y-coordinate in scene coordinates
        """
        # Emit the signal with scene coordinates
        self.point_moved.emit(side, index, scene_x, scene_y)
    
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
    
    def add_grid_lines(self, side: str, lines):
        """
        Add grid lines to the specified view.
        
        Args:
            side (str): 'left' or 'right'
            lines: List of ((x1, y1), (x2, y2)) tuples in pixel coordinates
        """
        if side not in ['left', 'right']:
            logger.error(f"Invalid side specified: {side}")
            return
        
        # Choose the appropriate scene and storage list
        if side == 'left':
            scene = self.left_scene
            grid_lines = self.left_grid_lines
            width_scale, height_scale = self._get_scale_factors('left')
        else:
            scene = self.right_scene
            grid_lines = self.right_grid_lines
            width_scale, height_scale = self._get_scale_factors('right')
        
        # Create QPainterPath for each line
        for (x1, y1), (x2, y2) in lines:
            # Convert pixel coordinates to scene coordinates
            scene_x1, scene_y1 = pixel_to_scene((x1, y1), width_scale, height_scale)
            scene_x2, scene_y2 = pixel_to_scene((x2, y2), width_scale, height_scale)
            
            # Create path item
            path = QPainterPath()
            path.moveTo(scene_x1, scene_y1)
            path.lineTo(scene_x2, scene_y2)
            
            line_item = scene.addPath(path, QPen(Colors.ACCENT, 1.5))
            grid_lines.append(line_item)
    
    def draw_grid_lines(self, side: str, points: List[Tuple[float, float]], rows: int, cols: int):
        """
        Draw grid lines connecting calibration points in the specified pattern.
        
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
            scene = self.left_scene
            grid_lines = self.left_grid_lines
        else:
            for line in self.right_grid_lines:
                self.right_scene.removeItem(line)
            self.right_grid_lines.clear()
            scene = self.right_scene
            grid_lines = self.right_grid_lines
        
        # 포인트가 14개 미만이면 그리드 라인을 그리지 않음
        if len(points) < 14:
            logger.warning(f"Not enough points to draw grid lines: {len(points)}/14")
            return
        
        # 현재 지정된 패턴대로 그리드 라인 그리기
        # 가로 라인
        horizontal_groups = [
            [0, 1, 2, 3],     # 1-2-3-4
            [4, 5, 6],        # 5-6-7
            [7, 8, 9],        # 8-9-10
            [10, 11, 12, 13]  # 11-12-13-14
        ]
        
        # 세로 라인
        vertical_groups = [
            [0, 10],          # 1-11
            [1, 4, 7, 10],    # 2-5-8-11
            [5, 8],           # 6-9
            [2, 6, 9, 12],    # 3-7-10-13
            [3, 13]           # 4-14
        ]
        
        # 펜 설정
        pen = QPen(QColor(0, 180, 255, 200), 2.0)  # 파란색 라인
        pen.setStyle(Qt.PenStyle.SolidLine)
        
        # 가로 라인 그리기
        for group in horizontal_groups:
            if all(idx < len(points) for idx in group):
                path = QPainterPath()
                first = True
                for idx in group:
                    x, y = points[idx]
                    if first:
                        path.moveTo(x, y)
                        first = False
                    else:
                        path.lineTo(x, y)
                
                line_item = scene.addPath(path, pen)
                line_item.setZValue(1.0)  # 포인트 아래, ROI 위
                grid_lines.append(line_item)
        
        # 세로 라인 그리기
        for group in vertical_groups:
            if all(idx < len(points) for idx in group):
                path = QPainterPath()
                first = True
                for idx in group:
                    x, y = points[idx]
                    if first:
                        path.moveTo(x, y)
                        first = False
                    else:
                        path.lineTo(x, y)
                
                line_item = scene.addPath(path, pen)
                line_item.setZValue(1.0)
                grid_lines.append(line_item)
        
        logger.debug(f"Drew custom grid lines for {side} side")
    
    def update_roi_overlay(self, side: str, roi):
        """
        Update ROI overlay on the specified view.
        
        Args:
            side (str): 'left' or 'right'
            roi: ROI data (center_x, center_y, radius)
        """
        if side not in ['left', 'right']:
            logger.error(f"Invalid side specified: {side}")
            return
        
        # Clear existing ROI overlay
        self.hide_roi(side)
        
        # Create a new overlay if ROI data is provided
        if roi:
            center_x, center_y, radius = roi
            
            scene = self.left_scene if side == 'left' else self.right_scene
            
            # Create a semi-transparent yellow rect for the ROI
            rect_x = center_x - radius
            rect_y = center_y - radius
            rect_width = radius * 2
            rect_height = radius * 2
            
            rect_item = QGraphicsRectItem(rect_x, rect_y, rect_width, rect_height)
            
            # Semi-transparent yellow fill
            rect_item.setBrush(QBrush(QColor(255, 255, 0, 60)))  # 반투명 노란색
            
            # Yellow border
            rect_item.setPen(QPen(QColor(255, 255, 0, 180), 1.5))  # 반투명 노란색 테두리
            
            rect_item.setZValue(0.5)  # Between image and points
            
            # Add to scene and store reference
            scene.addItem(rect_item)
            if side == 'left':
                self.left_roi_overlay = rect_item
            else:
                self.right_roi_overlay = rect_item
    
    def show_roi(self, side: str, center: Tuple[float, float], radius: float):
        """
        Show a ROI (Region of Interest) circle around a point.
        
        Args:
            side (str): 'left' or 'right'
            center (Tuple[float, float]): (x, y) center of the ROI in scene coordinates
            radius (float): Radius of the ROI in scene units
        """
        if side not in ['left', 'right']:
            logger.error(f"Invalid side specified: {side}")
            return
        
        # Clear any existing ROI overlay
        self.hide_roi(side)
        
        # Create a rectangle for the ROI
        roi_x = center[0] - radius
        roi_y = center[1] - radius
        roi_width = radius * 2
        roi_height = radius * 2
        
        # Create a rectangle item
        if side == 'left':
            rect_item = QGraphicsRectItem(roi_x, roi_y, roi_width, roi_height)
            rect_item.setBrush(QBrush(QColor(255, 255, 0, 60)))  # 반투명 노란색
            rect_item.setPen(QPen(QColor(255, 255, 0, 180), 1.5))  # 노란색 테두리
            rect_item.setZValue(0.5)  # 이미지와 포인트 사이에 표시
            self.left_scene.addItem(rect_item)
            self.left_roi_overlay = rect_item
        else:
            rect_item = QGraphicsRectItem(roi_x, roi_y, roi_width, roi_height)
            rect_item.setBrush(QBrush(QColor(255, 255, 0, 60)))  # 반투명 노란색
            rect_item.setPen(QPen(QColor(255, 255, 0, 180), 1.5))  # 노란색 테두리
            rect_item.setZValue(0.5)
            self.right_scene.addItem(rect_item)
            self.right_roi_overlay = rect_item
    
    def hide_roi(self, side: str):
        """
        Hide the ROI overlay for the specified view.
        
        Args:
            side (str): 'left' or 'right'
        """
        if side == 'left' and self.left_roi_overlay:
            self.left_scene.removeItem(self.left_roi_overlay)
            self.left_roi_overlay = None
        elif side == 'right' and self.right_roi_overlay:
            self.right_scene.removeItem(self.right_roi_overlay)
            self.right_roi_overlay = None
    
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
        self.left_view.fitInView(self.left_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.right_view.fitInView(self.right_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
    
    def _get_scale_factors(self, side: str) -> Tuple[float, float]:
        """
        Get the scale factors for the specified view.
        
        Args:
            side (str): 'left' or 'right'
            
        Returns:
            Tuple[float, float]: (width_scale, height_scale)
        """
        if side == 'left':
            pixmap_width = self.left_image_width
            pixmap_height = self.left_image_height
            scene_width = self.left_scene.width()
            scene_height = self.left_scene.height()
        else:
            pixmap_width = self.right_image_width
            pixmap_height = self.right_image_height
            scene_width = self.right_scene.width()
            scene_height = self.right_scene.height()
        
        # Use utility function to calculate scale factors
        return get_scale_factors(scene_width, scene_height, pixmap_width, pixmap_height)
    
    def scene_to_pixel(self, side: str, scene_x: float, scene_y: float) -> Tuple[float, float]:
        """
        Convert scene coordinates to pixel coordinates.
        
        Args:
            side (str): 'left' or 'right'
            scene_x (float): X-coordinate in scene coordinates
            scene_y (float): Y-coordinate in scene coordinates
            
        Returns:
            Tuple[float, float]: Pixel coordinates
        """
        width_scale, height_scale = self._get_scale_factors(side)
        return scene_to_pixel((scene_x, scene_y), width_scale, height_scale)
    
    def pixel_to_scene(self, side: str, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """
        Convert pixel coordinates to scene coordinates.
        
        Args:
            side (str): 'left' or 'right'
            pixel_x (float): X-coordinate in pixel coordinates
            pixel_y (float): Y-coordinate in pixel coordinates
            
        Returns:
            Tuple[float, float]: Scene coordinates
        """
        width_scale, height_scale = self._get_scale_factors(side)
        return pixel_to_scene((pixel_x, pixel_y), width_scale, height_scale)
    
    def normalized_to_scene(self, side: str, norm_x: float, norm_y: float) -> Tuple[float, float]:
        """
        Convert normalized coordinates (0-1) to scene coordinates.
        
        Args:
            side (str): 'left' or 'right'
            norm_x (float): X-coordinate in normalized space (0-1)
            norm_y (float): Y-coordinate in normalized space (0-1)
            
        Returns:
            Tuple[float, float]: Scene coordinates
        """
        if side == 'left':
            image_width = self.left_image_width
            image_height = self.left_image_height
        else:
            image_width = self.right_image_width
            image_height = self.right_image_height
        
        # First convert normalized to pixel
        pixel_x, pixel_y = normalized_to_pixel((norm_x, norm_y), image_width, image_height)
        
        # Then convert pixel to scene
        return self.pixel_to_scene(side, pixel_x, pixel_y)
    
    def scene_to_normalized(self, side: str, scene_x: float, scene_y: float) -> Tuple[float, float]:
        """
        Convert scene coordinates to normalized coordinates (0-1).
        
        Args:
            side (str): 'left' or 'right'
            scene_x (float): X-coordinate in scene coordinates
            scene_y (float): Y-coordinate in scene coordinates
            
        Returns:
            Tuple[float, float]: Normalized coordinates (0-1)
        """
        # First convert scene to pixel
        pixel_x, pixel_y = self.scene_to_pixel(side, scene_x, scene_y)
        
        # Then convert pixel to normalized
        if side == 'left':
            image_width = self.left_image_width
            image_height = self.left_image_height
        else:
            image_width = self.right_image_width
            image_height = self.right_image_height
        
        return pixel_to_normalized((pixel_x, pixel_y), image_width, image_height)
    
    def resizeEvent(self, event):
        """Handle resize events to maintain aspect ratio."""
        super().resizeEvent(event)
        
        # Maintain aspect ratio when resizing
        if self.left_pixmap:
            self.left_view.fitInView(self.left_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        if self.right_pixmap:
            self.right_view.fitInView(self.right_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
    
    def update_grid_lines(self, side: str = None):
        """
        Update grid lines based on the current points.
        
        Args:
            side (str, optional): 'left', 'right', or None for both sides
        """
        # Signal that grid lines need to be updated
        # The controller will handle the actual grid line creation logic
        # This method is provided as a convenience for the controller
        logger.debug(f"Request to update grid lines for {side if side else 'both sides'}")
        
        # If no specific side is requested, update both
        if side is None:
            # Clear both sides' grid lines
            for line in self.left_grid_lines:
                self.left_scene.removeItem(line)
            self.left_grid_lines.clear()
            
            for line in self.right_grid_lines:
                self.right_scene.removeItem(line)
            self.right_grid_lines.clear()
        elif side == 'left':
            # Clear only left side grid lines
            for line in self.left_grid_lines:
                self.left_scene.removeItem(line)
            self.left_grid_lines.clear()
        elif side == 'right':
            # Clear only right side grid lines
            for line in self.right_grid_lines:
                self.right_scene.removeItem(line)
            self.right_grid_lines.clear()
        else:
            logger.error(f"Invalid side specified for grid line update: {side}")
    
    def get_left_image(self):
        """
        Get the left image as a numpy array.
        
        Returns:
            numpy.ndarray: The left image, or None if no image is available
        """
        if self.left_pixmap is None:
            return None
            
        # Convert QPixmap to QImage
        image = self.left_pixmap.toImage()
        
        # Convert QImage to numpy array
        width = image.width()
        height = image.height()
        
        # Get the image format
        format = image.format()
        
        # Create numpy array from the image data
        if format == QImage.Format_RGB32 or format == QImage.Format_ARGB32:
            # For 32-bit formats (4 bytes per pixel)
            buffer = image.constBits()
            # Use buffer protocol to create numpy array directly
            arr = np.array(buffer, dtype=np.uint8).reshape(height, width, 4)
            # Convert RGBA/ARGB to RGB
            rgb_arr = arr[:, :, :3]
        else:
            # Convert to RGB32 format first
            image = image.convertToFormat(QImage.Format_RGB32)
            buffer = image.constBits()
            arr = np.array(buffer, dtype=np.uint8).reshape(height, width, 4)
            rgb_arr = arr[:, :, :3]
        
        # Convert RGB to BGR (for OpenCV compatibility)
        bgr_arr = rgb_arr[:, :, ::-1].copy()
        
        return bgr_arr
    
    def get_right_image(self):
        """
        Get the right image as a numpy array.
        
        Returns:
            numpy.ndarray: The right image, or None if no image is available
        """
        if self.right_pixmap is None:
            return None
            
        # Convert QPixmap to QImage
        image = self.right_pixmap.toImage()
        
        # Convert QImage to numpy array
        width = image.width()
        height = image.height()
        
        # Get the image format
        format = image.format()
        
        # Create numpy array from the image data
        if format == QImage.Format_RGB32 or format == QImage.Format_ARGB32:
            # For 32-bit formats (4 bytes per pixel)
            buffer = image.constBits()
            # Use buffer protocol to create numpy array directly
            arr = np.array(buffer, dtype=np.uint8).reshape(height, width, 4)
            # Convert RGBA/ARGB to RGB
            rgb_arr = arr[:, :, :3]
        else:
            # Convert to RGB32 format first
            image = image.convertToFormat(QImage.Format_RGB32)
            buffer = image.constBits()
            arr = np.array(buffer, dtype=np.uint8).reshape(height, width, 4)
            rgb_arr = arr[:, :, :3]
        
        # Convert RGB to BGR (for OpenCV compatibility)
        bgr_arr = rgb_arr[:, :, ::-1].copy()
        
        return bgr_arr 