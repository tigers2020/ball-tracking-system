#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calibration Tab module.
This module contains the CalibrationTab class which serves as the view for the Court Calibration tab.
"""

import logging
from typing import List, Tuple, Dict, Optional

from PySide6.QtCore import Qt, Signal, Slot, QPointF, QRectF
from PySide6.QtGui import QPen, QBrush, QColor, QPainter, QFont, QCursor, QPixmap, QImage
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGraphicsView, QGraphicsScene,
    QGraphicsEllipseItem, QGraphicsTextItem, QGraphicsPathItem, QPushButton,
    QGraphicsItem, QGroupBox, QLabel, QMessageBox, QGraphicsRectItem
)

from src.utils.ui_constants import Layout, Calibration, Messages
from src.utils.ui_theme import StyleManager

logger = logging.getLogger(__name__)


class CalibrationPointItem(QGraphicsEllipseItem):
    """
    Custom QGraphicsEllipseItem for calibration points that can be moved by the user.
    """
    
    def __init__(self, x: float, y: float, radius: float, index: int, is_original: bool = True):
        """
        Initialize the calibration point item.
        
        Args:
            x (float): X coordinate
            y (float): Y coordinate
            radius (float): Radius of the point
            index (int): Index of the point
            is_original (bool): Whether this is an original point or an adjusted point
        """
        # Create point with slightly larger radius for better visibility
        display_radius = radius * 1.5
        super().__init__(0, 0, display_radius * 2, display_radius * 2)
        
        self.index = index
        self.is_original = is_original
        self.radius = display_radius
        self.center_x = x
        self.center_y = y
        
        # Set position to center the ellipse on the desired point
        self.setPos(x - display_radius, y - display_radius)
        
        # Set flags to make it movable and selectable
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        
        # Set Z value (original points should be above adjusted points)
        self.setZValue(Calibration.Z_VALUE_ORIGINAL if is_original else Calibration.Z_VALUE_ADJUSTED)
        
        # Set color based on type - use more vivid colors for better visibility
        color = QColor(*Calibration.ORIGINAL_POINT_COLOR if is_original else Calibration.ADJUSTED_POINT_COLOR)
        # Use thicker pen for better visibility
        self.setPen(QPen(color, 2))
        # Use semi-transparent fill for better visibility
        color.setAlpha(180) 
        self.setBrush(QBrush(color))
        
        # Add label with better visibility
        self.label = QGraphicsTextItem(str(index + 1), self)
        font = QFont()
        font.setBold(True)
        font.setPointSize(10)
        self.label.setFont(font)
        # Add contrasting background for label
        self.label.setDefaultTextColor(Qt.white if is_original else Qt.black)
        self.update_label_position()
        
        # Add crop region visualization
        crop_size = display_radius * 5  # Make crop region 5x the point radius
        self.crop_rect = QGraphicsRectItem(-crop_size/2, -crop_size/2, crop_size, crop_size, self)
        self.crop_rect.setPen(QPen(Qt.blue, 1, Qt.DashLine))
        self.crop_rect.setPos(display_radius, display_radius)  # Center it on the point
        
        # Set cursor to hand to indicate it's draggable
        self.setCursor(QCursor(Qt.PointingHandCursor))
    
    def update_label_position(self):
        """Update the position of the label."""
        # Position the label to the top-right of the point for better visibility
        offset_x, offset_y = Calibration.POINT_LABEL_OFFSET
        self.label.setPos(self.radius + offset_x, -self.label.boundingRect().height())
        
        # Add a background rectangle for the label
        label_rect = self.label.boundingRect()
        if not hasattr(self, 'label_bg') or not self.label_bg:
            self.label_bg = QGraphicsRectItem(label_rect, self)
            self.label_bg.setBrush(QBrush(QColor(0, 0, 0, 120)))  # Semi-transparent background
            self.label_bg.setPen(QPen(Qt.NoPen))  # No border
            self.label_bg.setZValue(-1)  # Place behind text
            self.label.setZValue(1)  # Make sure text is on top
        else:
            self.label_bg.setRect(label_rect)
        
        # Position the background with the label
        self.label_bg.setPos(self.radius + offset_x, -self.label.boundingRect().height())
    
    def itemChange(self, change, value):
        """
        Handle item changes, including position changes.
        
        Args:
            change: Type of change
            value: New value
            
        Returns:
            New value
        """
        if change == QGraphicsItem.ItemPositionChange:
            # Update label position when point position changes
            self.update_label_position()
            
            # Update center position
            new_pos = value
            self.center_x = new_pos.x() + self.radius
            self.center_y = new_pos.y() + self.radius
        
        return super().itemChange(change, value)
    
    def get_position(self) -> Tuple[float, float]:
        """
        Get the center position of the point.
        
        Returns:
            Tuple[float, float]: (x, y) coordinates
        """
        # Return the stored center position for accuracy
        return (self.center_x, self.center_y)
    
    def paint(self, painter, option, widget):
        """
        Override the paint method to enhance visibility
        """
        # Turn off the selection outline
        option.state &= ~QGraphicsItem.ItemIsSelected
        
        # Call the parent paint method
        super().paint(painter, option, widget)
        
        # Highlight selected state with a glow effect if selected
        if self.isSelected():
            painter.setPen(QPen(QColor(255, 255, 0, 180), 2, Qt.SolidLine))
            painter.setBrush(QBrush(QColor(255, 255, 0, 40)))
            painter.drawEllipse(self.rect().adjusted(-3, -3, 3, 3))


class CalibrationGraphicsView(QGraphicsView):
    """
    Custom QGraphicsView for calibration that handles clicks for adding points.
    """
    
    # Signal emitted when a point is clicked on the scene
    point_clicked = Signal(int, tuple)
    
    # Signal emitted when a point should be added at the clicked position
    point_added = Signal(tuple)
    
    def __init__(self, parent=None):
        """
        Initialize the calibration graphics view.
        
        Args:
            parent (QWidget, optional): Parent widget
        """
        super().__init__(parent)
        
        # Create scene
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # Set properties
        self.setRenderHint(QPainter.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setInteractive(True)
        
        # Store points
        self.points = []
    
    def mousePressEvent(self, event):
        """
        Handle mouse press events.
        
        Args:
            event (QMouseEvent): Mouse event
        """
        if event.button() == Qt.LeftButton:
            # Get position in scene coordinates
            scene_pos = self.mapToScene(event.pos())
            
            # Check if clicked on an existing point
            item = self.scene.itemAt(scene_pos, self.transform())
            
            if isinstance(item, CalibrationPointItem):
                # Emit signal with point index and position
                self.point_clicked.emit(item.index, item.get_position())
            else:
                # Emit signal to add a new point
                self.point_added.emit((scene_pos.x(), scene_pos.y()))
        
        # Pass event to parent class
        super().mousePressEvent(event)


class CalibrationTab(QWidget):
    """
    Court Calibration tab for the Stereo Image Player application.
    """
    
    # Signal emitted when a point is added
    point_added = Signal(str, tuple)
    
    # Signal emitted when a point is updated (moved by the user)
    point_updated = Signal(str, int, tuple)
    
    # Signal emitted when points are cleared
    points_cleared = Signal(object)  # Use object type to allow None
    
    # Signal emitted when fine-tune button is clicked
    fine_tune_requested = Signal()
    
    # Signal emitted when save button is clicked
    save_requested = Signal()
    
    # Signal emitted when load button is clicked
    load_requested = Signal()
    
    # Signal emitted when current frame button is clicked
    load_current_frame_requested = Signal()
    
    def __init__(self, parent=None):
        """
        Initialize the calibration tab.
        
        Args:
            parent (QWidget, optional): Parent widget
        """
        super().__init__(parent)
        
        # 모델 저장 변수 추가
        self.model = None
        
        # Set up UI
        self._setup_ui()
        
        # Connect internal signals
        self._connect_signals()
    
    def _setup_ui(self):
        """Set up the user interface."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(Layout.MARGIN, Layout.MARGIN, Layout.MARGIN, Layout.MARGIN)
        main_layout.setSpacing(Layout.SPACING)
        
        # Add image views in a horizontal layout
        views_layout = QHBoxLayout()
        main_layout.addLayout(views_layout)
        
        # Left image view group
        left_group = QGroupBox("Left Camera")
        left_layout = QVBoxLayout(left_group)
        self.left_view = CalibrationGraphicsView()
        left_layout.addWidget(self.left_view)
        views_layout.addWidget(left_group)
        
        # Right image view group
        right_group = QGroupBox("Right Camera")
        right_layout = QVBoxLayout(right_group)
        self.right_view = CalibrationGraphicsView()
        right_layout.addWidget(self.right_view)
        views_layout.addWidget(right_group)
        
        # Control panel
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        main_layout.addWidget(control_panel)
        
        # Button layout
        button_layout = QHBoxLayout()
        button_layout.setSpacing(Layout.SPACING)
        
        # Clear buttons
        clear_group = QGroupBox("Clear Points")
        clear_layout = QVBoxLayout(clear_group)
        
        self.clear_left_btn = QPushButton("Clear Left")
        clear_layout.addWidget(self.clear_left_btn)
        
        self.clear_right_btn = QPushButton("Clear Right")
        clear_layout.addWidget(self.clear_right_btn)
        
        self.clear_all_btn = QPushButton("Clear All")
        clear_layout.addWidget(self.clear_all_btn)
        
        button_layout.addWidget(clear_group)
        
        # Action buttons
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout(action_group)
        
        self.save_btn = QPushButton("Save Calibration")
        self.save_btn.setStyleSheet("QPushButton[class='primary'] { background-color: #2A82DA; }")
        self.save_btn.setProperty("class", "primary")
        action_layout.addWidget(self.save_btn)
        
        self.load_btn = QPushButton("Load Calibration")
        action_layout.addWidget(self.load_btn)
        
        self.load_current_frame_btn = QPushButton("Load Current Frame")
        action_layout.addWidget(self.load_current_frame_btn)
        
        self.fine_tune_btn = QPushButton("Fine-Tune Points")
        action_layout.addWidget(self.fine_tune_btn)
        
        button_layout.addWidget(action_group)
        
        # Add button layout to control layout
        control_layout.addLayout(button_layout)
        
        # Apply styles
        for btn in [self.clear_left_btn, self.clear_right_btn, self.clear_all_btn,
                   self.save_btn, self.load_btn, self.load_current_frame_btn, self.fine_tune_btn]:
            btn.setMinimumHeight(Layout.BUTTON_HEIGHT)
            
        # Set up scenes for graphics views
        self.left_scene = QGraphicsScene()
        self.left_view.setScene(self.left_scene)
        
        self.right_scene = QGraphicsScene()
        self.right_view.setScene(self.right_scene)
        
        # Setup graphics view settings
        for view in [self.left_view, self.right_view]:
            view.setRenderHint(QPainter.Antialiasing)
            view.setRenderHint(QPainter.SmoothPixmapTransform)
            view.setRenderHint(QPainter.TextAntialiasing)
            view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
            view.setDragMode(QGraphicsView.ScrollHandDrag)
            view.setOptimizationFlags(QGraphicsView.DontAdjustForAntialiasing |
                                    QGraphicsView.DontSavePainterState)
            view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
            view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
    
    def _connect_signals(self):
        """Connect signals to slots."""
        # Left view signals
        self.left_view.point_added.connect(self._on_left_point_added)
        self.left_view.point_clicked.connect(self._on_left_point_clicked)
        
        # Right view signals
        self.right_view.point_added.connect(self._on_right_point_added)
        self.right_view.point_clicked.connect(self._on_right_point_clicked)
        
        # Button signals
        self.clear_left_btn.clicked.connect(self._on_clear_left)
        self.clear_right_btn.clicked.connect(self._on_clear_right)
        self.clear_all_btn.clicked.connect(self._on_clear_all)
        self.save_btn.clicked.connect(self.save_requested)
        self.load_btn.clicked.connect(self.load_requested)
        self.load_current_frame_btn.clicked.connect(self.load_current_frame_requested)
        self.fine_tune_btn.clicked.connect(self.fine_tune_requested)
    
    def _on_left_point_added(self, position):
        """
        Handle adding a point on the left view.
        
        Args:
            position (tuple): (x, y) coordinates
        """
        self.point_added.emit("left", position)
    
    def _on_right_point_added(self, position):
        """
        Handle adding a point on the right view.
        
        Args:
            position (tuple): (x, y) coordinates
        """
        self.point_added.emit("right", position)
    
    def _on_left_point_clicked(self, index, position):
        """
        Handle clicking on a point in the left view.
        
        Args:
            index (int): Point index
            position (tuple): (x, y) coordinates
        """
        logger.debug(f"Left point {index} clicked at {position}")
    
    def _on_right_point_clicked(self, index, position):
        """
        Handle clicking on a point in the right view.
        
        Args:
            index (int): Point index
            position (tuple): (x, y) coordinates
        """
        logger.debug(f"Right point {index} clicked at {position}")
    
    def _on_clear_left(self):
        """Handle clearing points on the left view."""
        self.points_cleared.emit("left")
    
    def _on_clear_right(self):
        """Handle clearing points on the right view."""
        self.points_cleared.emit("right")
    
    def _on_clear_all(self):
        """Handle clearing all points."""
        self.points_cleared.emit(None)  # None means both sides
    
    def update_points(self, side: str, points: List[Tuple[float, float]]):
        """
        Update the displayed points for the specified side.
        
        Args:
            side (str): 'left' or 'right' side
            points (List[Tuple[float, float]]): List of (x, y) coordinates
        """
        # Get the appropriate view
        view = self.left_view if side == "left" else self.right_view
        
        # 화면 해상도에 따라 포인트 크기 조절
        # 기본 반지름(5)는 1080p 기준, 현재 해상도에 맞게 반지름 계산
        scene_rect = view.scene.sceneRect()
        scene_width = scene_rect.width()
        
        # 씬이 아직 없는 경우 기본 반지름 사용
        if scene_width == 0:
            point_radius = Calibration.POINT_RADIUS
        else:
            # 1920(1080p 기준 너비)에 대한 현재 씬 너비의 비율로 크기 조절
            reference_width = 1920  # 1080p 기준 해상도 너비
            scale_factor = scene_width / reference_width
            point_radius = max(3, int(Calibration.POINT_RADIUS * scale_factor))  # 최소 3픽셀 보장
        
        logger.debug(f"Adjusting point radius for {side} view: {point_radius} (scene width: {scene_width})")
        
        # Clear existing points from the scene
        for item in view.scene.items():
            if isinstance(item, CalibrationPointItem):
                view.scene.removeItem(item)
        
        # 포인트 리스트 초기화
        view.points = []
        
        # Add new points
        for i, (x, y) in enumerate(points):
            point_item = CalibrationPointItem(x, y, point_radius, i)
            view.scene.addItem(point_item)
            
            # Connect point movement
            point_item.setPos(x - point_radius, y - point_radius)
            
            # Store reference to the point
            view.points.append(point_item)
        
        # Add connecting lines if there are any points
        if len(points) > 0:
            self._draw_connecting_lines(side, points)
    
    def _draw_connecting_lines(self, side: str, points: List[Tuple[float, float]]):
        """
        Draw connecting lines between points.
        
        Args:
            side (str): 'left' or 'right' side
            points (List[Tuple[float, float]]): List of (x, y) coordinates
        """
        # Implementation will be added in Week 2
        pass
    
    def point_moved(self, side: str, index: int, new_position: Tuple[float, float]):
        """
        Handle a point being moved by the user.
        
        Args:
            side (str): 'left' or 'right' side
            index (int): Index of the point
            new_position (Tuple[float, float]): New (x, y) coordinates
        """
        # Emit signal for controller to update model
        self.point_updated.emit(side, index, new_position)
    
    def show_error(self, message: str):
        """
        Show an error message.
        
        Args:
            message (str): Error message
        """
        QMessageBox.critical(self, "Error", message)
    
    def show_warning(self, message: str):
        """
        Show a warning message.
        
        Args:
            message (str): Warning message
        """
        QMessageBox.warning(self, "Warning", message)
    
    def show_info(self, message: str):
        """
        Show an information message.
        
        Args:
            message (str): Information message
        """
        QMessageBox.information(self, "Information", message)
    
    def set_images(self, left_image_path: str, right_image_path: str):
        """
        Set the images for calibration.
        Load the images and display them in the graphics views.
        
        Args:
            left_image_path (str): Path to the left image
            right_image_path (str): Path to the right image
        """
        import cv2
        from PySide6.QtGui import QPixmap, QImage
        import numpy as np
        import os
        
        # Clear existing scenes
        self.left_scene.clear()
        self.right_scene.clear()
        
        # Log the image paths
        logger.info(f"Setting calibration images: left={left_image_path}, right={right_image_path}")
        
        # Check if files exist
        if not os.path.exists(left_image_path) or not os.path.exists(right_image_path):
            logger.error(f"One or both image files do not exist: {left_image_path}, {right_image_path}")
            self.show_error(f"Image files not found: {left_image_path}, {right_image_path}")
            return
            
        try:
            # Load images using OpenCV
            left_cv_image = cv2.imread(left_image_path)
            right_cv_image = cv2.imread(right_image_path)
            
            if left_cv_image is None or right_cv_image is None:
                logger.error("Failed to load image using OpenCV")
                self.show_error("Failed to load one or both images")
                return
                
            # Convert from BGR to RGB
            left_cv_image = cv2.cvtColor(left_cv_image, cv2.COLOR_BGR2RGB)
            right_cv_image = cv2.cvtColor(right_cv_image, cv2.COLOR_BGR2RGB)
            
            # Convert to QImage
            left_h, left_w, left_ch = left_cv_image.shape
            left_qimage = QImage(left_cv_image.data, left_w, left_h, left_w * left_ch, QImage.Format_RGB888)
            
            right_h, right_w, right_ch = right_cv_image.shape
            right_qimage = QImage(right_cv_image.data, right_w, right_h, right_w * right_ch, QImage.Format_RGB888)
            
            # Convert to QPixmap and add to scenes
            left_pixmap = QPixmap.fromImage(left_qimage)
            right_pixmap = QPixmap.fromImage(right_qimage)
            
            # Add to scenes
            self.left_scene.addPixmap(left_pixmap)
            self.right_scene.addPixmap(right_pixmap)
            
            # Fit views to content
            self.left_view.fitInView(self.left_scene.sceneRect(), Qt.KeepAspectRatio)
            self.right_view.fitInView(self.right_scene.sceneRect(), Qt.KeepAspectRatio)
            
            # Redraw any existing points
            if hasattr(self, 'model') and self.model:
                self.update_points("left", self.model.left_pts)
                self.update_points("right", self.model.right_pts)
            
            logger.info("Calibration images set successfully")
            
        except Exception as e:
            logger.error(f"Error setting calibration images: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.show_error(f"Error setting images: {str(e)}") 