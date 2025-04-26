#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bounce Overlay module.
This module contains the BounceOverlayWidget class for displaying the tennis court and ball bounces in a 2D top-down view.
"""

import logging
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene, 
                              QGraphicsItem, QGraphicsEllipseItem, QGraphicsPathItem)
from PySide6.QtCore import Qt, QPointF, QRectF, Signal, Slot, QTimer
from PySide6.QtGui import QPen, QBrush, QColor, QPainterPath, QTransform, QPainter

from src.services.geometry.court_frame import CourtFrame


class CourtGraphicsItem(QGraphicsItem):
    """Tennis court graphics item for the bounce overlay view."""
    
    # Tennis court dimensions in meters (standard)
    COURT_WIDTH = 10.97  # Width of singles court (meters)
    COURT_LENGTH = 23.77  # Length of court (meters)
    SERVICE_LINE_DIST = 6.40  # Distance from baseline to service line (meters)
    CENTER_MARK_LENGTH = 0.10  # Length of center mark on baseline (meters)
    NET_HEIGHT = 0.914  # Height of net at center (meters)
    
    # Colors
    COURT_COLOR = QColor(42, 128, 42)  # Court green
    LINE_COLOR = QColor(255, 255, 255)  # Lines white
    NET_COLOR = QColor(50, 50, 50)  # Net dark gray
    
    def __init__(self, parent=None):
        """Initialize the court graphics item."""
        super(CourtGraphicsItem, self).__init__(parent)
        
        # Court scale (pixels per meter)
        self.scale_factor = 20  # Default scale
        
        # Court boundaries
        self.court_rect = QRectF(-self.COURT_WIDTH/2 * self.scale_factor, 
                                 -self.COURT_LENGTH/2 * self.scale_factor,
                                 self.COURT_WIDTH * self.scale_factor, 
                                 self.COURT_LENGTH * self.scale_factor)
        
        # Set up pens
        self.line_pen = QPen(self.LINE_COLOR, 2)
        self.net_pen = QPen(self.NET_COLOR, 3)
        
        # Set up brushes
        self.court_brush = QBrush(self.COURT_COLOR)
        
        # Flags
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.setFlag(QGraphicsItem.ItemIsMovable, False)
        
    def boundingRect(self) -> QRectF:
        """Return the bounding rectangle of the court."""
        # Add some margin around the court
        margin = 2.0 * self.scale_factor
        return self.court_rect.adjusted(-margin, -margin, margin, margin)
    
    def set_scale(self, scale_factor: float):
        """
        Set the scale factor (pixels per meter).
        
        Args:
            scale_factor (float): New scale factor
        """
        self.scale_factor = scale_factor
        
        # Update court rectangle
        self.court_rect = QRectF(-self.COURT_WIDTH/2 * self.scale_factor, 
                               -self.COURT_LENGTH/2 * self.scale_factor,
                               self.COURT_WIDTH * self.scale_factor, 
                               self.COURT_LENGTH * self.scale_factor)
        
        # Trigger redraw
        self.update()
    
    def paint(self, painter, option, widget=None):
        """
        Paint the court.
        
        Args:
            painter: QPainter
            option: QStyleOptionGraphicsItem
            widget: QWidget (optional)
        """
        # Enable antialiasing for smoother lines
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        
        # Draw court surface
        painter.setPen(Qt.NoPen)
        painter.setBrush(self.court_brush)
        painter.drawRect(self.court_rect)
        
        # Draw court lines
        painter.setPen(self.line_pen)
        
        # Singles sidelines
        half_width = self.COURT_WIDTH/2 * self.scale_factor
        half_length = self.COURT_LENGTH/2 * self.scale_factor
        
        # Left sideline
        painter.drawLine(-half_width, -half_length, -half_width, half_length)
        # Right sideline
        painter.drawLine(half_width, -half_length, half_width, half_length)
        # Baseline top
        painter.drawLine(-half_width, -half_length, half_width, -half_length)
        # Baseline bottom
        painter.drawLine(-half_width, half_length, half_width, half_length)
        
        # Service lines
        service_line_y1 = -half_length + self.SERVICE_LINE_DIST * self.scale_factor
        service_line_y2 = half_length - self.SERVICE_LINE_DIST * self.scale_factor
        
        # Top service line
        painter.drawLine(-half_width, service_line_y1, half_width, service_line_y1)
        # Bottom service line
        painter.drawLine(-half_width, service_line_y2, half_width, service_line_y2)
        
        # Center service line
        painter.drawLine(0, service_line_y1, 0, service_line_y2)
        
        # Center marks on baselines
        center_mark_half_length = self.CENTER_MARK_LENGTH/2 * self.scale_factor
        
        # Top center mark
        painter.drawLine(0, -half_length, 0, -half_length + center_mark_half_length * 2)
        # Bottom center mark
        painter.drawLine(0, half_length, 0, half_length - center_mark_half_length * 2)
        
        # Net line
        painter.setPen(self.net_pen)
        painter.drawLine(-half_width, 0, half_width, 0)


class TrajectoryGraphicsItem(QGraphicsPathItem):
    """Graphics item for displaying ball trajectory."""
    
    # Maximum number of positions to keep in trajectory
    MAX_TRAJECTORY_POINTS = 100
    
    # Colors
    TRAJECTORY_COLOR = QColor(255, 100, 0, 180)  # Orange semi-transparent
    
    def __init__(self, parent=None):
        """Initialize the trajectory graphics item."""
        super(TrajectoryGraphicsItem, self).__init__(parent)
        
        # Set up pen
        self.trajectory_pen = QPen(self.TRAJECTORY_COLOR, 2, Qt.DashLine)
        self.setPen(self.trajectory_pen)
        
        # Initialize path
        self.path = QPainterPath()
        self.setPath(self.path)
        
        # Store positions
        self.positions = []
        self.scale_factor = 20  # Default scale (pixels per meter)
        
    def set_scale(self, scale_factor: float):
        """
        Set the scale factor (pixels per meter).
        
        Args:
            scale_factor (float): New scale factor
        """
        self.scale_factor = scale_factor
        self.update_path()
    
    def add_position(self, x: float, y: float, z: float):
        """
        Add a new 3D position to the trajectory.
        
        Args:
            x (float): X coordinate in court frame (meters)
            y (float): Y coordinate in court frame (meters)
            z (float): Z coordinate in court frame (meters)
        """
        # Add position to list
        self.positions.append((x, y, z))
        
        # Limit number of positions
        if len(self.positions) > self.MAX_TRAJECTORY_POINTS:
            self.positions.pop(0)
        
        # Update path
        self.update_path()
    
    def update_path(self):
        """Update the trajectory path based on current positions."""
        # Create new path
        path = QPainterPath()
        
        # Add positions to path
        for i, (x, y, z) in enumerate(self.positions):
            # Convert court coordinates to view coordinates
            px = x * self.scale_factor
            py = y * self.scale_factor
            
            if i == 0:
                path.moveTo(px, py)
            else:
                path.lineTo(px, py)
        
        # Update path
        self.setPath(path)
    
    def clear(self):
        """Clear the trajectory."""
        self.positions = []
        self.path = QPainterPath()
        self.setPath(self.path)


class BounceGraphicsItem(QGraphicsEllipseItem):
    """Graphics item for displaying a ball bounce."""
    
    # Colors
    IN_COLOR = QColor(0, 255, 0, 200)  # Green semi-transparent
    OUT_COLOR = QColor(255, 0, 0, 200)  # Red semi-transparent
    
    def __init__(self, x: float, y: float, is_inside_court: bool, parent=None):
        """
        Initialize the bounce graphics item.
        
        Args:
            x (float): X coordinate in court frame (meters)
            y (float): Y coordinate in court frame (meters)
            is_inside_court (bool): Whether the bounce is inside the court
            parent: Parent item (optional)
        """
        # Default size
        self.radius = 10  # pixels
        self.scale_factor = 20  # Default scale (pixels per meter)
        self.is_inside_court = is_inside_court
        self.x_pos = x
        self.y_pos = y
        
        # Calculate position in pixels
        px = x * self.scale_factor
        py = y * self.scale_factor
        
        super(BounceGraphicsItem, self).__init__(
            px - self.radius, py - self.radius, 
            self.radius * 2, self.radius * 2, 
            parent
        )
        
        # Set color based on in/out status
        color = self.IN_COLOR if is_inside_court else self.OUT_COLOR
        self.setBrush(QBrush(color))
        self.setPen(QPen(Qt.NoPen))
        
        # Animation timer for fade out
        self.opacity = 1.0
        self.fade_timer = QTimer()
        self.fade_timer.timeout.connect(self.fade)
        self.fade_timer.start(100)  # Update every 100ms
        
    def set_scale(self, scale_factor: float):
        """
        Set the scale factor (pixels per meter).
        
        Args:
            scale_factor (float): New scale factor
        """
        # Update scale factor
        self.scale_factor = scale_factor
        
        # Update position and size
        px = self.x_pos * self.scale_factor
        py = self.y_pos * self.scale_factor
        
        self.setRect(px - self.radius, py - self.radius, 
                    self.radius * 2, self.radius * 2)
    
    def fade(self):
        """Fade the bounce marker over time."""
        # Reduce opacity
        self.opacity -= 0.01
        
        if self.opacity <= 0:
            # Remove from scene
            if self.scene():
                self.scene().removeItem(self)
            # Stop timer
            self.fade_timer.stop()
        else:
            # Update opacity
            color = self.IN_COLOR if self.is_inside_court else self.OUT_COLOR
            self.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 
                                      int(255 * self.opacity))))


class BallGraphicsItem(QGraphicsEllipseItem):
    """Graphics item for displaying the current ball position."""
    
    # Colors
    BALL_COLOR = QColor(255, 255, 0)  # Yellow
    SHADOW_COLOR = QColor(100, 100, 100, 150)  # Gray semi-transparent
    
    def __init__(self, parent=None):
        """Initialize the ball graphics item."""
        # Default size
        self.radius = 6  # pixels
        self.scale_factor = 20  # Default scale (pixels per meter)
        
        super(BallGraphicsItem, self).__init__(0, 0, self.radius * 2, self.radius * 2, parent)
        
        # Set color
        self.setBrush(QBrush(self.BALL_COLOR))
        self.setPen(QPen(Qt.black, 1))
        
        # Shadow item (shows where ball is relative to ground)
        self.shadow = QGraphicsEllipseItem(0, 0, self.radius, self.radius, parent)
        self.shadow.setBrush(QBrush(self.SHADOW_COLOR))
        self.shadow.setPen(QPen(Qt.NoPen))
        
        # Hide initially
        self.hide()
        self.shadow.hide()
        
        # Position and height
        self.x_pos = 0
        self.y_pos = 0
        self.z_pos = 0
    
    def set_scale(self, scale_factor: float):
        """
        Set the scale factor (pixels per meter).
        
        Args:
            scale_factor (float): New scale factor
        """
        # Update scale factor
        self.scale_factor = scale_factor
        
        # Update position
        self.update_position(self.x_pos, self.y_pos, self.z_pos)
    
    def update_position(self, x: float, y: float, z: float):
        """
        Update the ball position on the tennis court overlay.
        
        Args:
            x (float): X coordinate in court frame (meters)
            y (float): Y coordinate in court frame (meters)
            z (float): Z coordinate in court frame (meters)
        """
        # Store position
        self.x_pos = x
        self.y_pos = y
        self.z_pos = z
        
        # Convert court coordinates to scene coordinates
        px = x * self.scale_factor
        py = y * self.scale_factor
        
        # Update ball position
        self.setRect(px - self.radius, py - self.radius, 
                    self.radius * 2, self.radius * 2)
        
        # Update shadow
        self.shadow.setRect(px - self.radius/2, py - self.radius/2, 
                           self.radius, self.radius)
        
        # Log position update
        logging.debug(f"Ball position updated: court=({x:.2f}, {y:.2f}, {z:.2f}), px={px:.1f}, py={py:.1f}")
        
        # Show ball only when it's above the ground
        if z >= 0:
            self.show()
            self.shadow.show()
        else:
            self.hide()
            self.shadow.hide()


class BounceOverlayWidget(QGraphicsView):
    """Widget for displaying the tennis court and ball bounces in a 2D top-down view."""
    
    def __init__(self, parent=None):
        """
        Initialize the bounce overlay widget.
        
        Args:
            parent (QWidget, optional): Parent widget
        """
        super(BounceOverlayWidget, self).__init__(parent)
        
        # Set up scene
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # Configure view
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        
        # Create court item
        self.court = CourtGraphicsItem()
        self.scene.addItem(self.court)
        
        # Create trajectory item
        self.trajectory = TrajectoryGraphicsItem()
        self.scene.addItem(self.trajectory)
        
        # Create ball item
        self.ball = BallGraphicsItem()
        self.scene.addItem(self.ball)
        
        # Keep track of bounce items
        self.bounces = []
        
        # Scale factor (pixels per meter)
        self.scale_factor = 20
        
        # Frame reference
        self.court_frame = CourtFrame()
        
        # Center view on court
        self.centerOn(0, 0)
        
        # Set initial zoom level
        self.setSceneRect(-self.court.COURT_WIDTH/2 * self.scale_factor - 50,
                        -self.court.COURT_LENGTH/2 * self.scale_factor - 50,
                        self.court.COURT_WIDTH * self.scale_factor + 100,
                        self.court.COURT_LENGTH * self.scale_factor + 100)
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        
        # Game analyzer reference
        self.game_analyzer = None
        
    def resizeEvent(self, event):
        """
        Handle resize events.
        
        Args:
            event: QResizeEvent
        """
        super(BounceOverlayWidget, self).resizeEvent(event)
        
        # Fit view to court
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        
    def wheelEvent(self, event):
        """
        Handle mouse wheel events for zooming.
        
        Args:
            event: QWheelEvent
        """
        # Get zoom factor
        factor = 1.2
        if event.angleDelta().y() < 0:
            factor = 1.0 / factor
            
        # Apply zoom
        self.scale(factor, factor)
        
    def reset(self):
        """Reset the overlay view."""
        # Clear trajectory
        self.trajectory.clear()
        
        # Remove all bounce items
        for bounce in self.bounces:
            self.scene.removeItem(bounce)
        self.bounces = []
        
        # Hide ball
        self.ball.hide()
        self.ball.shadow.hide()
        
        # Reset view
        self.centerOn(0, 0)
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        
    def add_bounce(self, x: float, y: float, is_inside_court: bool):
        """
        Add a bounce marker to the court.
        
        Args:
            x (float): X coordinate in court frame (meters)
            y (float): Y coordinate in court frame (meters)
            is_inside_court (bool): Whether the bounce is inside the court
        """
        # Create bounce item
        bounce = BounceGraphicsItem(x, y, is_inside_court)
        self.scene.addItem(bounce)
        
        # Add to list
        self.bounces.append(bounce)
        
        # Remove old bounces
        if len(self.bounces) > 20:  # Keep only 20 most recent bounces
            old_bounce = self.bounces.pop(0)
            if old_bounce.scene():
                self.scene.removeItem(old_bounce)
    
    def update_ball_position(self, x: float, y: float, z: float):
        """
        Update the ball position.
        
        Args:
            x (float): X coordinate in court frame (meters)
            y (float): Y coordinate in court frame (meters)
            z (float): Z coordinate in court frame (meters)
        """
        # Update ball position
        self.ball.update_position(x, y, z)
        
        # Add position to trajectory
        self.trajectory.add_position(x, y, z)
        
    def connect_game_analyzer(self, game_analyzer):
        """
        Connect signals from the GameAnalyzer to this widget.
        
        Args:
            game_analyzer: The GameAnalyzer instance
        """
        # Disconnect any existing connections
        if self.game_analyzer:
            self.game_analyzer.court_position_updated.disconnect(self._on_ball_position_updated)
            self.game_analyzer.bounce_detected.disconnect(self._on_bounce_detected)
        
        self.game_analyzer = game_analyzer
        
        if game_analyzer:
            game_analyzer.court_position_updated.connect(self._on_ball_position_updated)
            game_analyzer.bounce_detected.connect(self._on_bounce_detected)
            
    def _on_ball_position_updated(self, x: float, y: float, z: float):
        """
        Update the ball position when receiving a signal from the game analyzer.
        
        Args:
            x (float): X position in court coordinates (meters)
            y (float): Y position in court coordinates (meters)
            z (float): Z position in court coordinates (meters)
        """
        logging.debug(f"Ball position update received: ({x:.2f}, {y:.2f}, {z:.2f})")
        
        # Update the ball position
        self.update_ball_position(x, y, z)
    
    def _on_bounce_detected(self, bounce_event):
        """
        Handle bounce events from game analyzer.
        
        Args:
            bounce_event: BounceEvent object
        """
        # Add bounce marker to overlay
        position = bounce_event.position
        self.add_bounce(position[0], position[1], bounce_event.is_inside_court)
 