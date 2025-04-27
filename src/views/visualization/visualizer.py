#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualizer interface and implementations.
Provides a unified interface for drawing elements on both OpenCV images and Qt scenes.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Optional, Union, TypeVar

import cv2
import numpy as np
from PySide6.QtWidgets import QGraphicsScene, QGraphicsItem, QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsLineItem
from PySide6.QtGui import QPen, QBrush, QColor
from PySide6.QtCore import Qt, QPointF

from src.utils import viz_utils
from src.utils.constants import COLOR, TRACKING, ROI

# Type variable for the raw input/output
T = TypeVar('T')

class IVisualizer(ABC):
    """
    Interface for visualization operations.
    Implementations can target different backends (OpenCV, Qt, etc).
    """
    
    @abstractmethod
    def draw_point(self, raw: T, 
                  point: Tuple[int, int], 
                  color: Optional[Tuple[int, int, int]] = None,
                  radius: int = 5,
                  thickness: int = -1,
                  label: Optional[str] = None,
                  cross_size: int = 10) -> T:
        """Draw a point with optional cross and label."""
        pass
    
    @abstractmethod
    def draw_points(self, raw: T, 
                   points: List[Tuple[int, int]], 
                   color: Optional[Tuple[int, int, int]] = None,
                   radius: int = 5,
                   thickness: int = -1,
                   labels: Optional[List[str]] = None,
                   numbered: bool = False) -> T:
        """Draw multiple points."""
        pass
    
    @abstractmethod
    def draw_line(self, raw: T, 
                 pt1: Tuple[int, int], 
                 pt2: Tuple[int, int], 
                 color: Optional[Tuple[int, int, int]] = None,
                 thickness: int = 2,
                 dashed: bool = False) -> T:
        """Draw a line between two points."""
        pass
    
    @abstractmethod
    def draw_rectangle(self, raw: T,
                      pt1: Tuple[int, int],
                      pt2: Tuple[int, int],
                      color: Optional[Tuple[int, int, int]] = None,
                      thickness: int = 2,
                      fill: bool = False,
                      fill_alpha: float = 0.2) -> T:
        """Draw a rectangle between two corner points."""
        pass
    
    @abstractmethod
    def draw_grid_lines(self, raw: T, 
                       points: List[Tuple[int, int]], 
                       rows: int, 
                       cols: int,
                       color: Optional[Tuple[int, int, int]] = None,
                       thickness: int = 2,
                       dashed: bool = False) -> T:
        """Draw grid lines connecting points in a grid pattern."""
        pass
    
    @abstractmethod
    def draw_roi(self, raw: T, 
                roi: Union[Tuple[int, int, int, int], Dict], 
                color: Optional[Tuple[int, int, int]] = None,
                thickness: int = TRACKING.ROI_THICKNESS,
                show_center: bool = True) -> T:
        """Draw ROI rectangle and optional center point."""
        pass
    
    @abstractmethod
    def draw_circle(self, raw: T, 
                   center: Tuple[int, int], 
                   radius: int,
                   color: Optional[Tuple[int, int, int]] = None,
                   thickness: int = TRACKING.CIRCLE_THICKNESS,
                   show_center: bool = True,
                   label: Optional[str] = None) -> T:
        """Draw a circle with optional center point and label."""
        pass
    
    @abstractmethod
    def draw_circles(self, raw: T,
                    circles: List[Tuple[int, int, int]],
                    color: Optional[Tuple[int, int, int]] = None,
                    thickness: int = TRACKING.CIRCLE_THICKNESS,
                    label_circles: bool = False) -> T:
        """Draw multiple circles."""
        pass
    
    @abstractmethod
    def draw_prediction(self, raw: T,
                       current_pos: Optional[Tuple[int, int]],
                       predicted_pos: Tuple[int, int],
                       arrow_color: Optional[Tuple[int, int, int]] = None,
                       thickness: int = TRACKING.PREDICTION_THICKNESS,
                       draw_uncertainty: bool = False,
                       uncertainty_radius: int = TRACKING.UNCERTAINTY_RADIUS) -> T:
        """Draw prediction arrow between current and predicted position."""
        pass
    
    @abstractmethod
    def draw_trajectory(self, raw: T,
                       positions: List[Tuple[float, float]],
                       color: Optional[Tuple[int, int, int]] = None,
                       thickness: int = TRACKING.TRAJECTORY_THICKNESS,
                       max_points: int = TRACKING.TRAJECTORY_MAX_POINTS) -> T:
        """Draw trajectory from list of positions."""
        pass


class OpenCVVisualizer(IVisualizer):
    """
    OpenCV implementation of the IVisualizer interface.
    Uses the viz_utils module to draw on OpenCV images.
    """
    
    def draw_point(self, raw: np.ndarray, 
                  point: Tuple[int, int], 
                  color: Optional[Tuple[int, int, int]] = None,
                  radius: int = 5,
                  thickness: int = -1,
                  label: Optional[str] = None,
                  cross_size: int = 10) -> np.ndarray:
        """Draw a point with optional cross and label on an OpenCV image."""
        if color is None:
            color = COLOR.RED
        return viz_utils.draw_point(raw, point, color, radius, thickness, label, cross_size)
    
    def draw_points(self, raw: np.ndarray, 
                   points: List[Tuple[int, int]], 
                   color: Optional[Tuple[int, int, int]] = None,
                   radius: int = 5,
                   thickness: int = -1,
                   labels: Optional[List[str]] = None,
                   numbered: bool = False) -> np.ndarray:
        """Draw multiple points on an OpenCV image."""
        if color is None:
            color = COLOR.RED
        return viz_utils.draw_points(raw, points, color, radius, thickness, labels, numbered)
    
    def draw_line(self, raw: np.ndarray, 
                 pt1: Tuple[int, int], 
                 pt2: Tuple[int, int], 
                 color: Optional[Tuple[int, int, int]] = None,
                 thickness: int = 2,
                 dashed: bool = False) -> np.ndarray:
        """Draw a line between two points on an OpenCV image."""
        if color is None:
            color = COLOR.GREEN
        return viz_utils.draw_line(raw, pt1, pt2, color, thickness, dashed=dashed)
    
    def draw_rectangle(self, raw: np.ndarray,
                      pt1: Tuple[int, int],
                      pt2: Tuple[int, int],
                      color: Optional[Tuple[int, int, int]] = None,
                      thickness: int = 2,
                      fill: bool = False,
                      fill_alpha: float = 0.2) -> np.ndarray:
        """Draw a rectangle between two corner points on an OpenCV image."""
        output = raw.copy()
        if color is None:
            color = COLOR.GREEN
            
        if fill:
            # Create a filled rectangle with transparency
            overlay = output.copy()
            cv2.rectangle(overlay, pt1, pt2, color, -1)  # Filled rectangle
            cv2.addWeighted(overlay, fill_alpha, output, 1 - fill_alpha, 0, output)
            
        # Draw the rectangle outline
        cv2.rectangle(output, pt1, pt2, color, thickness)
        return output
    
    def draw_grid_lines(self, raw: np.ndarray, 
                       points: List[Tuple[int, int]], 
                       rows: int, 
                       cols: int,
                       color: Optional[Tuple[int, int, int]] = None,
                       thickness: int = 2,
                       dashed: bool = False) -> np.ndarray:
        """Draw grid lines connecting points in a grid pattern on an OpenCV image."""
        if color is None:
            color = COLOR.GREEN
        return viz_utils.draw_grid_lines(raw, points, rows, cols, color, thickness, dashed)
    
    def draw_roi(self, raw: np.ndarray, 
                roi: Union[Tuple[int, int, int, int], Dict], 
                color: Optional[Tuple[int, int, int]] = None,
                thickness: int = TRACKING.ROI_THICKNESS,
                show_center: bool = True) -> np.ndarray:
        """Draw ROI rectangle and optional center point on an OpenCV image."""
        if color is None:
            color = COLOR.GREEN
        return viz_utils.draw_roi(raw, roi, color, thickness, show_center)
    
    def draw_circle(self, raw: np.ndarray, 
                   center: Tuple[int, int], 
                   radius: int,
                   color: Optional[Tuple[int, int, int]] = None,
                   thickness: int = TRACKING.CIRCLE_THICKNESS,
                   show_center: bool = True,
                   label: Optional[str] = None) -> np.ndarray:
        """Draw a circle with optional center point and label on an OpenCV image."""
        if color is None:
            color = COLOR.YELLOW
        return viz_utils.draw_circle(raw, center, radius, color, thickness, show_center, COLOR.RED, label)
    
    def draw_circles(self, raw: np.ndarray,
                    circles: List[Tuple[int, int, int]],
                    color: Optional[Tuple[int, int, int]] = None,
                    thickness: int = TRACKING.CIRCLE_THICKNESS,
                    label_circles: bool = False) -> np.ndarray:
        """Draw multiple circles on an OpenCV image."""
        if color is None:
            color = TRACKING.MAIN_CIRCLE_COLOR
        return viz_utils.draw_circles(raw, circles, color, thickness, label_circles)
    
    def draw_prediction(self, raw: np.ndarray,
                       current_pos: Optional[Tuple[int, int]],
                       predicted_pos: Tuple[int, int],
                       arrow_color: Optional[Tuple[int, int, int]] = None,
                       thickness: int = TRACKING.PREDICTION_THICKNESS,
                       draw_uncertainty: bool = False,
                       uncertainty_radius: int = TRACKING.UNCERTAINTY_RADIUS) -> np.ndarray:
        """Draw prediction arrow between current and predicted position on an OpenCV image."""
        if arrow_color is None:
            arrow_color = TRACKING.PREDICTION_ARROW_COLOR
        return viz_utils.draw_prediction(raw, current_pos, predicted_pos, arrow_color, thickness, draw_uncertainty, uncertainty_radius)
    
    def draw_trajectory(self, raw: np.ndarray,
                       positions: List[Tuple[float, float]],
                       color: Optional[Tuple[int, int, int]] = None,
                       thickness: int = TRACKING.TRAJECTORY_THICKNESS,
                       max_points: int = TRACKING.TRAJECTORY_MAX_POINTS) -> np.ndarray:
        """Draw trajectory from list of positions on an OpenCV image."""
        if color is None:
            color = TRACKING.TRAJECTORY_COLOR
        return viz_utils.draw_trajectory(raw, positions, color, thickness, max_points)


class QtVisualizer(IVisualizer):
    """
    Qt implementation of the IVisualizer interface.
    Draws on QGraphicsScene using Qt graphics items.
    """
    
    def __init__(self, scene: QGraphicsScene):
        """
        Initialize with a QGraphicsScene to draw on.
        
        Args:
            scene: QGraphicsScene to draw on
        """
        self.scene = scene
        self.items = {}  # Dictionary to store created items by ID
    
    def _get_qt_color(self, color: Optional[Tuple[int, int, int]]) -> QColor:
        """Convert BGR tuple to QColor (RGB)."""
        if color is None:
            return QColor(COLOR.RED[2], COLOR.RED[1], COLOR.RED[0])  # BGR to RGB
        return QColor(color[2], color[1], color[0])  # BGR to RGB
    
    def draw_point(self, raw: QGraphicsScene, 
                  point: Tuple[int, int], 
                  color: Optional[Tuple[int, int, int]] = None,
                  radius: int = 5,
                  thickness: int = -1,
                  label: Optional[str] = None,
                  cross_size: int = 10) -> QGraphicsScene:
        """Draw a point with optional cross and label on a Qt scene."""
        x, y = point
        qt_color = self._get_qt_color(color)
        
        # Create pen and brush
        pen = QPen(qt_color, 1 if thickness <= 0 else thickness)
        brush = QBrush(qt_color) if thickness <= 0 else QBrush()
        
        # Create circle for point
        circle = self.scene.addEllipse(x - radius, y - radius, radius * 2, radius * 2, pen, brush)
        
        # Draw cross markers
        h_line = self.scene.addLine(x - cross_size, y, x + cross_size, y, QPen(qt_color, 2))
        v_line = self.scene.addLine(x, y - cross_size, x, y + cross_size, QPen(qt_color, 2))
        
        # Add label if provided
        if label:
            text_item = self.scene.addText(label)
            text_item.setPos(x + 10, y - 10)
            text_item.setDefaultTextColor(qt_color)
        
        return self.scene
    
    def draw_points(self, raw: QGraphicsScene, 
                   points: List[Tuple[int, int]], 
                   color: Optional[Tuple[int, int, int]] = None,
                   radius: int = 5,
                   thickness: int = -1,
                   labels: Optional[List[str]] = None,
                   numbered: bool = False) -> QGraphicsScene:
        """Draw multiple points on a Qt scene."""
        for i, point in enumerate(points):
            # Determine label
            point_label = None
            if numbered:
                point_label = str(i + 1)
            elif labels and i < len(labels):
                point_label = labels[i]
                
            # Draw the point
            self.draw_point(raw, point, color, radius, thickness, point_label)
            
        return self.scene
    
    def draw_line(self, raw: QGraphicsScene, 
                 pt1: Tuple[int, int], 
                 pt2: Tuple[int, int], 
                 color: Optional[Tuple[int, int, int]] = None,
                 thickness: int = 2,
                 dashed: bool = False) -> QGraphicsScene:
        """Draw a line between two points on a Qt scene."""
        x1, y1 = pt1
        x2, y2 = pt2
        qt_color = self._get_qt_color(color if color else COLOR.GREEN)
        
        pen = QPen(qt_color, thickness)
        if dashed:
            pen.setStyle(Qt.DashLine)
            
        line = self.scene.addLine(x1, y1, x2, y2, pen)
        return self.scene
    
    def draw_rectangle(self, raw: QGraphicsScene,
                      pt1: Tuple[int, int],
                      pt2: Tuple[int, int],
                      color: Optional[Tuple[int, int, int]] = None,
                      thickness: int = 2,
                      fill: bool = False,
                      fill_alpha: float = 0.2) -> QGraphicsScene:
        """Draw a rectangle between two corner points on a Qt scene."""
        x1, y1 = pt1
        x2, y2 = pt2
        qt_color = self._get_qt_color(color if color else COLOR.GREEN)
        
        # Create pen for outline
        pen = QPen(qt_color, thickness)
        
        # Create brush for fill if needed
        brush = Qt.NoBrush
        if fill:
            # Create semi-transparent color
            fill_color = QColor(qt_color)
            fill_color.setAlphaF(fill_alpha)
            brush = QBrush(fill_color)
        
        # Calculate rectangle parameters
        x = min(x1, x2)
        y = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        # Draw the rectangle
        rect = self.scene.addRect(x, y, width, height, pen, brush)
        return self.scene
    
    def draw_grid_lines(self, raw: QGraphicsScene, 
                       points: List[Tuple[int, int]], 
                       rows: int, 
                       cols: int,
                       color: Optional[Tuple[int, int, int]] = None,
                       thickness: int = 2,
                       dashed: bool = False) -> QGraphicsScene:
        """Draw grid lines connecting points in a grid pattern on a Qt scene."""
        # Draw horizontal lines
        for row in range(rows):
            for col in range(cols - 1):
                idx1 = row * cols + col
                idx2 = row * cols + col + 1
                
                if idx1 < len(points) and idx2 < len(points):
                    self.draw_line(raw, points[idx1], points[idx2], color, thickness, dashed)
        
        # Draw vertical lines
        for col in range(cols):
            for row in range(rows - 1):
                idx1 = row * cols + col
                idx2 = (row + 1) * cols + col
                
                if idx1 < len(points) and idx2 < len(points):
                    self.draw_line(raw, points[idx1], points[idx2], color, thickness, dashed)
        
        return self.scene
    
    def draw_roi(self, raw: QGraphicsScene, 
                roi: Union[Tuple[int, int, int, int], Dict], 
                color: Optional[Tuple[int, int, int]] = None,
                thickness: int = TRACKING.ROI_THICKNESS,
                show_center: bool = True) -> QGraphicsScene:
        """Draw ROI rectangle and optional center point on a Qt scene."""
        # Convert dict format to tuple if needed
        if isinstance(roi, dict):
            if all(k in roi for k in ['x', 'y', 'width', 'height']):
                x, y = roi['x'], roi['y']
                w, h = roi['width'], roi['height']
            else:
                return self.scene  # Invalid format
        else:
            # Assume it's already a tuple (x, y, w, h)
            x, y, w, h = roi
        
        qt_color = self._get_qt_color(color if color else COLOR.GREEN)
        pen = QPen(qt_color, thickness)
        
        # Draw rectangle outline
        rect = self.scene.addRect(x, y, w, h, pen)
        
        # Draw center point if requested
        if show_center:
            center_x, center_y = x + w // 2, y + h // 2
            center_color = self._get_qt_color(COLOR.RED)
            
            # Add center point
            self.scene.addEllipse(
                center_x - ROI.CENTER_MARKER_SIZE // 2, 
                center_y - ROI.CENTER_MARKER_SIZE // 2,
                ROI.CENTER_MARKER_SIZE, 
                ROI.CENTER_MARKER_SIZE, 
                QPen(center_color), 
                QBrush(center_color)
            )
            
            # Add center cross
            self.scene.addLine(center_x - 10, center_y, center_x + 10, center_y, QPen(center_color, 2))
            self.scene.addLine(center_x, center_y - 10, center_x, center_y + 10, QPen(center_color, 2))
        
        return self.scene
    
    def draw_circle(self, raw: QGraphicsScene, 
                   center: Tuple[int, int], 
                   radius: int,
                   color: Optional[Tuple[int, int, int]] = None,
                   thickness: int = TRACKING.CIRCLE_THICKNESS,
                   show_center: bool = True,
                   label: Optional[str] = None) -> QGraphicsScene:
        """Draw a circle with optional center point and label on a Qt scene."""
        x, y = center
        qt_color = self._get_qt_color(color if color else COLOR.YELLOW)
        center_color = self._get_qt_color(COLOR.RED)
        
        pen = QPen(qt_color, thickness)
        
        # Draw circle
        circle = self.scene.addEllipse(x - radius, y - radius, radius * 2, radius * 2, pen)
        
        # Draw center point
        if show_center:
            self.scene.addEllipse(
                x - ROI.CENTER_MARKER_SIZE // 4, 
                y - ROI.CENTER_MARKER_SIZE // 4,
                ROI.CENTER_MARKER_SIZE // 2, 
                ROI.CENTER_MARKER_SIZE // 2, 
                QPen(center_color), 
                QBrush(center_color)
            )
        
        # Draw label if provided
        if label:
            text_item = self.scene.addText(label)
            text_item.setPos(x + radius, y - 10)
            text_item.setDefaultTextColor(qt_color)
        
        return self.scene
    
    def draw_circles(self, raw: QGraphicsScene,
                    circles: List[Tuple[int, int, int]],
                    color: Optional[Tuple[int, int, int]] = None,
                    thickness: int = TRACKING.CIRCLE_THICKNESS,
                    label_circles: bool = False) -> QGraphicsScene:
        """Draw multiple circles on a Qt scene."""
        if circles is None or len(circles) == 0:
            return self.scene
        
        main_color = color if color is not None else TRACKING.MAIN_CIRCLE_COLOR
        
        for i, circle in enumerate(circles):
            x, y, r = circle
            
            # Draw the circle
            label = str(i + 1) if label_circles else None
            self.draw_circle(
                self.scene, (int(x), int(y)), int(r), 
                main_color, thickness, True, label
            )
        
        return self.scene
    
    def draw_prediction(self, raw: QGraphicsScene,
                       current_pos: Optional[Tuple[int, int]],
                       predicted_pos: Tuple[int, int],
                       arrow_color: Optional[Tuple[int, int, int]] = None,
                       thickness: int = TRACKING.PREDICTION_THICKNESS,
                       draw_uncertainty: bool = False,
                       uncertainty_radius: int = TRACKING.UNCERTAINTY_RADIUS) -> QGraphicsScene:
        """Draw prediction arrow between current and predicted position on a Qt scene."""
        qt_color = self._get_qt_color(arrow_color if arrow_color else TRACKING.PREDICTION_ARROW_COLOR)
        pen = QPen(qt_color, thickness)
        
        # Draw arrow only if we have both positions
        if current_pos is not None:
            # Draw line from current to predicted position
            line = self.scene.addLine(
                current_pos[0], current_pos[1],
                predicted_pos[0], predicted_pos[1],
                pen
            )
            
            # Add an arrowhead - in Qt we need to manually draw this
            # Calculate arrow direction
            dx = predicted_pos[0] - current_pos[0]
            dy = predicted_pos[1] - current_pos[1]
            length = (dx**2 + dy**2)**0.5
            
            if length > 0:
                # Normalize
                dx, dy = dx/length, dy/length
                
                # Arrow head size
                arrow_size = 10
                
                # Arrow head points
                arrow_pt1 = (
                    predicted_pos[0] - arrow_size * dx + arrow_size * dy * 0.5,
                    predicted_pos[1] - arrow_size * dy - arrow_size * dx * 0.5
                )
                arrow_pt2 = (
                    predicted_pos[0] - arrow_size * dx - arrow_size * dy * 0.5,
                    predicted_pos[1] - arrow_size * dy + arrow_size * dx * 0.5
                )
                
                # Draw arrowhead
                self.scene.addLine(predicted_pos[0], predicted_pos[1], arrow_pt1[0], arrow_pt1[1], pen)
                self.scene.addLine(predicted_pos[0], predicted_pos[1], arrow_pt2[0], arrow_pt2[1], pen)
        
        # Draw the predicted position point
        self.scene.addEllipse(
            predicted_pos[0] - 5, predicted_pos[1] - 5,
            10, 10,
            pen, QBrush(qt_color)
        )
        
        # Draw uncertainty circle if requested
        if draw_uncertainty:
            uncertainty_pen = QPen(qt_color, 1)
            uncertainty_pen.setStyle(Qt.DashLine)
            self.scene.addEllipse(
                predicted_pos[0] - uncertainty_radius,
                predicted_pos[1] - uncertainty_radius,
                uncertainty_radius * 2,
                uncertainty_radius * 2,
                uncertainty_pen
            )
        
        return self.scene
    
    def draw_trajectory(self, raw: QGraphicsScene,
                       positions: List[Tuple[float, float]],
                       color: Optional[Tuple[int, int, int]] = None,
                       thickness: int = TRACKING.TRAJECTORY_THICKNESS,
                       max_points: int = TRACKING.TRAJECTORY_MAX_POINTS) -> QGraphicsScene:
        """Draw trajectory from list of positions on a Qt scene."""
        if not positions or len(positions) < 2:
            return self.scene
        
        # Limit number of positions to avoid clutter
        if len(positions) > max_points:
            positions = positions[-max_points:]
        
        qt_color = self._get_qt_color(color if color else TRACKING.TRAJECTORY_COLOR)
        pen = QPen(qt_color, thickness)
        
        # Draw lines connecting the points
        for i in range(len(positions) - 1):
            x1, y1 = positions[i]
            x2, y2 = positions[i + 1]
            self.scene.addLine(int(x1), int(y1), int(x2), int(y2), pen)
        
        return self.scene


class VisualizerFactory:
    """
    Factory class for creating IVisualizer instances.
    """
    
    @staticmethod
    def create(backend: str = "opencv", **kwargs) -> IVisualizer:
        """
        Create and return an IVisualizer implementation based on the specified backend.
        
        Args:
            backend: The visualization backend ("opencv" or "qt")
            **kwargs: Additional arguments for specific visualizer implementations
                      For Qt: 'scene' parameter is required (QGraphicsScene)
        
        Returns:
            An instance of IVisualizer implementation
            
        Raises:
            ValueError: If the backend is unknown or required parameters are missing
        """
        if backend.lower() == "opencv":
            return OpenCVVisualizer()
        elif backend.lower() == "qt":
            if "scene" not in kwargs:
                raise ValueError("Qt visualizer requires 'scene' parameter")
            return QtVisualizer(kwargs["scene"])
        else:
            raise ValueError(f"Unknown visualization backend: {backend}") 