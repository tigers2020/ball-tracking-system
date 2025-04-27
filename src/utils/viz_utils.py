#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization Utilities Module.
This module centralizes drawing functions used across the application.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, List, Dict, Any, Optional, Union

from src.utils.constants import COLOR, TRACKING, ROI, LAYOUT

# =============================================
# Point Drawing Functions
# =============================================

def draw_point(img: np.ndarray, 
              point: Tuple[int, int], 
              color: Tuple[int, int, int] = COLOR.RED,
              radius: int = 5,
              thickness: int = -1,
              label: Optional[str] = None,
              cross_size: int = 10) -> np.ndarray:
    """
    Draw a point with optional cross and label on an image.
    
    Args:
        img: Original BGR image
        point: (x, y) coordinates of the point
        color: Color for point marker (BGR)
        radius: Radius of the point circle
        thickness: Thickness of the circle (-1 for filled)
        label: Optional text label to display near the point
        cross_size: Size of the cross marker
        
    Returns:
        Image with point drawn
    """
    output_img = img.copy()
    x, y = point
    
    # Draw filled circle for the point
    cv2.circle(output_img, (x, y), radius, color, thickness)
    
    # Draw cross markers for better visibility
    cv2.line(output_img, (x - cross_size, y), (x + cross_size, y), color, 2)
    cv2.line(output_img, (x, y - cross_size), (x, y + cross_size), color, 2)
    
    # Draw label if provided
    if label:
        cv2.putText(output_img, label, (x + 10, y + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return output_img

def draw_points(img: np.ndarray, 
               points: List[Tuple[int, int]], 
               color: Tuple[int, int, int] = COLOR.RED,
               radius: int = 5,
               thickness: int = -1,
               labels: Optional[List[str]] = None,
               numbered: bool = False) -> np.ndarray:
    """
    Draw multiple points on an image.
    
    Args:
        img: Original BGR image
        points: List of (x, y) coordinates
        color: Color for points (BGR)
        radius: Radius of each point
        thickness: Thickness of the circles (-1 for filled)
        labels: Optional list of text labels for each point
        numbered: If True, points will be numbered (overrides labels)
        
    Returns:
        Image with all points drawn
    """
    output_img = img.copy()
    
    for i, point in enumerate(points):
        # Determine label
        point_label = None
        if numbered:
            point_label = str(i + 1)
        elif labels and i < len(labels):
            point_label = labels[i]
            
        # Draw the point
        output_img = draw_point(
            output_img, point, color, radius, thickness, point_label
        )
    
    return output_img

# =============================================
# Line Drawing Functions
# =============================================

def draw_line(img: np.ndarray, 
             pt1: Tuple[int, int], 
             pt2: Tuple[int, int], 
             color: Tuple[int, int, int] = COLOR.GREEN,
             thickness: int = 2,
             line_type: int = cv2.LINE_AA,
             dashed: bool = False,
             dash_length: int = 10,
             gap_length: int = 10) -> np.ndarray:
    """
    Draw a line between two points, with option for dashed line.
    
    Args:
        img: Original BGR image
        pt1: Starting point (x1, y1)
        pt2: Ending point (x2, y2)
        color: Line color (BGR)
        thickness: Line thickness
        line_type: OpenCV line type
        dashed: Whether to draw a dashed line
        dash_length: Length of each dash segment
        gap_length: Length of each gap segment
        
    Returns:
        Image with line drawn
    """
    output_img = img.copy()
    
    if not dashed:
        cv2.line(output_img, pt1, pt2, color, thickness, line_type)
        return output_img
    
    # Draw dashed line
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Calculate line length and angle
    dx, dy = x2 - x1, y2 - y1
    dist = np.sqrt(dx * dx + dy * dy)
    
    # Normalize direction vector
    if dist > 0:
        dx, dy = dx / dist, dy / dist
    else:
        return output_img  # Points are the same, nothing to draw
    
    # Calculate number of segments
    segment_length = dash_length + gap_length
    num_segments = int(np.ceil(dist / segment_length))
    
    # Draw dash segments
    for i in range(num_segments):
        start_dist = i * segment_length
        end_dist = start_dist + dash_length
        
        # Clip end distance to line length
        end_dist = min(end_dist, dist)
        
        # Calculate segment start and end points
        start_x = int(x1 + dx * start_dist)
        start_y = int(y1 + dy * start_dist)
        end_x = int(x1 + dx * end_dist)
        end_y = int(y1 + dy * end_dist)
        
        # Draw the dash segment
        cv2.line(output_img, (start_x, start_y), (end_x, end_y), 
                color, thickness, line_type)
    
    return output_img

def draw_grid_lines(img: np.ndarray, 
                   points: List[Tuple[int, int]], 
                   rows: int, 
                   cols: int,
                   color: Tuple[int, int, int] = COLOR.GREEN,
                   thickness: int = 2,
                   dashed: bool = False) -> np.ndarray:
    """
    Draw grid lines connecting points in a grid pattern.
    
    Args:
        img: Original BGR image
        points: List of (x, y) coordinates in row-major order
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        color: Line color (BGR)
        thickness: Line thickness
        dashed: Whether to use dashed lines
        
    Returns:
        Image with grid lines drawn
    """
    output_img = img.copy()
    
    # Draw horizontal lines
    for row in range(rows):
        for col in range(cols - 1):
            idx1 = row * cols + col
            idx2 = row * cols + col + 1
            
            if idx1 < len(points) and idx2 < len(points):
                output_img = draw_line(
                    output_img, points[idx1], points[idx2], 
                    color, thickness, dashed=dashed
                )
    
    # Draw vertical lines
    for col in range(cols):
        for row in range(rows - 1):
            idx1 = row * cols + col
            idx2 = (row + 1) * cols + col
            
            if idx1 < len(points) and idx2 < len(points):
                output_img = draw_line(
                    output_img, points[idx1], points[idx2], 
                    color, thickness, dashed=dashed
                )
    
    return output_img

def draw_custom_grid(img: np.ndarray, 
                    points: List[Tuple[int, int]],
                    horizontal_groups: List[List[int]],
                    vertical_groups: List[List[int]],
                    color: Tuple[int, int, int] = COLOR.GREEN,
                    thickness: int = 2,
                    dashed: bool = False) -> np.ndarray:
    """
    Draw grid lines connecting points according to custom grouping pattern.
    
    Args:
        img: Original BGR image
        points: List of (x, y) coordinates
        horizontal_groups: List of lists containing point indices for horizontal lines
        vertical_groups: List of lists containing point indices for vertical lines
        color: Line color (BGR)
        thickness: Line thickness
        dashed: Whether to use dashed lines
        
    Returns:
        Image with custom grid lines drawn
    """
    output_img = img.copy()
    
    # Draw horizontal lines
    for group in horizontal_groups:
        if all(idx < len(points) for idx in group):
            for i in range(len(group) - 1):
                idx1, idx2 = group[i], group[i + 1]
                output_img = draw_line(
                    output_img, points[idx1], points[idx2], 
                    color, thickness, dashed=dashed
                )
    
    # Draw vertical lines
    for group in vertical_groups:
        if all(idx < len(points) for idx in group):
            for i in range(len(group) - 1):
                idx1, idx2 = group[i], group[i + 1]
                output_img = draw_line(
                    output_img, points[idx1], points[idx2], 
                    color, thickness, dashed=dashed
                )
    
    return output_img

# =============================================
# Region of Interest (ROI) Functions
# =============================================

def draw_roi(img: np.ndarray, 
            roi: Optional[Union[Tuple[int, int, int, int], Dict]], 
            color: Tuple[int, int, int] = COLOR.GREEN,
            thickness: int = TRACKING.ROI_THICKNESS,
            show_center: bool = True,
            center_color: Tuple[int, int, int] = COLOR.RED,
            fill: bool = False,
            fill_color: Optional[Tuple[int, int, int, int]] = None,
            fill_alpha: float = 0.2) -> np.ndarray:
    """
    Draw ROI rectangle and optional center point on image.
    
    Args:
        img: Image to draw on
        roi: Either a (x, y, width, height) tuple or a dict with those keys
        color: Color for ROI rectangle (BGR)
        thickness: Line thickness
        show_center: Whether to show the center point
        center_color: Color for center point (BGR)
        fill: Whether to fill the ROI with semi-transparent color
        fill_color: Color for fill (BGRA), defaults to color with alpha if None
        fill_alpha: Alpha value for fill transparency (0-1)
        
    Returns:
        Image with ROI rectangle drawn
    """
    output_img = img.copy()
    
    if roi is None:
        return output_img
    
    # Convert dict format to tuple if needed
    if isinstance(roi, dict):
        if all(k in roi for k in ['x', 'y', 'width', 'height']):
            x, y = roi['x'], roi['y']
            w, h = roi['width'], roi['height']
        else:
            logging.warning(f"Invalid ROI dict format: {roi}")
            return output_img
    else:
        # Assume it's already a tuple (x, y, w, h)
        x, y, w, h = roi
    
    # Fill ROI if requested
    if fill:
        overlay = output_img.copy()
        if fill_color is None:
            # Use the specified color with alpha
            fill_color = (*color, int(255 * fill_alpha))
        
        cv2.rectangle(overlay, (x, y), (x + w, y + h), fill_color, -1)
        cv2.addWeighted(overlay, fill_alpha, output_img, 1 - fill_alpha, 0, output_img)
    
    # Draw rectangle outline
    cv2.rectangle(output_img, (x, y), (x + w, y + h), color, thickness)
    
    # Draw center point if requested
    if show_center:
        center_x, center_y = x + w // 2, y + h // 2
        cv2.circle(output_img, (center_x, center_y), ROI.CENTER_MARKER_SIZE, center_color, -1)
        cv2.line(output_img, (center_x - 10, center_y), (center_x + 10, center_y), center_color, 2)
        cv2.line(output_img, (center_x, center_y - 10), (center_x, center_y + 10), center_color, 2)
    
    return output_img

# =============================================
# Circle Drawing Functions 
# =============================================

def draw_circle(img: np.ndarray, 
               center: Tuple[int, int], 
               radius: int,
               color: Tuple[int, int, int] = COLOR.YELLOW,
               thickness: int = TRACKING.CIRCLE_THICKNESS,
               show_center: bool = True,
               center_color: Tuple[int, int, int] = COLOR.RED,
               label: Optional[str] = None) -> np.ndarray:
    """
    Draw a circle with optional center point and label.
    
    Args:
        img: Original BGR image
        center: (x, y) coordinates of circle center
        radius: Circle radius
        color: Circle color (BGR)
        thickness: Circle line thickness
        show_center: Whether to show center point
        center_color: Color for center point (BGR)
        label: Optional text label
        
    Returns:
        Image with circle drawn
    """
    output_img = img.copy()
    x, y = center
    
    # Draw circle
    cv2.circle(output_img, (x, y), radius, color, thickness)
    
    # Draw center point
    if show_center:
        cv2.circle(output_img, (x, y), ROI.CENTER_MARKER_SIZE // 2, center_color, -1)
    
    # Draw label if provided
    if label:
        cv2.putText(output_img, label, (x + radius, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return output_img

def draw_circles(img: np.ndarray,
                circles: List[Tuple[int, int, int]],
                main_color: Tuple[int, int, int] = TRACKING.MAIN_CIRCLE_COLOR,
                thickness: int = TRACKING.CIRCLE_THICKNESS,
                label_circles: bool = False) -> np.ndarray:
    """
    Draw multiple circles on an image.
    
    Args:
        img: Original BGR image
        circles: List of (x, y, radius) tuples
        main_color: Color for circles (BGR)
        thickness: Circle line thickness
        label_circles: Whether to number the circles
        
    Returns:
        Image with circles drawn
    """
    output_img = img.copy()
    
    if circles is None or len(circles) == 0:
        return output_img
    
    for i, circle in enumerate(circles):
        x, y, r = circle
        
        # Draw the circle
        label = str(i + 1) if label_circles else None
        output_img = draw_circle(
            output_img, (int(x), int(y)), int(r), 
            main_color, thickness, True, COLOR.RED, label
        )
    
    return output_img

# =============================================
# Advanced Visualization Functions
# =============================================

def draw_prediction(img: np.ndarray,
                   current_pos: Optional[Tuple[int, int]],
                   predicted_pos: Tuple[int, int],
                   arrow_color: Tuple[int, int, int] = TRACKING.PREDICTION_ARROW_COLOR,
                   thickness: int = TRACKING.PREDICTION_THICKNESS,
                   draw_uncertainty: bool = False,
                   uncertainty_radius: int = TRACKING.UNCERTAINTY_RADIUS) -> np.ndarray:
    """
    Draw prediction arrow between current and predicted position.
    
    Args:
        img: Original BGR image
        current_pos: Current position (x, y) or None
        predicted_pos: Predicted position (x, y)
        arrow_color: Color for arrow (BGR)
        thickness: Arrow line thickness
        draw_uncertainty: Whether to draw uncertainty circle
        uncertainty_radius: Radius for uncertainty circle
        
    Returns:
        Image with prediction arrow drawn
    """
    output_img = img.copy()
    
    # Draw arrow only if we have both positions
    if current_pos is not None:
        # Draw the arrow from current to predicted position
        cv2.arrowedLine(
            output_img,
            (int(current_pos[0]), int(current_pos[1])),
            (int(predicted_pos[0]), int(predicted_pos[1])),
            arrow_color,
            thickness,
            tipLength=0.3
        )
    
    # Draw the predicted position point
    cv2.circle(
        output_img,
        (int(predicted_pos[0]), int(predicted_pos[1])),
        5,
        arrow_color,
        -1
    )
    
    # Draw uncertainty circle if requested
    if draw_uncertainty:
        cv2.circle(
            output_img,
            (int(predicted_pos[0]), int(predicted_pos[1])),
            uncertainty_radius,
            arrow_color,
            1,
            cv2.LINE_AA
        )
    
    return output_img

def draw_trajectory(img: np.ndarray,
                   positions: List[Tuple[float, float]],
                   color: Tuple[int, int, int] = TRACKING.TRAJECTORY_COLOR,
                   thickness: int = TRACKING.TRAJECTORY_THICKNESS,
                   max_points: int = TRACKING.TRAJECTORY_MAX_POINTS) -> np.ndarray:
    """
    Draw trajectory from list of positions.
    
    Args:
        img: Original BGR image
        positions: List of (x, y) positions
        color: Color for trajectory line (BGR)
        thickness: Line thickness
        max_points: Maximum number of points to include in trajectory
        
    Returns:
        Image with trajectory drawn
    """
    output_img = img.copy()
    
    if not positions or len(positions) < 2:
        return output_img
    
    # Limit number of positions to avoid clutter
    if len(positions) > max_points:
        positions = positions[-max_points:]
    
    # Convert positions to integer coordinates
    pts = np.array([(int(x), int(y)) for x, y in positions])
    
    # Draw polyline for trajectory
    cv2.polylines(
        output_img,
        [pts],
        False,
        color,
        thickness,
        cv2.LINE_AA
    )
    
    return output_img

# =============================================
# HSV Visualization Functions
# =============================================

def get_dynamic_color(hsv_settings: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Generate a dynamic color based on HSV settings.
    Returns a BGR color that represents the average of the HSV range.
    
    Args:
        hsv_settings: Dictionary containing HSV ranges (h_min, h_max, s_min, s_max, v_min, v_max)
        
    Returns:
        BGR color tuple representing the HSV range
    """
    # Get HSV values with defaults
    h_min = hsv_settings.get('h_min', 0)
    h_max = hsv_settings.get('h_max', 179)
    s_min = hsv_settings.get('s_min', 0)
    s_max = hsv_settings.get('s_max', 255)
    v_min = hsv_settings.get('v_min', 0)
    v_max = hsv_settings.get('v_max', 255)
    
    # Calculate average values or handle wraparound for hue
    if h_min > h_max:  # Wraparound case (e.g., red color)
        # For red wraparound, use a more representative hue value
        h_avg = 0  # Red
    else:
        h_avg = (h_min + h_max) // 2
    
    s_avg = (s_min + s_max) // 2
    v_avg = (v_min + v_max) // 2
    
    # Create an HSV color array with the average values
    hsv_color = np.array([[[h_avg, s_avg, v_avg]]], dtype=np.uint8)
    
    # Convert HSV to BGR
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
    
    # Extract the BGR color values
    b, g, r = bgr_color[0, 0]
    
    # Ensure minimum brightness for visibility
    min_brightness = 100
    b = max(b, min_brightness)
    g = max(g, min_brightness)
    r = max(r, min_brightness)
    
    return (int(b), int(g), int(r))

def apply_mask_overlay(img: np.ndarray, 
                      mask: np.ndarray, 
                      alpha: float = 0.3, 
                      mask_color: Tuple[int, int, int] = COLOR.GREEN,
                      hsv_settings: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Apply HSV mask overlay on the original image.
    
    Args:
        img: Original BGR image
        mask: Binary mask (single channel)
        alpha: Transparency factor (0.0-1.0)
        mask_color: Default color for mask overlay (BGR)
        hsv_settings: Optional HSV settings dict to generate dynamic color
        
    Returns:
        Image with overlay applied
    """
    output_img = img.copy()
    if mask is None or not mask.any():
        return output_img
    
    # Use dynamic color based on HSV settings if provided
    if hsv_settings:
        color = get_dynamic_color(hsv_settings)
    else:
        color = mask_color
        
    # Create colored mask
    colored_mask = np.zeros_like(output_img)
    colored_mask[mask > 0] = color
    
    # Apply colored mask with alpha blending
    cv2.addWeighted(colored_mask, alpha, output_img, 1.0, 0, output_img)
    
    return output_img 