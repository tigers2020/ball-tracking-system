import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Union, List

def draw_prediction(img: np.ndarray,
                   current_pos: Optional[Tuple[int, int]],
                   predicted_pos: Optional[Tuple[int, int]],
                   arrow_color: Tuple[int, int, int] = (0, 255, 255),
                   thickness: int = 4,
                   draw_uncertainty: bool = False,
                   uncertainty_radius: int = 15,
                   uncertainty_color: Tuple[int, int, int] = (0, 165, 255),
                   world_pos: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Draw Kalman prediction arrow on image
    
    Args:
        img: Image to draw on
        current_pos: Current position (x, y) or None
        predicted_pos: Predicted position (x, y) or None
        arrow_color: Color for prediction arrow (BGR)
        thickness: Line thickness
        draw_uncertainty: Whether to draw uncertainty circle
        uncertainty_radius: Radius of uncertainty circle
        uncertainty_color: Color for uncertainty circle
        world_pos: Optional 3D world position for height clamping
        
    Returns:
        Image with prediction arrow drawn
    """
    output_img = img.copy()
    frame_h, frame_w = output_img.shape[:2]
    
    # Emergency height clamping for visibility
    if world_pos is not None:
        x_w, y_w, z_w = world_pos
        # 응급 처치 - Z축 클램핑
        MAX_Z = 10.0        # m (보일 수 있는 최대 높이)
        Z_OFFSET = -20.0    # "20 m 깎기" 옵션
        if z_w > MAX_Z:
            z_w += Z_OFFSET   # 20 m 감소
            if z_w < 0:       # 음수가 되면 그리지 않음
                return output_img  # 프레임 skip
    
    if current_pos is not None and predicted_pos is not None:
        # Clamp pixel coordinates to image dimensions
        current_x = min(max(0, current_pos[0]), frame_w-1)
        current_y = min(max(0, current_pos[1]), frame_h-1)
        predicted_x = min(max(0, predicted_pos[0]), frame_w-1)
        predicted_y = min(max(0, predicted_pos[1]), frame_h-1)
        
        clamped_current = (current_x, current_y)
        clamped_predicted = (predicted_x, predicted_y)
        
        # Draw arrow from current to predicted position
        cv2.arrowedLine(output_img, clamped_current, clamped_predicted, arrow_color, thickness, tipLength=0.3)
        
        # Draw uncertainty circle if requested
        if draw_uncertainty:
            cv2.circle(output_img, clamped_predicted, uncertainty_radius, uncertainty_color, 2)
        
        logging.debug(f"Kalman prediction drawn: {clamped_current} -> {clamped_predicted}")
    
    return output_img


def draw_trajectory(img: np.ndarray,
                   positions: List[Tuple[int, int]],
                   color: Tuple[int, int, int] = (255, 255, 0),
                   thickness: int = 3,
                   max_points: int = 10) -> np.ndarray:
    """
    Draw trajectory line from position history
    
    Args:
        img: Image to draw on
        positions: List of (x, y) positions
        color: Color for trajectory line (BGR)
        thickness: Line thickness
        max_points: Maximum number of points to include
        
    Returns:
        Image with trajectory drawn
    """
    output_img = img.copy()
    frame_h, frame_w = output_img.shape[:2]
    
    if not positions or len(positions) < 2:
        return output_img
    
    # Limit number of points if needed
    if max_points > 0 and len(positions) > max_points:
        positions = positions[-max_points:]
    
    # Clamp all positions to image dimensions
    clamped_positions = []
    for pos in positions:
        x = min(max(0, pos[0]), frame_w-1)
        y = min(max(0, pos[1]), frame_h-1)
        clamped_positions.append((x, y))
    
    # Convert to numpy array for polylines
    points = np.array(clamped_positions, dtype=np.int32).reshape((-1, 1, 2))
    
    # Draw polyline
    cv2.polylines(output_img, [points], False, color, thickness)
    
    return output_img