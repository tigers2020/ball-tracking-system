import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Union, List

def draw_prediction(img: np.ndarray,
                   current_pos: Optional[Tuple[int, int]],
                   predicted_pos: Optional[Tuple[int, int]],
                   arrow_color: Tuple[int, int, int] = (0, 255, 255),
                   thickness: int = 2,
                   draw_uncertainty: bool = False,
                   uncertainty_radius: int = 10,
                   uncertainty_color: Tuple[int, int, int] = (0, 165, 255)) -> np.ndarray:
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
        
    Returns:
        Image with prediction arrow drawn
    """
    output_img = img.copy()
    
    if current_pos is not None and predicted_pos is not None:
        # Draw arrow from current to predicted position
        cv2.arrowedLine(output_img, current_pos, predicted_pos, arrow_color, thickness, tipLength=0.3)
        
        # Draw uncertainty circle if requested
        if draw_uncertainty:
            cv2.circle(output_img, predicted_pos, uncertainty_radius, uncertainty_color, 1)
        
        logging.debug(f"Kalman prediction drawn: {current_pos} -> {predicted_pos}")
    
    return output_img


def draw_trajectory(img: np.ndarray,
                   positions: List[Tuple[int, int]],
                   color: Tuple[int, int, int] = (255, 255, 0),
                   thickness: int = 1,
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
    
    if not positions or len(positions) < 2:
        return output_img
    
    # Limit number of points if needed
    if max_points > 0 and len(positions) > max_points:
        positions = positions[-max_points:]
    
    # Convert to numpy array for polylines
    points = np.array(positions, dtype=np.int32).reshape((-1, 1, 2))
    
    # Draw polyline
    cv2.polylines(output_img, [points], False, color, thickness)
    
    return output_img 