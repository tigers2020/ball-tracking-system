import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Union

def draw_roi(img: np.ndarray, 
            roi: Optional[Union[Tuple[int, int, int, int], Dict]], 
            color: Tuple[int, int, int] = (255, 255, 0),
            thickness: int = 2,
            show_center: bool = True,
            center_color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
    """
    Draw ROI rectangle and optional center point on image
    
    Args:
        img: Image to draw on
        roi: Either a (x, y, width, height) tuple or a dict with those keys
        color: Color for ROI rectangle (BGR)
        thickness: Line thickness
        show_center: Whether to show the center point
        center_color: Color for center point (BGR)
        
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
    
    # Draw rectangle
    cv2.rectangle(output_img, (x, y), (x + w, y + h), color, thickness)
    
    # Draw center point if requested
    if show_center:
        center_x, center_y = x + w // 2, y + h // 2
        cv2.circle(output_img, (center_x, center_y), 3, center_color, -1)
        cv2.line(output_img, (center_x - 5, center_y), (center_x + 5, center_y), center_color, 1)
        cv2.line(output_img, (center_x, center_y - 5), (center_x, center_y + 5), center_color, 1)
    
    return output_img 