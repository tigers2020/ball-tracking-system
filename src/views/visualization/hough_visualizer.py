import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Union, Dict, Any

def draw_circles(img: np.ndarray,
                circles: Optional[Union[List[Tuple[int, int, int]], List[Dict[str, Any]]]],
                main_color: Tuple[int, int, int] = (0, 255, 0),
                secondary_color: Tuple[int, int, int] = (255, 0, 0),
                center_color: Tuple[int, int, int] = (0, 0, 255),
                thickness: int = 3,
                label_circles: bool = False) -> np.ndarray:
    """
    Draw detected Hough circles on image
    
    Args:
        img: Image to draw on
        circles: List of (x, y, radius) tuples or dicts with 'x', 'y', 'r' keys, or None
        main_color: Color for main (first) circle
        secondary_color: Color for other circles
        center_color: Color for circle centers
        thickness: Line thickness
        label_circles: Whether to add number labels to circles
        
    Returns:
        Image with circles drawn
    """
    output_img = img.copy()
    
    if not circles:
        return output_img
    
    for i, circle in enumerate(circles):
        # Extract coordinates based on whether circle is a tuple or dict
        if isinstance(circle, dict):
            if all(k in circle for k in ['x', 'y', 'r']):
                x, y, r = circle['x'], circle['y'], circle['r']
            else:
                logging.warning(f"Invalid circle dict format: {circle}")
                continue
        else:
            # Assume it's a tuple (x, y, r)
            x, y, r = circle
        
        # Draw circle outline
        circle_color = main_color if i == 0 else secondary_color
        cv2.circle(output_img, (x, y), r, circle_color, thickness)
        
        # Draw center point
        cv2.circle(output_img, (x, y), 4, center_color, -1)
        
        # Add label if requested
        if label_circles:
            cv2.putText(output_img, str(i+1), (x+r//2, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, circle_color, 2, cv2.LINE_AA)
    
    return output_img 