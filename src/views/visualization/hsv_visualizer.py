import cv2
import numpy as np
import logging
from typing import Tuple, Optional

def apply_mask_overlay(img: np.ndarray, mask: np.ndarray, 
                      alpha: float = 0.3, 
                      mask_color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Apply HSV mask overlay on the original image
    
    Args:
        img: Original BGR image
        mask: Binary mask (single channel)
        alpha: Transparency factor (0.0-1.0)
        mask_color: Color for mask overlay (BGR)
        
    Returns:
        Image with overlay applied
    """
    output_img = img.copy()
    if mask is None or not mask.any():
        return output_img
        
    # Create colored mask
    colored_mask = np.zeros_like(output_img)
    colored_mask[mask > 0] = mask_color
    
    # Apply colored mask with alpha blending
    cv2.addWeighted(colored_mask, alpha, output_img, 1.0, 0, output_img)
    
    # Return the combined image
    return output_img


def draw_centroid(img: np.ndarray, 
                 centroid: Optional[Tuple[int, int]], 
                 radius: int = 7,
                 color: Tuple[int, int, int] = (255, 0, 255)) -> np.ndarray:
    """
    Draw centroid point on image
    
    Args:
        img: Image to draw on
        centroid: (x, y) coordinates of centroid or None
        radius: Size of centroid marker
        color: Color for centroid marker (BGR)
        
    Returns:
        Image with centroid drawn
    """
    output_img = img.copy()
    if centroid is not None:
        cv2.circle(output_img, centroid, radius, color, -1)
        logging.debug(f"HSV centroid drawn at {centroid}")
    return output_img 