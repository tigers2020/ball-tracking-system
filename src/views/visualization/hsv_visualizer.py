import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any

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
    
    logging.debug(f"Generated dynamic color BGR({b},{g},{r}) from HSV({h_avg},{s_avg},{v_avg})")
    
    return (int(b), int(g), int(r))

def apply_mask_overlay(img: np.ndarray, mask: np.ndarray, 
                      alpha: float = 0.3, 
                      mask_color: Tuple[int, int, int] = (0, 255, 0),
                      hsv_settings: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Apply HSV mask overlay on the original image
    
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
        logging.debug(f"Using dynamic mask color: {color}")
    else:
        color = mask_color
        
    # Create colored mask
    colored_mask = np.zeros_like(output_img)
    colored_mask[mask > 0] = color
    
    # Apply colored mask with alpha blending
    cv2.addWeighted(colored_mask, alpha, output_img, 1.0, 0, output_img)
    
    # Return the combined image
    return output_img

def draw_centroid(img: np.ndarray, 
                 point: Tuple[int, int], 
                 color: Tuple[int, int, int] = (255, 0, 255),
                 radius: int = 5,
                 thickness: int = -1,
                 cross_size: int = 10) -> np.ndarray:
    """
    Draw centroid point with optional cross on image
    
    Args:
        img: Original BGR image
        point: (x, y) coordinates of the centroid
        color: Color for centroid marker (BGR)
        radius: Radius of the centroid circle
        thickness: Thickness of the circle (-1 for filled)
        cross_size: Size of the cross marker
        
    Returns:
        Image with centroid drawn
    """
    output_img = img.copy()
    x, y = point
    
    # Draw filled circle for the centroid
    cv2.circle(output_img, (x, y), radius, color, thickness)
    
    # Draw cross markers for better visibility
    cv2.line(output_img, (x - cross_size, y), (x + cross_size, y), color, 2)
    cv2.line(output_img, (x, y - cross_size), (x, y + cross_size), color, 2)
    
    return output_img 