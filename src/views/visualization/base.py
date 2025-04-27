#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base Visualization Interface.
This module defines the common interface for all visualizers in the application.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Optional, Union
import numpy as np


class IVisualizer(ABC):
    """
    Abstract base class for visualization components.
    All visualization implementations must conform to this interface.
    """
    
    @abstractmethod
    def draw_point(self, img: np.ndarray, 
                  point: Tuple[int, int], 
                  color: Optional[Tuple[int, int, int]] = None,
                  radius: Optional[int] = None,
                  thickness: Optional[int] = None,
                  label: Optional[str] = None) -> np.ndarray:
        """
        Draw a point (marker) on an image.
        
        Args:
            img: Original image
            point: (x, y) coordinates
            color: Optional color override
            radius: Optional radius override
            thickness: Optional thickness override
            label: Optional text label
            
        Returns:
            Image with point drawn
        """
        pass
    
    @abstractmethod
    def draw_points(self, img: np.ndarray, 
                   points: List[Tuple[int, int]], 
                   color: Optional[Tuple[int, int, int]] = None,
                   radius: Optional[int] = None,
                   thickness: Optional[int] = None,
                   labels: Optional[List[str]] = None,
                   numbered: bool = False) -> np.ndarray:
        """
        Draw multiple points on an image.
        
        Args:
            img: Original image
            points: List of (x, y) coordinates
            color: Optional color override
            radius: Optional radius override
            thickness: Optional thickness override
            labels: Optional list of text labels
            numbered: Whether to number the points
            
        Returns:
            Image with points drawn
        """
        pass
    
    @abstractmethod
    def draw_line(self, img: np.ndarray, 
                 pt1: Tuple[int, int], 
                 pt2: Tuple[int, int], 
                 color: Optional[Tuple[int, int, int]] = None,
                 thickness: Optional[int] = None,
                 dashed: bool = False) -> np.ndarray:
        """
        Draw a line between two points.
        
        Args:
            img: Original image
            pt1: Starting point
            pt2: Ending point
            color: Optional color override
            thickness: Optional thickness override
            dashed: Whether to draw a dashed line
            
        Returns:
            Image with line drawn
        """
        pass
    
    @abstractmethod
    def draw_grid_lines(self, img: np.ndarray, 
                       points: List[Tuple[int, int]], 
                       rows: int, 
                       cols: int,
                       color: Optional[Tuple[int, int, int]] = None,
                       thickness: Optional[int] = None,
                       dashed: bool = False) -> np.ndarray:
        """
        Draw grid lines connecting points in a grid pattern.
        
        Args:
            img: Original image
            points: List of (x, y) coordinates in row-major order
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            color: Optional color override
            thickness: Optional thickness override
            dashed: Whether to use dashed lines
            
        Returns:
            Image with grid lines drawn
        """
        pass
    
    @abstractmethod
    def draw_roi(self, img: np.ndarray, 
                roi: Union[Tuple[int, int, int, int], Dict[str, int]], 
                color: Optional[Tuple[int, int, int]] = None,
                thickness: Optional[int] = None,
                show_center: bool = True,
                fill: bool = False,
                fill_alpha: Optional[float] = None) -> np.ndarray:
        """
        Draw ROI rectangle on image.
        
        Args:
            img: Original image
            roi: Either a (x, y, width, height) tuple or a dict with those keys
            color: Optional color override
            thickness: Optional thickness override
            show_center: Whether to show the center point
            fill: Whether to fill the ROI with semi-transparent color
            fill_alpha: Alpha value for fill transparency
            
        Returns:
            Image with ROI rectangle drawn
        """
        pass
    
    @abstractmethod
    def draw_circle(self, img: np.ndarray, 
                   center: Tuple[int, int], 
                   radius: int,
                   color: Optional[Tuple[int, int, int]] = None,
                   thickness: Optional[int] = None,
                   show_center: bool = True,
                   label: Optional[str] = None) -> np.ndarray:
        """
        Draw a circle with optional center point and label.
        
        Args:
            img: Original image
            center: (x, y) coordinates of circle center
            radius: Circle radius
            color: Optional color override
            thickness: Optional thickness override
            show_center: Whether to show center point
            label: Optional text label
            
        Returns:
            Image with circle drawn
        """
        pass
    
    @abstractmethod
    def draw_circles(self, img: np.ndarray,
                    circles: List[Tuple[int, int, int]],
                    color: Optional[Tuple[int, int, int]] = None,
                    thickness: Optional[int] = None,
                    label_circles: bool = False) -> np.ndarray:
        """
        Draw multiple circles on an image.
        
        Args:
            img: Original image
            circles: List of (x, y, radius) tuples
            color: Optional color override
            thickness: Optional thickness override
            label_circles: Whether to number the circles
            
        Returns:
            Image with circles drawn
        """
        pass
    
    @abstractmethod
    def draw_prediction(self, img: np.ndarray,
                       current_pos: Optional[Tuple[int, int]],
                       predicted_pos: Tuple[int, int],
                       color: Optional[Tuple[int, int, int]] = None,
                       thickness: Optional[int] = None,
                       draw_uncertainty: bool = False,
                       uncertainty_radius: Optional[int] = None) -> np.ndarray:
        """
        Draw prediction arrow between current and predicted position.
        
        Args:
            img: Original image
            current_pos: Current position (x, y) or None
            predicted_pos: Predicted position (x, y)
            color: Optional color override
            thickness: Optional thickness override
            draw_uncertainty: Whether to draw uncertainty circle
            uncertainty_radius: Optional radius override for uncertainty circle
            
        Returns:
            Image with prediction arrow drawn
        """
        pass
    
    @abstractmethod
    def draw_trajectory(self, img: np.ndarray,
                       positions: List[Tuple[float, float]],
                       color: Optional[Tuple[int, int, int]] = None,
                       thickness: Optional[int] = None,
                       max_points: Optional[int] = None) -> np.ndarray:
        """
        Draw trajectory from list of positions.
        
        Args:
            img: Original image
            positions: List of (x, y) positions
            color: Optional color override
            thickness: Optional thickness override
            max_points: Maximum number of points to include in trajectory
            
        Returns:
            Image with trajectory drawn
        """
        pass
    
    @abstractmethod
    def apply_mask_overlay(self, img: np.ndarray, 
                          mask: np.ndarray, 
                          alpha: Optional[float] = None, 
                          mask_color: Optional[Tuple[int, int, int]] = None,
                          hsv_settings: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Apply mask overlay on the original image.
        
        Args:
            img: Original image
            mask: Binary mask (single channel)
            alpha: Optional transparency factor override
            mask_color: Optional color override for mask
            hsv_settings: Optional HSV settings dict to generate dynamic color
            
        Returns:
            Image with overlay applied
        """
        pass 