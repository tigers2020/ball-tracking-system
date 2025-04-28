#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization utilities module.
This module contains functions for visualizing ball tracks, bounce events, and other data.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
import numpy.typing as npt

from src.models.ball_track import BallTrack
from src.models.bounce_event import BounceEvent


def draw_ball_tracks(image: npt.NDArray, ball_tracks: List[BallTrack], 
                     homography: Optional[npt.NDArray] = None, 
                     color_map: Dict[int, Tuple[int, int, int]] = None, 
                     thickness: int = 2, 
                     radius: int = 3) -> npt.NDArray:
    """
    Draw ball tracks on an image.
    
    Args:
        image: Input image to draw on
        ball_tracks: List of ball tracks to draw
        homography: Optional homography matrix to convert 3D positions to 2D image coordinates
        color_map: Optional dictionary mapping track IDs to colors
        thickness: Line thickness
        radius: Point radius
        
    Returns:
        Image with ball tracks drawn
    """
    # Make a copy of the image to avoid modifying the original
    result_image = image.copy()
    
    # Create default color map if not provided
    if color_map is None:
        color_map = {}
    
    # Define a list of colors for tracks
    default_colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
    ]
    
    # Draw each track
    for track_idx, track in enumerate(ball_tracks):
        # Get track color
        if track.id in color_map:
            color = color_map[track.id]
        else:
            color = default_colors[track_idx % len(default_colors)]
            color_map[track.id] = color
        
        # Get track points
        track_points = track.track_points
        
        # Skip if no points
        if not track_points:
            continue
        
        # Draw track
        prev_point = None
        for point in track_points:
            # Get point position
            position = point.position
            
            # Apply homography if provided
            if homography is not None:
                # Convert 3D position to homogeneous coordinates
                homogeneous_pos = np.array([position[0], position[1], 1.0])
                
                # Apply homography
                transformed_pos = np.dot(homography, homogeneous_pos)
                
                # Convert back to 2D coordinates
                transformed_pos = transformed_pos / transformed_pos[2]
                
                # Extract x, y coordinates
                x, y = int(transformed_pos[0]), int(transformed_pos[1])
            else:
                # Use only x, y coordinates
                x, y = int(position[0]), int(position[1])
            
            # Draw point
            cv2.circle(result_image, (x, y), radius, color, -1)
            
            # Draw line connecting to previous point
            if prev_point is not None:
                cv2.line(result_image, prev_point, (x, y), color, thickness)
            
            # Update previous point
            prev_point = (x, y)
    
    return result_image


def draw_bounce_events(image: npt.NDArray, bounce_events: List[BounceEvent], 
                       homography: Optional[npt.NDArray] = None, 
                       color: Tuple[int, int, int] = (0, 0, 255), 
                       radius: int = 5, 
                       thickness: int = 2, 
                       draw_labels: bool = True,
                       label_font_scale: float = 0.5,
                       label_thickness: int = 1) -> npt.NDArray:
    """
    Draw bounce events on an image.
    
    Args:
        image: Input image to draw on
        bounce_events: List of bounce events to draw
        homography: Optional homography matrix to convert 3D positions to 2D image coordinates
        color: Color to draw bounce events
        radius: Circle radius
        thickness: Circle thickness
        draw_labels: Whether to draw frame indices as labels
        label_font_scale: Font scale for labels
        label_thickness: Thickness of label text
        
    Returns:
        Image with bounce events drawn
    """
    # Make a copy of the image to avoid modifying the original
    result_image = image.copy()
    
    # Draw each bounce event
    for bounce in bounce_events:
        # Get bounce position
        position = bounce.position
        
        # Apply homography if provided
        if homography is not None:
            # Convert 3D position to homogeneous coordinates
            homogeneous_pos = np.array([position[0], position[1], 1.0])
            
            # Apply homography
            transformed_pos = np.dot(homography, homogeneous_pos)
            
            # Convert back to 2D coordinates
            transformed_pos = transformed_pos / transformed_pos[2]
            
            # Extract x, y coordinates
            x, y = int(transformed_pos[0]), int(transformed_pos[1])
        else:
            # Use only x, y coordinates
            x, y = int(position[0]), int(position[1])
        
        # Draw bounce event - a circle with an X in the middle
        cv2.circle(result_image, (x, y), radius, color, thickness)
        cv2.line(result_image, (x - radius, y - radius), (x + radius, y + radius), color, thickness)
        cv2.line(result_image, (x - radius, y + radius), (x + radius, y - radius), color, thickness)
        
        # Draw label if enabled
        if draw_labels:
            # Create label text
            label = f"{bounce.frame_index}"
            
            # Get text size
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, label_thickness)
            
            # Calculate text position
            text_x = x - text_size[0] // 2
            text_y = y - radius - 5
            
            # Draw text
            cv2.putText(result_image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        label_font_scale, color, label_thickness)
    
    return result_image


def plot_ball_trajectory(ball_track: BallTrack, bounce_events: List[BounceEvent] = None, 
                         view: str = '3d', figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot a ball trajectory in 2D or 3D.
    
    Args:
        ball_track: Ball track to plot
        bounce_events: Optional list of bounce events to mark
        view: View type ('3d', 'xy', 'xz', or 'yz')
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Extract track points
    positions = np.array([point.position for point in ball_track.track_points])
    timestamps = np.array([point.timestamp for point in ball_track.track_points])
    
    # Normalize time for colormap
    norm_time = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
    
    # Extract bounce positions if provided
    bounce_positions = []
    if bounce_events:
        for bounce in bounce_events:
            # Check if bounce is associated with this track
            if bounce.metadata and 'track_id' in bounce.metadata:
                if bounce.metadata['track_id'] == ball_track.id:
                    bounce_positions.append(bounce.position)
    
    bounce_positions = np.array(bounce_positions) if bounce_positions else None
    
    # Plot based on view type
    if view == '3d':
        # 3D plot
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory
        scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                             c=norm_time, cmap='viridis', s=20)
        
        # Plot line connecting points
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'gray', alpha=0.5)
        
        # Plot bounce events if available
        if bounce_positions is not None and len(bounce_positions) > 0:
            ax.scatter(bounce_positions[:, 0], bounce_positions[:, 1], bounce_positions[:, 2], 
                       color='red', s=100, marker='X', label='Bounces')
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
    else:
        # 2D plot
        ax = fig.add_subplot(111)
        
        if view == 'xy':
            # Top-down view
            scatter = ax.scatter(positions[:, 0], positions[:, 1], c=norm_time, cmap='viridis', s=20)
            ax.plot(positions[:, 0], positions[:, 1], 'gray', alpha=0.5)
            
            # Plot bounce events if available
            if bounce_positions is not None and len(bounce_positions) > 0:
                ax.scatter(bounce_positions[:, 0], bounce_positions[:, 1], 
                           color='red', s=100, marker='X', label='Bounces')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
        elif view == 'xz':
            # Side view (X-Z)
            scatter = ax.scatter(positions[:, 0], positions[:, 2], c=norm_time, cmap='viridis', s=20)
            ax.plot(positions[:, 0], positions[:, 2], 'gray', alpha=0.5)
            
            # Plot bounce events if available
            if bounce_positions is not None and len(bounce_positions) > 0:
                ax.scatter(bounce_positions[:, 0], bounce_positions[:, 2], 
                           color='red', s=100, marker='X', label='Bounces')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            
        elif view == 'yz':
            # Side view (Y-Z)
            scatter = ax.scatter(positions[:, 1], positions[:, 2], c=norm_time, cmap='viridis', s=20)
            ax.plot(positions[:, 1], positions[:, 2], 'gray', alpha=0.5)
            
            # Plot bounce events if available
            if bounce_positions is not None and len(bounce_positions) > 0:
                ax.scatter(bounce_positions[:, 1], bounce_positions[:, 2], 
                           color='red', s=100, marker='X', label='Bounces')
            
            ax.set_xlabel('Y')
            ax.set_ylabel('Z')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Time')
    
    # Add title
    plt.title(f'Ball Trajectory (Track {ball_track.id})')
    
    # Add legend if bounce events were plotted
    if bounce_positions is not None and len(bounce_positions) > 0:
        plt.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def plot_bounce_distribution(bounce_events: List[BounceEvent], 
                             court_bounds: Dict[str, Any] = None,
                             figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot the distribution of bounce events on the court.
    
    Args:
        bounce_events: List of bounce events to plot
        court_bounds: Optional court bounds for drawing the court outline
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # Extract bounce positions
    positions = np.array([bounce.position[:2] for bounce in bounce_events])
    
    # Plot bounce positions
    plt.scatter(positions[:, 0], positions[:, 1], color='red', s=50, alpha=0.7)
    
    # Draw court outline if bounds are provided
    if court_bounds is not None:
        # Get court dimensions
        min_x, max_x = court_bounds['x_range']
        min_y, max_y = court_bounds['y_range']
        net_y = court_bounds.get('net_y', (min_y + max_y) / 2)
        
        # Draw court outline
        court_outline = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, 
                                     edgecolor='black', facecolor='none', linewidth=2)
        ax.add_patch(court_outline)
        
        # Draw net line
        plt.axhline(y=net_y, color='black', linestyle='--', linewidth=1)
        
        # Draw center mark
        center_x = (min_x + max_x) / 2
        plt.axvline(x=center_x, color='black', linestyle='--', linewidth=1)
    
    # Set labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Bounce Distribution')
    
    # Equal aspect ratio
    plt.axis('equal')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig 