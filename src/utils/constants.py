#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Universal Constants Module for the Stereo Image Player application.
This file centralizes all constants used across the application following MVC and SRP principles.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple, Dict, Any


# Get the absolute path to the root directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


# =============================================
# OpenCV Image Processing Constants
# =============================================

@dataclass
class HOUGH:
    """Hough Circle Transform parameters"""
    dp: float = 1.3
    min_dist: int = 112
    param1: int = 64
    param2: int = 31
    min_radius: int = 13
    max_radius: int = 25
    adaptive: bool = False


@dataclass
class HSV:
    """HSV Thresholding parameters"""
    # Hue Range (0-179 in OpenCV)
    h_min: int = 0 
    h_max: int = 10
    # Saturation Range (0-255 in OpenCV)
    s_min: int = 100
    s_max: int = 255
    # Value Range (0-255 in OpenCV)
    v_min: int = 100
    v_max: int = 255
    # Morphological operations
    blur_size: int = 5
    morph_iterations: int = 2
    dilation_iterations: int = 1


@dataclass
class MORPHOLOGY:
    """Morphological operations parameters"""
    kernel_size: Tuple[int, int] = (5, 5)
    iterations: int = 2


@dataclass
class KALMAN:
    """Kalman Filter parameters"""
    process_noise: float = 0.03
    measurement_noise: float = 1.0
    max_lost_frames: int = 20
    dynamic_process_noise: bool = True
    adaptive_measurement_noise: bool = True


@dataclass
class COLOR:
    """Color constants in BGR format (OpenCV standard)"""
    # Primary colors
    RED: Tuple[int, int, int] = (0, 0, 255)
    GREEN: Tuple[int, int, int] = (0, 255, 0)
    BLUE: Tuple[int, int, int] = (255, 0, 0)
    # Secondary colors
    YELLOW: Tuple[int, int, int] = (0, 255, 255)
    MAGENTA: Tuple[int, int, int] = (255, 0, 255)
    CYAN: Tuple[int, int, int] = (255, 255, 0)
    # Grayscale
    WHITE: Tuple[int, int, int] = (255, 255, 255)
    BLACK: Tuple[int, int, int] = (0, 0, 0)
    GRAY: Tuple[int, int, int] = (128, 128, 128)
    LIGHT_GRAY: Tuple[int, int, int] = (192, 192, 192)
    DARK_GRAY: Tuple[int, int, int] = (64, 64, 64)
    # Alpha versions (BGRA)
    RED_ALPHA: Tuple[int, int, int, int] = (0, 0, 255, 64)
    GREEN_ALPHA: Tuple[int, int, int, int] = (0, 255, 0, 64)
    BLUE_ALPHA: Tuple[int, int, int, int] = (255, 0, 0, 64)


# =============================================
# UI Constants
# =============================================

@dataclass
class WINDOW:
    """Window size constants"""
    DEFAULT_WIDTH: int = 1280
    DEFAULT_HEIGHT: int = 720
    MIN_WIDTH: int = 800
    MIN_HEIGHT: int = 600


@dataclass
class LAYOUT:
    """Layout constants"""
    MARGIN: int = 10
    SPACING: int = 5
    BUTTON_HEIGHT: int = 30
    BUTTON_WIDTH: int = 120
    SLIDER_HEIGHT: int = 20
    TOOLBAR_HEIGHT: int = 40
    IMAGE_VIEW_RATIO: float = 0.45  # Each image view takes 45% of the width
    CONTROLS_HEIGHT: int = 50


@dataclass
class TIMING:
    """Timing constants"""
    DEFAULT_FPS: int = 30
    MIN_FPS: int = 1
    MAX_FPS: int = 120
    DEFAULT_PLAYBACK_INTERVAL: int = 1000 // DEFAULT_FPS  # in milliseconds


# =============================================
# Region of Interest (ROI) Constants
# =============================================

@dataclass
class ROI:
    """ROI constants"""
    DEFAULT_WIDTH: int = 100
    DEFAULT_HEIGHT: int = 100
    MIN_SIZE: int = 10
    MAX_SIZE: int = 500
    BORDER_THICKNESS: int = 2
    BORDER_COLOR: Tuple[int, int, int] = COLOR.RED
    FILL_COLOR: Tuple[int, int, int, int] = COLOR.RED_ALPHA
    CENTER_MARKER_SIZE: int = 5
    CENTER_MARKER_COLOR: Tuple[int, int, int] = COLOR.WHITE
    ENABLED: bool = True
    AUTO_CENTER: bool = True


# =============================================
# File and Path Constants
# =============================================

class FILE_DIALOG:
    """File dialog constants"""
    DIALOG_CAPTION: str = "Select Stereo Images Folder"
    DIALOG_FILTER: str = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"


class PATHS:
    """Path constants"""
    # Directory paths
    ICONS_DIR: str = os.path.join(ROOT_DIR, "src", "resources", "icons")
    
    # Icon files
    PLAY: str = os.path.join(ICONS_DIR, "play.svg")
    PAUSE: str = os.path.join(ICONS_DIR, "pause.svg")
    STOP: str = os.path.join(ICONS_DIR, "stop.svg")
    NEXT: str = os.path.join(ICONS_DIR, "next_frame.svg")
    PREV: str = os.path.join(ICONS_DIR, "prev_frame.svg")
    SETTINGS: str = os.path.join(ICONS_DIR, "check.svg")  # Using check.svg as a substitute
    OPEN: str = os.path.join(ICONS_DIR, "check.svg")      # Using check.svg as a substitute
    EXIT: str = os.path.join(ICONS_DIR, "stop.svg")


# =============================================
# Message Constants
# =============================================

class MESSAGES:
    """Message strings for dialogs and status bar"""
    LOADING_FOLDER: str = "Loading images from folder..."
    GENERATING_XML: str = "Generating frames info XML..."
    MOVING_FILES: str = "Moving files to appropriate folders..."
    NO_IMAGES_FOUND: str = "No stereo images found in the selected folder"
    ERROR_LOADING_XML: str = "Error loading XML file"
    ERROR_PARSING_XML: str = "Error parsing XML file"
    ERROR_LOADING_IMAGE: str = "Error loading image"
    ERROR_SAVING_XML: str = "Error saving XML file"
    PLAYBACK_STARTED: str = "Playback started"
    PLAYBACK_PAUSED: str = "Playback paused"
    PLAYBACK_STOPPED: str = "Playback stopped"
    READY: str = "Ready"


# =============================================
# XML Constants
# =============================================

class XML:
    """XML constants"""
    FRAMES_INFO_FILENAME: str = "frames_info.xml"
    ROOT_ELEMENT: str = "stereo_frames"
    FRAME_ELEMENT: str = "frame"
    INDEX_ATTR: str = "index"
    LEFT_IMAGE_ELEMENT: str = "left_image"
    RIGHT_IMAGE_ELEMENT: str = "right_image"
    PATH_ATTR: str = "path" 

# =============================================
# Tennis Court Constants
# =============================================

@dataclass
class COURT:
    """Tennis court dimensions and properties (in meters)"""
    LENGTH: float = 23.77  # Total court length
    WIDTH: float = 8.23  # Total court width
    WIDTH_HALF: float = 8.23 / 2  # Half court width
    LENGTH_HALF: float = 23.77 / 2  # Half court length
    HALF_WIDTH: float = 11.885  # Half court width
    NET_HEIGHT: float = 0.914  # Height of the net
    NET_Y: float = 23.77 / 2  # Net position (y-coordinate)
    SERVICE_LINE_Y: float = 6.40  # Distance from baseline to service line
    CENTER_SERVICE_LINE_WIDTH: float = 0.05  # Width of center service line
    BASELINE_WIDTH: float = 0.10  # Width of the baseline
    SIDELINE_WIDTH: float = 0.05  # Width of the sideline
    SINGLES_COURT_HALF_WIDTH: float = 4.115  # Half width of singles court
    INSIDE_EPS: float = 0.05  # 5 cm margin for IN/OUT decisions
    BOUNDARY_MARGIN: float = 5.0  # Margin outside court bounds for valid ball position (m)

# =============================================
# Game Analysis Constants
# =============================================

@dataclass
class ANALYSIS:
    """Game analysis parameters"""
    # Bounce detection
    MIN_BOUNCE_VELOCITY_CHANGE: float = 0.5  # Minimum velocity change for bounce detection (m/s)
    MAX_GROUND_HEIGHT: float = 0.03  # Maximum height to consider as ground contact (m)
    MIN_FRAMES_BETWEEN_BOUNCES: int = 3  # Minimum frames between consecutive bounces
    
    # Ball height constraints
    MAX_VALID_HEIGHT: float = 15.0  # Maximum valid height for ball detection (m)
    MIN_VALID_HEIGHT: float = -0.05  # Minimum valid height, slight negative allowed for measurement error (m)
    
    # Ball physics
    MAX_BALL_SPEED: float = 50.0  # Maximum expected tennis ball speed (m/s)
    REASONABLE_DISPLACEMENT: float = 2.0  # Reasonable displacement per frame (m)
    MAX_BALL_SPEED_KMH: float = 180.0  # ~50 m/s in km/h
    DEFAULT_FPS: float = 60.0  # Default frame rate
    MIN_FRAME_TIME: float = 0.001  # Minimum frame time to prevent division by zero
    
    # Trajectory analysis
    NET_CROSSING_FRAMES_THRESHOLD: int = 30  # Minimum frames between net crossing detections
    TRAJECTORY_UPDATE_INTERVAL: int = 3  # Update trajectory visualization every N frames
    TRAJECTORY_DISPLAY_POINTS: int = 30  # Number of points to display in trajectory
    
    # Kalman filter 3D
    PROCESS_NOISE_POS: float = 0.01  # Process noise for position
    PROCESS_NOISE_VEL: float = 0.1   # Process noise for velocity 
    MEASUREMENT_NOISE: float = 0.05  # Measurement noise (m)
    VELOCITY_DECAY: float = 0.99  # Velocity decay coefficient
    GRAVITY: float = 9.81  # Gravitational acceleration (m/s²)
    MIN_UPDATES_REQUIRED: int = 5  # Minimum updates before allowing filter reset
    RESET_THRESHOLD: float = 10.0  # Distance threshold for filter reset (m)
    MAX_HISTORY_LENGTH: int = 120  # Maximum history length (frames)
    BOUNCE_MARKER_SIZE: int = 8  # Size of bounce markers
    BOUNCE_IN_COLOR: Tuple[int, int, int] = COLOR.GREEN  # Color for IN bounces
    BOUNCE_OUT_COLOR: Tuple[int, int, int] = COLOR.RED   # Color for OUT bounces
    TRAJECTORY_COLOR: Tuple[int, int, int] = COLOR.BLUE  # Color for ball trajectory
    COURT_LINE_COLOR: Tuple[int, int, int] = COLOR.WHITE  # Color for court lines
    COURT_LINE_THICKNESS: int = 2  # Thickness of court lines
    
    # Height confidence calculation
    HEIGHT_CONFIDENCE_FACTOR: float = 5.0  # Divisor for height confidence calculation
    MIN_HEIGHT_CONFIDENCE: float = 0.2  # Minimum confidence for height validation
    
    # Extremely high position threshold
    EXTREME_HEIGHT_THRESHOLD: float = 10.0  # Threshold for logging extremely high positions
    
    # Position confidence factors
    MIN_NEGATIVE_HEIGHT_CONFIDENCE: float = 0.9  # Confidence factor for negative heights
    MIN_BOUNDARY_CONFIDENCE: float = 0.3  # Minimum confidence for boundary validation
    
    # Extreme boundary
    EXTREME_BOUNDARY_FACTOR: float = 2.0  # Factor for extreme boundary positions
    
    # Z-axis noise multiplier
    Z_AXIS_NOISE_MULTIPLIER: float = 1.5  # Multiplier for z-axis measurement noise
    
    # State uncertainty values
    POSITION_UNCERTAINTY: Tuple[float, float, float] = (0.1, 0.1, 0.2)  # Initial position uncertainty
    VELOCITY_UNCERTAINTY: Tuple[float, float, float] = (25.0, 25.0, 25.0)  # Initial velocity uncertainty
    
    # Confidence scaling
    MIN_CONFIDENCE_SCALING: float = 0.05  # Minimum confidence scaling factor
    MIN_CONFIDENCE_THRESHOLD: float = 0.1  # Minimum confidence threshold for measurements 