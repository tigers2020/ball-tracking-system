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
import numpy as np
import logging
import cv2


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
    # Adaptive parameters for cropped images
    MIN_RADIUS_CROPPED: int = 8
    MAX_RADIUS_CROPPED: int = 14
    MAX_PARAM2_CROPPED: int = 25


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
class TRACKING:
    """Tracking parameters"""
    ENABLED: bool = True
    MAX_LOST_FRAMES: int = 20
    MIN_DETECTION_CONFIDENCE: float = 0.5
    VALID_RADIUS_MIN: int = 10
    VALID_RADIUS_MAX: int = 30
    HISTORY_LENGTH: int = 120
    DETECTION_INTERVAL: int = 1
    PREDICTION_FRAMES: int = 5
    MIN_PIXEL_THRESHOLD: int = 50  # Minimum number of white pixels required
    # Visualization parameters
    ROI_THICKNESS: int = 4  # Line thickness for ROI rectangles
    CIRCLE_THICKNESS: int = 4  # Line thickness for detected circles
    PREDICTION_THICKNESS: int = 4  # Line thickness for prediction arrows
    UNCERTAINTY_RADIUS: int = 20  # Radius for uncertainty circle
    TRAJECTORY_THICKNESS: int = 5  # Line thickness for trajectory
    TRAJECTORY_MAX_POINTS: int = 20  # Maximum number of points to display in trajectory
    # ROI movement constraints
    MAX_ROI_JUMP_FACTOR: float = 0.3  # Maximum allowed ROI movement as a fraction of frame size
    # Visualization colors
    MAIN_CIRCLE_COLOR: Tuple[int, int, int] = (0, 255, 0)  # Green for main circles
    PREDICTION_ARROW_COLOR: Tuple[int, int, int] = (0, 255, 255)  # Yellow-green for prediction arrows
    TRAJECTORY_COLOR: Tuple[int, int, int] = (255, 255, 0)  # Yellow for trajectory lines
    CENTER_POINT_COLOR: Tuple[int, int, int] = (0, 0, 255)  # Red for center points


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
    ORANGE: Tuple[int, int, int] = (0, 165, 255)
    PURPLE: Tuple[int, int, int] = (128, 0, 128)
    LIME: Tuple[int, int, int] = (0, 255, 191)


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
    IMAGES_FILTER: str = "Images (*.jpg *.jpeg *.png *.bmp);;All Files (*)"
    XML_FILTER: str = "XML Files (*.xml);;All Files (*)"
    CONFIG_FILTER: str = "JSON Files (*.json);;All Files (*)"
    VIDEO_FILTER: str = "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
    DATA_FILTER: str = "Data Files (*.csv *.json);;All Files (*)"
    ALL_FILES_FILTER: str = "All Files (*)"


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
    LOADING_CONFIG: str = "Loading configuration..."
    SAVING_CONFIG: str = "Saving configuration..."
    ERROR_LOADING_CONFIG: str = "Error loading configuration"
    ERROR_SAVING_CONFIG: str = "Error saving configuration"
    CALIBRATION_STARTED: str = "Calibration started"
    CALIBRATION_COMPLETED: str = "Calibration completed"
    CALIBRATION_FAILED: str = "Calibration failed"
    CALIBRATION_CANCELLED: str = "Calibration cancelled"
    CALIBRATION_SAVING: str = "Saving calibration data..."
    CALIBRATION_LOADING: str = "Loading calibration data..."
    CALIBRATION_ERROR: str = "Error during calibration"
    DETECTION_STARTED: str = "Ball detection started"
    DETECTION_COMPLETED: str = "Ball detection completed"
    DETECTION_FAILED: str = "Ball detection failed"
    DETECTION_CANCELLED: str = "Ball detection cancelled"
    DETECTION_SAVING: str = "Saving detection data..."
    DETECTION_LOADING: str = "Loading detection data..."
    DETECTION_ERROR: str = "Error during detection"


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
    MAX_VALID_HEIGHT: float = 999.0  # Maximum valid height for ball detection (m) - increased to allow all heights
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

# =============================================
# Region of Interest (ROI) Synchronization Constants
# =============================================

@dataclass
class ROI_SYNC:
    """ROI synchronization parameters"""
    MAX_JUMP_PX: int = 40  # Maximum allowed ROI jump in pixels
    SYNC_ROI: bool = True  # Whether to synchronize left and right ROI
    SYNC_PARAMS: bool = True  # Whether to synchronize detection parameters
    MASTER_CAMERA: str = "left"  # Which camera is the master for ROI tracking
    SYNC_OFFSET_X: int = 0  # X-offset between left and right camera ROIs
    SYNC_OFFSET_Y: int = 0  # Y-offset between left and right camera ROIs


# =============================================
# Stereo Calibration Constants
# =============================================

@dataclass
class STEREO:
    """Stereo calibration constants"""
    DEFAULT_BASELINE_M: float = 0.60  # Default baseline in meters
    DEFAULT_CAMERA_HEIGHT_M: float = 3.0  # Default camera height in meters
    DEFAULT_SCALE: float = 0.00290  # Default scale (meters per pixel)
    DATA_TYPE: type = np.float32  # Data type for matrices and vectors
    
    # Default rotation angles in degrees
    DEFAULT_PITCH_DEG: float = 0.0  # Rotation around X-axis
    DEFAULT_YAW_DEG: float = 0.0  # Rotation around Y-axis
    DEFAULT_ROLL_DEG: float = 0.0  # Rotation around Z-axis
    
    # Rotation matrix calculation parameters
    DEG_TO_RAD: float = np.pi / 180.0  # Conversion factor for degrees to radians
    
    # Data validation thresholds
    MIN_VALID_DISTANCE: float = 0.5  # Minimum valid distance in meters
    MAX_VALID_DISTANCE: float = 15.0  # Maximum valid distance in meters
    
    # Triangulation parameters
    MIN_DISPARITY: float = 5.0  # Minimum disparity in pixels
    TRIANGULATION_METHOD: int = 0  # Use cv2.triangulatePoints function instead of a constant 

# =============================================
# UI Colors (CSS Format for stylesheets)
# =============================================

@dataclass
class UI_COLORS:
    """Color constants in CSS format for UI styling"""
    # Background gradients
    BG_DARK: str = "#12122a"  
    BG_MID: str = "#151530"   
    BG_LIGHT: str = "#1a1a2e"  
    
    # Accent colors
    ACCENT_PRIMARY: str = "#5d48e0"  
    ACCENT_SECONDARY: str = "#b545e0"  
    ACCENT_TERTIARY: str = "#4848e0"  
    
    # Feature colors
    FEATURE_TITLE: str = "#80d8ff"  
    FEATURE_TEXT: str = "#c5d0e0"  
    
    # Tab colors
    TAB_BG: str = "#1c1c35"  
    TAB_BG_HOVER: str = "#2f2f65"  
    TAB_TEXT: str = "#b0b0d0"  
    TAB_TEXT_SELECTED: str = "#ffffff"  
    
    # Card colors
    CARD_BG_DARK: str = "#151533"  
    CARD_BG_LIGHT: str = "#191940"  
    CARD_BORDER: str = "rgba(100, 100, 255, 0.1)"  
    
    # Highlight colors
    HIGHLIGHT_YELLOW: str = "#ffd966"  
    HIGHLIGHT_PINK: str = "#ff7eb9"  
    
    # Converts BGR COLOR constants to CSS hex format
    @staticmethod
    def bgr_to_css(bgr_color):
        """Convert BGR color tuple to CSS hex color string"""
        if len(bgr_color) == 3:
            b, g, r = bgr_color
            return f"#{r:02x}{g:02x}{b:02x}"
        return None
        
    # COLOR class constants converted to CSS
    RED: str = "#ff0000"  # COLOR.RED in CSS
    GREEN: str = "#00ff00"  # COLOR.GREEN in CSS
    BLUE: str = "#0000ff"  # COLOR.BLUE in CSS
    YELLOW: str = "#ffff00"  # COLOR.YELLOW in CSS
    MAGENTA: str = "#ff00ff"  # COLOR.MAGENTA in CSS
    CYAN: str = "#00ffff"  # COLOR.CYAN in CSS
    WHITE: str = "#ffffff"  # COLOR.WHITE in CSS
    BLACK: str = "#000000"  # COLOR.BLACK in CSS
    GRAY: str = "#808080"  # COLOR.GRAY in CSS
    LIGHT_GRAY: str = "#c0c0c0"  # COLOR.LIGHT_GRAY in CSS
    DARK_GRAY: str = "#404040"  # COLOR.DARK_GRAY in CSS
    ORANGE: str = "#ffa500"  # COLOR.ORANGE in CSS
    PURPLE: str = "#800080"  # COLOR.PURPLE in CSS
    LIME: str = "#bfff00"  # COLOR.LIME in CSS 

# =============================================
# Camera and Tracking Constants
# =============================================

# Camera indexes for left and right cameras
LEFT_CAMERA_INDEX = 0
RIGHT_CAMERA_INDEX = 1

# Tracking data storage path
TRACKING_DATA_DIR = os.path.join(ROOT_DIR, "data", "tracking")
DEFAULT_STORAGE_PATH = os.path.join(ROOT_DIR, "data") 