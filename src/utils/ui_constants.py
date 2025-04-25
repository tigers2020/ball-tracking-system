#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
UI Constants for the Stereo Image Player application.
This file contains all the constants used in the UI.
"""

import os
from pathlib import Path


# Get the absolute path to the root directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class WindowSize:
    """Window size constants"""
    DEFAULT_WIDTH = 1280
    DEFAULT_HEIGHT = 720
    MIN_WIDTH = 800
    MIN_HEIGHT = 600


class Layout:
    """Layout constants"""
    MARGIN = 10
    SPACING = 5
    BUTTON_HEIGHT = 30
    BUTTON_WIDTH = 120
    SLIDER_HEIGHT = 20
    TOOLBAR_HEIGHT = 40
    IMAGE_VIEW_RATIO = 0.45  # Each image view takes 45% of the width
    CONTROLS_HEIGHT = 50


class Timing:
    """Timing constants"""
    DEFAULT_FPS = 30
    MIN_FPS = 1
    MAX_FPS = 120
    DEFAULT_PLAYBACK_INTERVAL = 1000 // DEFAULT_FPS  # in milliseconds


class FileDialog:
    """File dialog constants"""
    DIALOG_CAPTION = "Select Stereo Images Folder"
    DIALOG_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"


class XML:
    """XML constants"""
    FRAMES_INFO_FILENAME = "frames_info.xml"
    ROOT_ELEMENT = "stereo_frames"
    FRAME_ELEMENT = "frame"
    INDEX_ATTR = "index"
    LEFT_IMAGE_ELEMENT = "left_image"
    RIGHT_IMAGE_ELEMENT = "right_image"
    PATH_ATTR = "path"


class Messages:
    """Message strings for dialogs and status bar"""
    LOADING_FOLDER = "Loading images from folder..."
    GENERATING_XML = "Generating frames info XML..."
    MOVING_FILES = "Moving files to appropriate folders..."
    NO_IMAGES_FOUND = "No stereo images found in the selected folder"
    ERROR_LOADING_XML = "Error loading XML file"
    ERROR_PARSING_XML = "Error parsing XML file"
    ERROR_LOADING_IMAGE = "Error loading image"
    ERROR_SAVING_XML = "Error saving XML file"
    PLAYBACK_STARTED = "Playback started"
    PLAYBACK_PAUSED = "Playback paused"
    PLAYBACK_STOPPED = "Playback stopped"
    READY = "Ready"
    # Calibration tab messages
    CALIBRATION_POINTS_ADDED = "Calibration point added"
    CALIBRATION_POINTS_UPDATED = "Calibration point updated"
    CALIBRATION_POINTS_CLEARED = "Calibration points cleared"
    CALIBRATION_SAVED = "Calibration saved"
    CALIBRATION_LOADED = "Calibration loaded"
    CALIBRATION_FINE_TUNE_START = "Starting fine-tune process..."
    CALIBRATION_FINE_TUNE_COMPLETE = "Fine-tune complete"
    CALIBRATION_FINE_TUNE_ERROR = "Error during fine-tune process"
    ROI_CROPPING_ERROR = "Error cropping ROI"
    SKELETONIZE_ERROR = "Error during skeletonization"
    INTERSECTION_ERROR = "Error finding intersection points"


class Icons:
    """Icon resource paths (absolute paths)"""
    # Directory paths
    ICONS_DIR = os.path.join(ROOT_DIR, "src", "resources", "icons")
    
    # Icon files
    PLAY = os.path.join(ICONS_DIR, "play.svg")
    PAUSE = os.path.join(ICONS_DIR, "pause.svg")
    STOP = os.path.join(ICONS_DIR, "stop.svg")
    NEXT = os.path.join(ICONS_DIR, "next_frame.svg")
    PREV = os.path.join(ICONS_DIR, "prev_frame.svg")
    SETTINGS = os.path.join(ICONS_DIR, "check.svg")  # Using check.svg as a substitute
    OPEN = os.path.join(ICONS_DIR, "check.svg")      # Using check.svg as a substitute
    EXIT = os.path.join(ICONS_DIR, "stop.svg") 
    # Calibration icons
    SAVE = os.path.join(ICONS_DIR, "check.svg")      # Using check.svg as a substitute
    LOAD = os.path.join(ICONS_DIR, "check.svg")      # Using check.svg as a substitute
    CLEAR = os.path.join(ICONS_DIR, "stop.svg")      # Using stop.svg as a substitute
    FINE_TUNE = os.path.join(ICONS_DIR, "check.svg") # Using check.svg as a substitute


class ROI:
    """ROI constants"""
    DEFAULT_WIDTH = 100
    DEFAULT_HEIGHT = 100
    MIN_SIZE = 10
    MAX_SIZE = 500
    BORDER_THICKNESS = 2
    BORDER_COLOR = (0, 0, 255)  # BGR format (Red)
    FILL_COLOR = (0, 0, 255, 64)  # BGRA format (Red with transparency)
    CENTER_MARKER_SIZE = 5
    CENTER_MARKER_COLOR = (255, 255, 255)  # BGR format (White)


class Calibration:
    """Calibration constants"""
    POINT_RADIUS = 5
    ORIGINAL_POINT_COLOR = (255, 0, 0)  # RGB format (Red)
    ADJUSTED_POINT_COLOR = (0, 255, 0)  # RGB format (Green)
    POINT_LABEL_OFFSET = (10, 10)
    LINE_COLOR = (0, 0, 255)  # RGB format (Blue)
    LINE_WIDTH = 1
    Z_VALUE_ORIGINAL = 1.0
    Z_VALUE_ADJUSTED = 0.0
    
    # Fine-tuning parameters
    ROI_SIZE_MIN = 40  # Minimum ROI size in pixels
    ROI_SIZE_FACTOR = 2.5  # Factor to multiply radius by for ROI size
    
    # Hough transform parameters
    HOUGH_THRESHOLD_FACTOR = 0.005  # Threshold as fraction of ROI area
    HOUGH_MIN_LINE_LENGTH_FACTOR = 0.5  # MinLineLength as fraction of ROI size
    HOUGH_MAX_LINE_GAP_FACTOR = 0.167  # MaxLineGap as fraction of ROI size (1/6)
    
    # File path for configuration
    CONFIG_FILE = os.path.join(ROOT_DIR, "config", "calibration.json")
    
    # Configuration save cooldown time (seconds)
    CONFIG_SAVE_COOLDOWN = 2.0 