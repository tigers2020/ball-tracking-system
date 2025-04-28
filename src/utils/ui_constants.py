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