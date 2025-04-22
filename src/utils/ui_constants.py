#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
UI Constants for the Stereo Image Player application.
This file contains all the constants used in the UI.
"""


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
    """Icon resource paths"""
    PLAY = "resources/play.png"
    PAUSE = "resources/pause.png"
    STOP = "resources/stop.png"
    NEXT = "resources/next.png"
    PREVIOUS = "resources/previous.png"
    OPEN = "resources/open.png"
    SAVE = "resources/save.png"
    EXIT = "resources/exit.png" 