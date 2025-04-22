#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Application Controller module.
This module contains the AppController class which connects the model and view components.
"""

import logging
import os
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Slot, Signal, Qt, QThread
from PySide6.QtGui import QPixmap
import numpy as np

from src.models.stereo_image_model import StereoImageModel
from src.views.main_window import MainWindow
from src.utils.ui_constants import Messages, Timing
from src.utils.config_manager import ConfigManager
from src.controllers.ball_tracking_controller import BallTrackingController
from src.views.ball_tracking_settings_dialog import BallTrackingSettingsDialog


class FrameLoaderThread(QThread):
    """
    Thread for preloading frames in the background.
    This improves UI responsiveness during playback.
    """
    
    def __init__(self, model, frame_indices, parent=None):
        """
        Initialize the frame loader thread.
        
        Args:
            model (StereoImageModel): The model containing frames
            frame_indices (list): List of frame indices to preload
            parent (QObject, optional): Parent object
        """
        super(FrameLoaderThread, self).__init__(parent)
        self.model = model
        self.frame_indices = frame_indices
        
    def run(self):
        """Execute the thread's main task: preloading frames."""
        for index in self.frame_indices:
            if self.isInterruptionRequested():
                return
                
            frame = self.model.get_frame(index)
            if frame:
                # Just load the images into memory
                frame.load_images()


class AppController(QObject):
    """
    Controller class for the Stereo Image Player application.
    Connects the model and view components.
    """
    
    def __init__(self):
        """Initialize the application controller."""
        super(AppController, self).__init__()
        
        # Create configuration manager
        self.config_manager = ConfigManager()
        
        # Create model and view
        self.model = StereoImageModel()
        self.view = MainWindow()
        
        # Create playback timer
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._on_next_frame)
        
        # Frame skipping flag
        self.skip_frames = False
        self.skip_frame_count = 16  # Number of frames to skip
        
        # Preload thread
        self.preload_thread = None
        
        # Connect signals from model
        self.model.loading_progress.connect(self._on_loading_progress)
        self.model.loading_complete.connect(self._on_loading_complete)
        self.model.loading_error.connect(self.view.show_error_message)
        self.model.frame_changed.connect(self._on_frame_changed)
        
        # Connect signals from view
        self.view.image_view.playback_controls.play_clicked.connect(self._on_play)
        self.view.image_view.playback_controls.pause_clicked.connect(self._on_pause)
        self.view.image_view.playback_controls.stop_clicked.connect(self._on_stop)
        self.view.image_view.playback_controls.next_clicked.connect(self._on_next_frame)
        self.view.image_view.playback_controls.prev_clicked.connect(self._on_prev_frame)
        self.view.image_view.playback_controls.frame_changed.connect(self._on_frame_slider_changed)
        self.view.image_view.playback_controls.fps_changed.connect(self._on_fps_changed)
        self.view.image_view.playback_controls.skip_frames_changed.connect(self._on_skip_frames_changed)
        
        # Connect signals from settings view
        self.view.setting_view.settings_changed.connect(self._on_settings_changed)
        self.view.setting_view.folder_selected.connect(self.open_folder)
        
        # Override view methods
        self.view.open_folder = self.open_folder
        
        # Initialize UI state
        self.view.image_view.enable_controls(False)
        
        # Update settings view with last folder path
        last_folder = self.config_manager.get_last_image_folder()
        if last_folder:
            self.view.setting_view.set_folder_path(last_folder)
        
        # Variables
        self.progress_dialog = None
        
        # Initialize ball tracking controller with the same config manager
        self.ball_tracking_controller = BallTrackingController()
        self.ball_tracking_controller.mask_updated.connect(self._on_mask_updated)
        
        # Initialize ball tracking settings dialog
        self.ball_tracking_dialog = None
        
        # Connect ball tracking button
        self.view.image_view.playback_controls.ball_tracking_clicked.connect(self._on_ball_tracking_button_clicked)
    
    def show(self):
        """Show the main window."""
        self.view.show()
        
        # Load last folder if available
        last_folder = self.config_manager.get_last_image_folder()
        if last_folder and os.path.isdir(last_folder):
            self.open_folder(last_folder)
    
    def open_folder(self, folder_path):
        """
        Open a folder containing stereo images.
        
        Args:
            folder_path (str): Path to the folder
        """
        if not folder_path or not os.path.isdir(folder_path):
            self.view.show_error_message("Invalid folder path")
            return
            
        logging.info(f"Opening folder: {folder_path}")
        self.view.update_status(Messages.LOADING_FOLDER)
        
        # Create progress dialog
        self.progress_dialog = self.view.create_progress_dialog(
            "Loading Images",
            Messages.LOADING_FOLDER,
            0, 100
        )
        self.progress_dialog.show()
        
        # Save to config
        self.config_manager.set_last_image_folder(folder_path)
        
        # Update settings view
        self.view.setting_view.set_folder_path(folder_path)
        
        # Load the folder in the model
        self.model.load_from_folder(folder_path)
    
    @Slot(int, int)
    def _on_loading_progress(self, current, total):
        """
        Handle loading progress updates.
        
        Args:
            current (int): Current progress
            total (int): Total progress
        """
        if self.progress_dialog and total > 0:
            progress_percentage = int((current / total) * 100)
            self.progress_dialog.setValue(progress_percentage)
    
    @Slot()
    def _on_loading_complete(self):
        """Handle loading completion."""
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        # Update UI with loaded frames
        total_frames = len(self.model.frames)
        
        if total_frames > 0:
            self.view.update_status(f"Loaded {total_frames} frames")
            self.view.image_view.playback_controls.set_total_frames(total_frames)
            self.view.image_view.enable_controls(True)
            
            # Load the first frame
            self.model.set_current_frame_index(0)
        else:
            self.view.update_status(Messages.NO_IMAGES_FOUND)
            self.view.show_warning_message(Messages.NO_IMAGES_FOUND)
    
    @Slot(int)
    def _on_frame_changed(self, frame_index):
        """
        Handle frame change in the model.
        
        Args:
            frame_index (int): New frame index
        """
        # Update the view
        self.view.image_view.playback_controls.set_frame(frame_index)
        
        frame = self.model.get_current_frame()
        if frame:
            left_image = frame.get_left_image()
            right_image = frame.get_right_image()
            
            self.view.image_view.set_images(left_image, right_image)
            
            # Update ball tracking controller with new images
            if self.ball_tracking_controller.is_enabled:
                self.ball_tracking_controller.set_images(left_image, right_image)
        else:
            self.view.image_view.clear_images()
        
        # Preload frames in background
        self._preload_frames_in_background(5)
    
    def _preload_frames_in_background(self, num_frames):
        """
        Preload frames in a background thread for smoother playback.
        
        Args:
            num_frames (int): Number of frames to preload ahead
        """
        # Stop any existing preload thread
        if self.preload_thread and self.preload_thread.isRunning():
            self.preload_thread.requestInterruption()
            self.preload_thread.wait()
        
        # Calculate frame indices to preload
        current_index = self.model.current_frame_index
        total_frames = len(self.model.frames)
        
        if total_frames <= 1:
            return
            
        # Get skip frames state directly from UI
        is_skipping = self.view.image_view.playback_controls.is_skipping_frames()
        
        # Determine frame indices to preload based on skip setting
        indices = []
        step = self.skip_frame_count if is_skipping else 1
        
        for i in range(1, num_frames + 1):
            next_index = (current_index + i * step) % total_frames
            indices.append(next_index)
        
        # Create and start the preload thread
        self.preload_thread = FrameLoaderThread(self.model, indices)
        self.preload_thread.start()
    
    @Slot()
    def _on_play(self):
        """Handle play button click."""
        if not self.model.frames:
            return
        
        # Set up the timer with current FPS
        fps = self.view.image_view.playback_controls.get_current_fps()
        interval = 1000 // fps
        self.playback_timer.setInterval(interval)
        
        # Start the timer
        self.playback_timer.start()
        
        # Update model state
        self.model.is_playing = True
        
        # Close ball tracking dialog if open during playback
        if self.ball_tracking_dialog and self.ball_tracking_dialog.isVisible():
            self.ball_tracking_dialog.close()
        
        # Update status
        self.view.update_status(Messages.PLAYBACK_STARTED)
    
    @Slot()
    def _on_pause(self):
        """Handle pause button click."""
        if self.playback_timer.isActive():
            self.playback_timer.stop()
        
        # Update model state
        self.model.is_playing = False
        
        # Update status
        self.view.update_status(Messages.PLAYBACK_PAUSED)
        
        # Refresh ball tracking display if enabled
        if self.ball_tracking_controller.is_enabled:
            frame = self.model.get_current_frame()
            if frame:
                left_image = frame.get_left_image()
                right_image = frame.get_right_image()
                self.ball_tracking_controller.set_images(left_image, right_image)
    
    @Slot()
    def _on_stop(self):
        """Handle stop button click."""
        if self.playback_timer.isActive():
            self.playback_timer.stop()
        
        # Update model state
        self.model.is_playing = False
        
        # Go to the first frame
        self.model.set_current_frame_index(0)
        
        # Update status
        self.view.update_status(Messages.PLAYBACK_STOPPED)
        
        # Refresh ball tracking display if enabled
        if self.ball_tracking_controller.is_enabled:
            frame = self.model.get_current_frame()
            if frame:
                left_image = frame.get_left_image()
                right_image = frame.get_right_image()
                self.ball_tracking_controller.set_images(left_image, right_image)
    
    @Slot()
    def _on_next_frame(self):
        """Handle next frame button click or timer timeout."""
        # Get the current frame skip state directly from the UI
        self.skip_frames = self.view.image_view.playback_controls.is_skipping_frames()
        
        if self.skip_frames and self.model.is_playing:
            # Skip frames during playback
            current_index = self.model.current_frame_index
            total_frames = len(self.model.frames)
            
            if total_frames <= 1:
                return
                
            # Calculate next frame index with skipping
            next_index = (current_index + self.skip_frame_count) % total_frames
            self.model.set_current_frame_index(next_index)
        else:
            # Normal next frame behavior
            next_frame = self.model.next_frame()
            
            # If we reached the end of the sequence during playback, loop back to the start
            if not next_frame and self.model.is_playing:
                self.model.set_current_frame_index(0)
    
    @Slot()
    def _on_prev_frame(self):
        """Handle previous frame button click."""
        # Get the current frame skip state directly from the UI
        self.skip_frames = self.view.image_view.playback_controls.is_skipping_frames()
        
        if self.skip_frames:
            # Skip frames when navigating backwards
            current_index = self.model.current_frame_index
            total_frames = len(self.model.frames)
            
            if total_frames <= 1:
                return
                
            # Calculate previous frame index with skipping
            prev_index = (current_index - self.skip_frame_count) % total_frames
            self.model.set_current_frame_index(prev_index)
        else:
            # Normal previous frame behavior
            self.model.prev_frame()
    
    @Slot(int)
    def _on_frame_slider_changed(self, value):
        """
        Handle frame slider value change.
        
        Args:
            value (int): New slider value
        """
        self.model.set_current_frame_index(value)
    
    @Slot(int)
    def _on_fps_changed(self, fps):
        """
        Handle FPS value change.
        
        Args:
            fps (int): New FPS value
        """
        if self.playback_timer.isActive():
            # Update timer interval without stopping playback
            interval = 1000 // fps
            self.playback_timer.setInterval(interval)
    
    @Slot(bool)
    def _on_skip_frames_changed(self, skip):
        """
        Handle skip frames checkbox state change.
        
        Args:
            skip (bool): True if frames should be skipped, False otherwise
        """
        self.skip_frames = skip
        logging.info(f"Frame skipping {'enabled' if skip else 'disabled'}")
        
        # Synchronize with the view's checkbox
        is_checked = self.view.image_view.playback_controls.skip_frames_checkbox.isChecked()
        if is_checked != skip:
            self.view.image_view.playback_controls.skip_frames_checkbox.setChecked(skip)
    
    @Slot(dict)
    def _on_settings_changed(self, settings):
        """
        Handle settings changes from the settings view.
        
        Args:
            settings (dict): New settings
        """
        # Save last image folder
        if "last_image_folder" in settings:
            last_folder = settings["last_image_folder"]
            if last_folder:
                self.config_manager.set_last_image_folder(last_folder)

    @Slot(np.ndarray, np.ndarray)
    def _on_mask_updated(self, left_mask, right_mask):
        """
        Handle mask update from the ball tracking controller.
        
        Args:
            left_mask (numpy.ndarray): Updated left mask
            right_mask (numpy.ndarray): Updated right mask
        """
        # Update the view with the new masks
        self.view.image_view.set_masks(left_mask, right_mask)
    
    @Slot()
    def _on_ball_tracking_button_clicked(self):
        """Handle ball tracking button click."""
        # Create dialog if it doesn't exist
        if not self.ball_tracking_dialog:
            self.ball_tracking_dialog = BallTrackingSettingsDialog(self.view)
            
            # Connect dialog signals
            self.ball_tracking_dialog.hsv_changed.connect(self.ball_tracking_controller.set_hsv_values)
        
        # Set current HSV values in the dialog
        current_hsv = self.ball_tracking_controller.get_hsv_values()
        self.ball_tracking_dialog.set_hsv_values(current_hsv)
        
        # Enable ball tracking and mask overlay
        self.ball_tracking_controller.enable(True)
        self.view.image_view.enable_mask_overlay(True)
        
        # Set current images to the ball tracking controller
        current_frame = self.model.get_current_frame()
        if current_frame:
            left_image = current_frame.get_left_image()
            right_image = current_frame.get_right_image()
            self.ball_tracking_controller.set_images(left_image, right_image)
        
        # Show the dialog
        self.ball_tracking_dialog.exec() 