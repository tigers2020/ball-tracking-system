#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Application Controller module.
This module contains the AppController class which connects the model and view components.
"""

import logging
import os
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Slot, Signal, Qt
from PySide6.QtGui import QPixmap

from src.models.stereo_image_model import StereoImageModel
from src.views.main_window import MainWindow
from src.utils.ui_constants import Messages, Timing
from src.utils.config_manager import ConfigManager


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
        else:
            self.view.image_view.clear_images()
        
        # Preload some frames ahead for smoother playback
        self.model.preload_frames(5)
    
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
    
    @Slot()
    def _on_next_frame(self):
        """Handle next frame button click or timer timeout."""
        next_frame = self.model.next_frame()
        
        # If we reached the end of the sequence during playback, loop back to the start
        if not next_frame and self.model.is_playing:
            self.model.set_current_frame_index(0)
    
    @Slot()
    def _on_prev_frame(self):
        """Handle previous frame button click."""
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