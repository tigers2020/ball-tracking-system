#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Application Controller module.
This module contains the AppController class which connects the model and view components.
"""

import logging
import os
from pathlib import Path
import time

from PySide6.QtCore import QObject, QTimer, Slot, Signal, Qt, QThread
from PySide6.QtGui import QPixmap
import numpy as np

from src.models.stereo_image_model import StereoImageModel
from src.views.main_window import MainWindow
from src.utils.ui_constants import Messages, Timing
from src.utils.config_manager import ConfigManager
from src.utils.parameter_manager import ParameterManager
from src.utils.signal_binder import SignalBinder
from src.controllers.ball_tracking_controller import BallTrackingController, TrackingState
from src.controllers.game_analyzer import GameAnalyzer
from src.views.ball_tracking_settings_dialog import BallTrackingSettingsDialog
from src.controllers.calibration_controller import CalibrationController
from src.controllers.data_export_controller import DataExportController
from src.controllers.tracking_coordinates_controller import TrackingCoordinatesController
from src.controllers.tracking_integration import setup_tracking_overlay


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
        
        # Create parameter manager
        self.parameter_manager = ParameterManager(self.config_manager)
        
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
        
        # Connect app closing signal
        self.view.app_closing.connect(self._on_app_closing)
        
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
        
        # Initialize ball tracking controller with the parameter_manager
        self.ball_tracking_controller = BallTrackingController(self.model, self.config_manager)
        
        # Initialize game analyzer controller with the same config manager
        self.game_analyzer = GameAnalyzer(self.config_manager)
        
        # Connect ball_tracking_controller to game_analyzer
        self._connect_ball_tracking_to_game_analyzer()
        
        # Connect image_view to ball_tracking_controller
        self.view.image_view.connect_ball_tracking_controller(self.ball_tracking_controller)
        
        # Connect image_view to game_analyzer
        self.view.image_view.connect_game_analyzer(self.game_analyzer)
        
        # Connect main window to game_analyzer for IN/OUT indication
        self.view.connect_game_analyzer(self.game_analyzer)
        
        # Set up tracking overlay
        self._setup_tracking_overlay()
        
        # Initialize ball tracking settings dialog
        self.ball_tracking_dialog = None
        
        # Connect ball tracking button
        self.view.image_view.playback_controls.ball_tracking_clicked.connect(self._on_ball_tracking_button_clicked)
        
        # 볼 트래킹 버튼 활성화 (이미지 로드 후에 사용 가능하도록)
        self.view.image_view.playback_controls.ball_tracking_button.setEnabled(True)
        
        # Connect ball_tracking signals to view
        self.ball_tracking_controller.mask_updated.connect(self._on_mask_updated)
        self.ball_tracking_controller.roi_updated.connect(self._on_roi_updated)
        self.ball_tracking_controller.circles_processed.connect(self.view.image_view.set_images)
        
        # Set the stereo image model to the calibration controller
        self.view.calibration_controller.set_stereo_image_model(self.model)
        
        # Set the config manager to the calibration controller
        self.view.calibration_controller.set_config_manager(self.config_manager)
        
        # Connect parameter_manager signals to controllers
        self._connect_parameter_manager_signals()
        
        # Tracking data save settings
        self.tracking_data_save_enabled = True  # Enable/disable saving
        self.tracking_data_folder = os.path.join(os.getcwd(), "tracking_data")  # Default folder
        
        # Data export controller initialization
        self.data_export_controller = DataExportController(self.config_manager)
        
        # Connect data export signals
        self.data_export_controller.export_successful.connect(self._on_data_exported)
        self.data_export_controller.import_successful.connect(self._on_data_imported)
    
    def _setup_tracking_overlay(self):
        """
        트래킹 좌표 오버레이 컴포넌트 초기화 및 연결
        """
        # 트래킹 좌표 컨트롤러 설정 및 연결
        self.tracking_coord_controller = setup_tracking_overlay(
            app_window=self,
            ball_tracking_controller=self.ball_tracking_controller,
            config_manager=self.config_manager,
            image_view=self.view.image_view
        )
        
        # 트래킹 오버레이 기능 활성화
        self.view.image_view.enable_tracking_overlay(True)
        
        # 직접 signal-slot 연결 추가 (signal_binder 문제 해결)
        # ball_tracking_controller의 detection_updated 시그널과 image_view의 _on_detection_updated 메소드 연결
        self.ball_tracking_controller.detection_updated.connect(
            self.view.image_view._on_detection_updated
        )

        # 트래킹 활성화 상태 로깅
        logging.info("Tracking coordinates overlay initialized and connected")
        logging.debug(f"Tracking overlay visible: {self.view.image_view.tracking_overlay.isVisible()}")
        logging.debug(f"Ball tracking controller enabled: {self.ball_tracking_controller.is_enabled}")
        
        # Ball tracking 버튼이 있는지 확인하고 활성화
        if hasattr(self.view.image_view.playback_controls, 'ball_tracking_button'):
            self.view.image_view.playback_controls.ball_tracking_button.setEnabled(True)
            logging.debug("Ball tracking button enabled")
    
    def _connect_ball_tracking_to_game_analyzer(self):
        """
        Connect the ball tracking controller to game analyzer for 3D analysis.
        """
        # Connect ball tracking 2D detection signals to game analyzer using a lambda function
        connected = SignalBinder.bind_lambda(
            self.ball_tracking_controller, 
            "detection_updated",
            lambda frame_idx, detection_rate, left_coords, right_coords: 
                self._on_ball_detection_updated(frame_idx, detection_rate, left_coords, right_coords)
        )
        
        # Check if connection was successful
        logging.info(f"Ball tracking detection signal connected to game analyzer: {connected}")
        
        # Connect ball tracking controller signals using SignalBinder for lambda connections
        SignalBinder.bind_lambda(
            self.ball_tracking_controller,
            "tracking_enabled_changed",
            lambda enabled: self.view.image_view.playback_controls.ball_tracking_button.setChecked(enabled)
        )
        
        SignalBinder.bind_lambda(
            self.ball_tracking_controller,
            "tracking_state_changed",
            lambda state: self.game_analyzer.enable(state == TrackingState.TRACKING or state == TrackingState.TRACKING_LOST)
        )
        
        # Connect bounce overlay to game analyzer
        self.view.image_view.bounce_overlay.connect_game_analyzer(self.game_analyzer)
        
        logging.info("Ball tracking controller connected to game analyzer")
    
    @Slot(int, float, tuple, tuple)
    def _on_ball_detection_updated(self, frame_idx, detection_rate, left_coords, right_coords):
        """
        Handle ball detection updates from ball tracking controller.
        
        Args:
            frame_idx (int): Frame index
            detection_rate (float): Detection rate between 0-1
            left_coords (tuple): Coordinates in left image (x, y, r) or None
            right_coords (tuple): Coordinates in right image (x, y, r) or None
        """
        timestamp = self.model.get_frame_timestamp(frame_idx) if frame_idx >= 0 else 0.0
        
        # Log the call to GameAnalyzer
        logging.debug(f"Calling GameAnalyzer.on_ball_detected with frame={frame_idx}, left={left_coords}, right={right_coords}")
        
        # Extract pixel coordinates from tuple (ignoring radius)
        left_point = None if left_coords is None else (left_coords[0], left_coords[1])
        right_point = None if right_coords is None else (right_coords[0], right_coords[1])
        
        # Pass to game analyzer for 3D tracking
        self.game_analyzer.on_ball_detected(frame_idx, timestamp, detection_rate, left_point, right_point)
    
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

        # Update calibration tab with current images
        if hasattr(self, 'current_left_image') and hasattr(self, 'current_right_image'):
            self.view.update_calibration_images(
                self.current_left_image,
                self.current_right_image
            )
    
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
            
            # 현재 프레임이 로드되었으므로 볼 트래킹 버튼 활성화
            self.view.image_view.playback_controls.ball_tracking_button.setEnabled(True)
            
            # Initialize XML tracking for the current folder
            folder_name = os.path.basename(self.config_manager.get_last_image_folder())
            if folder_name:
                self.ball_tracking_controller.initialize_xml_tracking(folder_name)
                logging.info(f"XML tracking initialized for folder: {folder_name}")
            
            # Load the first frame
            self.model.set_current_frame_index(0)
        else:
            self.view.update_status(Messages.NO_IMAGES_FOUND)
            self.view.show_warning_message(Messages.NO_IMAGES_FOUND)
            
            # 이미지가 없으면 볼 트래킹 버튼 비활성화 유지
            self.view.image_view.playback_controls.ball_tracking_button.setEnabled(False)

        # Update calibration tab with current images
        if hasattr(self, 'current_left_image') and hasattr(self, 'current_right_image'):
            self.view.update_calibration_images(
                self.current_left_image,
                self.current_right_image
            )
    
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
                
                # Process the current frame for tracking and XML logging
                self.ball_tracking_controller.process_frame(frame_index)
                
                # Save entire XML file at intervals to prevent performance issues
                if frame_index % 10 == 0:  # Save every 10 frames
                    # Use folder name based on current image folder
                    folder_name = os.path.basename(self.config_manager.get_last_image_folder())
                    if not folder_name:
                        folder_name = "default"
                    
                    # Create specific folder for this dataset
                    tracking_folder = os.path.join(self.tracking_data_folder, folder_name)
                    
                    # Save XML tracking data periodically
                    self.ball_tracking_controller.save_xml_tracking_data(tracking_folder)
                    logging.debug(f"XML tracking data saved at frame {frame_index} (periodic save)")
        else:
            self.view.image_view.clear_images()
        
        # Preload frames in background
        self._preload_frames_in_background(5)

        # Update calibration tab with current images
        if hasattr(self, 'current_left_image') and hasattr(self, 'current_right_image'):
            self.view.update_calibration_images(
                self.current_left_image,
                self.current_right_image
            )
    
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
        
        # Reset ball tracking data
        self.ball_tracking_controller.reset_tracking()
        # Disable game analyzer when stopping playback
        self.game_analyzer.enable(False)
        
        # Reset detection rate and clear info view
        if hasattr(self.view, 'info_view'):
            self.view.info_view.clear_info()
        
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
    
    @Slot(dict, dict)
    def _on_roi_updated(self, left_roi, right_roi):
        """
        Handle ROI update from the ball tracking controller.
        
        Args:
            left_roi (dict): Left ROI data
            right_roi (dict): Right ROI data
        """
        self.view.image_view.set_rois(left_roi, right_roi)
    
    @Slot()
    def _on_ball_tracking_button_clicked(self):
        """Handle ball tracking button click."""
        # 현재 이미지가 로드되었는지 확인
        current_frame = self.model.get_current_frame()
        if not current_frame:
            self.view.show_warning_message("이미지를 먼저 로드해주세요.")
            return
            
        left_image = current_frame.get_left_image()
        right_image = current_frame.get_right_image()
        
        if left_image is None and right_image is None:
            self.view.show_warning_message("이미지 로딩이 완료되지 않았습니다. 잠시 후 다시 시도해주세요.")
            return
            
        # Create dialog if it doesn't exist
        if not self.ball_tracking_dialog:
            self.ball_tracking_dialog = BallTrackingSettingsDialog(self.view)
            
            # Connect dialog signals
            self.ball_tracking_dialog.hsv_changed.connect(self.ball_tracking_controller.set_hsv_values)
        
        # Set current HSV values in the dialog
        current_hsv = self.ball_tracking_controller.get_hsv_values()
        self.ball_tracking_dialog.set_hsv_values(current_hsv)
        
        # 순서 변경: 이미지 먼저 설정 후 활성화
        logging.info("Setting images before enabling ball tracking")
        
        # Set current images to the ball tracking controller
        self.ball_tracking_controller.set_images(left_image, right_image)
        
        # Enable ball tracking and mask overlay
        self.ball_tracking_controller.enable(True)
        # Enable game analyzer to process 3D positions
        self.game_analyzer.enable(True)
        self.view.image_view.enable_mask_overlay(True)
        
        # Show the dialog
        self.ball_tracking_dialog.exec()

    def set_tracking_data_save(self, enabled, folder=None):
        """
        Enable or disable tracking data saving and set the output folder.
        
        Args:
            enabled (bool): Whether to save tracking data
            folder (str, optional): Output folder path. If None, use default.
        """
        self.tracking_data_save_enabled = enabled
        
        if folder:
            self.tracking_data_folder = folder
        
        if enabled:
            os.makedirs(self.tracking_data_folder, exist_ok=True)
            logging.info(f"Tracking data saving enabled. Output folder: {self.tracking_data_folder}")
        else:
            logging.info("Tracking data saving disabled")

    def save_all_tracking_data(self):
        """
        Save all accumulated tracking data to a comprehensive XML or JSON file.
        Call this on application exit or when user wants to save all tracking
        data for further analysis.
        
        Returns:
            str: Path to the saved file
        """
        if not self.tracking_data_save_enabled:
            logging.info("Tracking data saving is disabled")
            return None
        
        if not hasattr(self, 'ball_tracking_controller') or not self.ball_tracking_controller:
            logging.warning("Ball tracking controller not initialized")
            return None
        
        # Folder name based on current image folder
        folder_name = os.path.basename(self.config_manager.get_last_image_folder())
        if not folder_name:
            folder_name = "default"
        
        # Create specific folder for this dataset
        tracking_folder = os.path.join(self.tracking_data_folder, folder_name)
        
        # First try XML saving (primary method)
        try:
            # Create XML data structure
            detection_rate = self.ball_tracking_controller.get_detection_rate()
            total_frames = self.ball_tracking_controller.detection_stats["total_frames"]
            detection_count = self.ball_tracking_controller.detection_stats["detection_count"]
            
            xml_data = {
                "root_tag": "TrackingData",
                "dataset": folder_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "detection_rate": detection_rate,
                "frames": {
                    "total": total_frames,
                    "detected": detection_count
                },
                "coordinate_history": self.ball_tracking_controller.get_coordinate_history()
            }
            
            # Generate filename
            xml_filename = f"tracking_data_{int(time.time())}"
            
            # Export XML data
            xml_path = self.data_export_controller.export_xml(
                xml_data,
                None,  # No parent widget (silent export)
                "Export Tracking Data",
                os.path.join(tracking_folder, xml_filename)
            )
            
            if xml_path:
                logging.info(f"All tracking data saved to XML: {xml_path}")
                return xml_path
        except Exception as e:
            logging.error(f"Error saving XML tracking data: {e}. Falling back to JSON.")
        
        # Fallback to JSON if XML saving failed
        try:
            # Use dedicated method in data_export_controller
            json_path = self.data_export_controller.export_tracking_data(
                self.ball_tracking_controller,
                None,  # No parent widget (silent export)
                tracking_folder
            )
            
            if json_path:
                logging.info(f"All tracking data saved to JSON (fallback): {json_path}")
                return json_path
        except Exception as e:
            logging.error(f"Error saving tracking data to JSON: {e}")
        
        return None

    @Slot()
    def _on_app_closing(self):
        """Handle application closing."""
        try:
            logging.info("Application closing...")
            
            # Stop playback if active
            if self.playback_timer.isActive():
                self.playback_timer.stop()
            
            # Stop preloading if active
            if self.preload_thread and self.preload_thread.isRunning():
                self.preload_thread.requestInterruption()
                self.preload_thread.wait(1000)  # Wait up to 1 second
            
            # Save settings
            # Any last-minute settings saving happens here
            
            # Reset game analyzer to clean up resources
            if hasattr(self, 'game_analyzer') and self.game_analyzer:
                logging.info("Cleaning up game analyzer resources...")
                self.game_analyzer.reset()
                
                # Save any game analysis data if needed
                if hasattr(self.game_analyzer, 'get_analysis_results'):
                    analysis_results = self.game_analyzer.get_analysis_results()
                    logging.info(f"Game analysis results: detected bounces={analysis_results.get('bounce_count', 0)}")
            
            # Reset ball tracking controller to ensure all data is saved
            if self.ball_tracking_controller:
                logging.info("Cleaning up ball tracking resources...")
                
                # Get detection statistics to include in XML
                detection_stats = None
                if hasattr(self.ball_tracking_controller, 'detection_stats'):
                    detection_stats = self.ball_tracking_controller.detection_stats
                
                # Finalize XML tracking data
                if hasattr(self.ball_tracking_controller, 'data_saver') and self.ball_tracking_controller.data_saver:
                    logging.info("Finalizing XML tracking data...")
                    self.ball_tracking_controller.data_saver.finalize_xml(detection_stats)
                
                self.ball_tracking_controller.reset_tracking()
                
            logging.info("Application cleanup complete")
            
        except Exception as e:
            logging.error(f"Error during application cleanup: {e}")
            
        # Final log message
        logging.info("Application closed")

    def _update_current_frame(self):
        """Update the current frame based on the slider position."""
        # ... existing code to update frame ...

        # Update calibration tab with current images
        if hasattr(self, 'current_left_image') and hasattr(self, 'current_right_image'):
            self.view.update_calibration_images(
                self.current_left_image,
                self.current_right_image
            )

    def _connect_parameter_manager_signals(self):
        """
        Connect parameter manager signals to appropriate controllers.
        """
        # Define lambda functions for parameter updates
        parameter_connections = {
            "hsv_parameters_updated": lambda params: self.ball_tracking_controller.set_hsv_values(params)
                if hasattr(self.ball_tracking_controller, 'set_hsv_values') else None,
            "roi_parameters_updated": lambda params: self.ball_tracking_controller.set_roi_settings(params)
                if hasattr(self.ball_tracking_controller, 'set_roi_settings') else None,
            "hough_parameters_updated": lambda params: self.ball_tracking_controller.set_hough_circle_settings(params)
                if hasattr(self.ball_tracking_controller, 'set_hough_circle_settings') else None,
            "kalman_parameters_updated": lambda params: self.ball_tracking_controller.update_kalman_settings(params)
                if hasattr(self.ball_tracking_controller, 'update_kalman_settings') else None,
            "camera_parameters_updated": lambda params: self.ball_tracking_controller.update_camera_settings(params)
                if hasattr(self.ball_tracking_controller, 'update_camera_settings') else None,
        }
        
        # Connect all parameter signals using SignalBinder
        for signal_name, handler_lambda in parameter_connections.items():
            SignalBinder.bind_lambda(
                self.parameter_manager,
                signal_name,
                handler_lambda
            )
        
        logging.info("Parameter manager signals connected to controllers")

    @Slot(str)
    def _on_data_exported(self, file_path):
        """
        Handle successful data export.
        
        Args:
            file_path (str): Path to the exported file
        """
        logging.info(f"Data successfully exported to: {file_path}")
        
        # Update status bar if available
        if hasattr(self.view, 'update_status_message'):
            self.view.update_status_message(f"Data exported to: {os.path.basename(file_path)}")

    @Slot(str)
    def _on_data_imported(self, file_path):
        """
        Handle successful data import.
        
        Args:
            file_path (str): Path to the imported file
        """
        logging.info(f"Data successfully imported from: {file_path}")
        
        # Update status bar if available
        if hasattr(self.view, 'update_status_message'):
            self.view.update_status_message(f"Data imported from: {os.path.basename(file_path)}")

    @Slot()
    def on_export_tracking_data(self):
        """
        Handle export tracking data button click.
        Opens a dialog for exporting tracking data.
        """
        try:
            # Check if tracking controller is initialized
            if not hasattr(self, 'ball_tracking_controller') or not self.ball_tracking_controller:
                self.view.show_warning_message("Ball tracking controller not initialized.")
                return
            
            # Use data export controller to export tracking data
            self.data_export_controller.export_tracking_data(
                self.ball_tracking_controller,
                self.view
            )
        except Exception as e:
            logging.error(f"Error exporting tracking data: {e}")
            self.view.show_error_message(f"Error exporting tracking data: {str(e)}")

    @Slot()
    def on_import_tracking_data(self):
        """
        Handle import tracking data button click.
        Opens a dialog for importing tracking data.
        """
        try:
            # Check if tracking controller is initialized
            if not hasattr(self, 'ball_tracking_controller') or not self.ball_tracking_controller:
                self.view.show_warning_message("Ball tracking controller not initialized.")
                return
            
            # Import JSON data
            tracking_data = self.data_export_controller.import_json(
                self.view,
                "Import Tracking Data"
            )
            
            if not tracking_data:
                return
            
            # Validate tracking data format
            if not isinstance(tracking_data, dict) or "coordinate_history" not in tracking_data:
                self.view.show_warning_message("Invalid tracking data format.")
                return
            
            # Process tracking data
            self._apply_imported_tracking_data(tracking_data)
            
        except Exception as e:
            logging.error(f"Error importing tracking data: {e}")
            self.view.show_error_message(f"Error importing tracking data: {str(e)}")
        
    def _apply_imported_tracking_data(self, tracking_data):
        """
        Apply imported tracking data to the ball tracking controller.
        
        Args:
            tracking_data (dict): Imported tracking data dictionary
        """
        if not self.ball_tracking_controller:
            return
        
        # Apply coordinate history if available
        if "coordinate_history" in tracking_data:
            # Clear existing history
            self.ball_tracking_controller.clear_coordinate_history()
            
            # Add imported coordinates to history
            coord_history = tracking_data["coordinate_history"]
            for side in ["left", "right"]:
                if side in coord_history:
                    for entry in coord_history[side]:
                        x = entry.get("x")
                        y = entry.get("y")
                        radius = entry.get("radius", 5.0)  # Default radius if not provided
                        timestamp = entry.get("timestamp", time.time())  # Default timestamp if not provided
                        
                        if x is not None and y is not None:
                            # Add to history (assuming model has this method)
                            if hasattr(self.ball_tracking_controller.model, 'add_to_coordinate_history'):
                                self.ball_tracking_controller.model.add_to_coordinate_history(
                                    side, (float(x), float(y), float(radius), timestamp)
                                )
        
        # Apply detection settings if available
        if "detection_settings" in tracking_data:
            settings = tracking_data["detection_settings"]
            
            # Apply HSV settings
            if "hsv" in settings and hasattr(self.ball_tracking_controller, 'set_hsv_values'):
                self.ball_tracking_controller.set_hsv_values(settings["hsv"])
            
            # Apply ROI settings
            if "roi" in settings and hasattr(self.ball_tracking_controller, 'set_roi_settings'):
                self.ball_tracking_controller.set_roi_settings(settings["roi"])
            
            # Apply Hough circle settings
            if "hough" in settings and hasattr(self.ball_tracking_controller, 'set_hough_circle_settings'):
                self.ball_tracking_controller.set_hough_circle_settings(settings["hough"])
            
            # Apply Kalman filter settings
            if "kalman" in settings and hasattr(self.ball_tracking_controller, 'update_kalman_settings'):
                self.ball_tracking_controller.update_kalman_settings(settings["kalman"])
        
        # Update UI to reflect imported data
        if hasattr(self.view, 'update_tracking_view'):
            self.view.update_tracking_view()
        
        self.view.show_info_message("Tracking data imported successfully.")

    @Slot()
    def on_export_calibration(self):
        """
        Handle export calibration button click.
        Opens a dialog for exporting calibration data.
        """
        try:
            if not hasattr(self, 'calibration_controller') or not self.calibration_controller:
                self.view.show_warning_message("Calibration controller not initialized.")
                return
            
            # Get calibration data
            calibration_data = self.calibration_controller.model.to_dict()
            
            # Export as JSON
            self.data_export_controller.export_json(
                calibration_data,
                self.view,
                "Export Calibration Data",
                "calibration_data"
            )
        except Exception as e:
            logging.error(f"Error exporting calibration data: {e}")
            self.view.show_error_message(f"Error exporting calibration data: {str(e)}")

    @Slot()
    def on_import_calibration(self):
        """
        Handle import calibration button click.
        Opens a dialog for importing calibration data.
        """
        try:
            if not hasattr(self, 'calibration_controller') or not self.calibration_controller:
                self.view.show_warning_message("Calibration controller not initialized.")
                return
            
            # Import JSON data
            calibration_data = self.data_export_controller.import_json(
                self.view,
                "Import Calibration Data"
            )
            
            if not calibration_data:
                return
            
            # Validate calibration data format
            if not isinstance(calibration_data, dict) or "points" not in calibration_data:
                self.view.show_warning_message("Invalid calibration data format.")
                return
            
            # Apply calibration data
            self.calibration_controller.model.clear_points()
            self.calibration_controller.model.from_dict(calibration_data)
            
            # Update view
            self.calibration_controller._render_points()
            
            self.view.show_info_message("Calibration data imported successfully.")
        except Exception as e:
            logging.error(f"Error importing calibration data: {e}")
            self.view.show_error_message(f"Error importing calibration data: {str(e)}") 