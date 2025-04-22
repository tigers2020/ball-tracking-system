#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Playback Controls Widget module.
This module contains the PlaybackControlsWidget class for controlling playback in the Stereo Image Player.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSlider, QPushButton, 
    QLabel, QSpinBox, QSizePolicy, QCheckBox
)

from src.utils.ui_constants import Layout, Timing, Icons


class PlaybackControlsWidget(QWidget):
    """
    Widget containing controls for playback of stereo images.
    """
    
    # Signals
    play_clicked = Signal()
    pause_clicked = Signal()
    stop_clicked = Signal()
    next_clicked = Signal()
    prev_clicked = Signal()
    frame_changed = Signal(int)
    fps_changed = Signal(int)
    skip_frames_changed = Signal(bool)
    
    def __init__(self, parent=None):
        """
        Initialize the playback controls widget.
        
        Args:
            parent (QWidget, optional): Parent widget
        """
        super(PlaybackControlsWidget, self).__init__(parent)
        
        # Current state
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0
        self.skip_frames = False
        
        # Set up UI
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(Layout.MARGIN, Layout.MARGIN, Layout.MARGIN, Layout.MARGIN)
        main_layout.setSpacing(Layout.SPACING)
        
        # Slider for frame navigation
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)  # Will be updated when frames are loaded
        self.frame_slider.setValue(0)
        self.frame_slider.setTracking(True)
        self.frame_slider.setFixedHeight(Layout.SLIDER_HEIGHT)
        self.frame_slider.valueChanged.connect(self._on_slider_value_changed)
        main_layout.addWidget(self.frame_slider)
        
        # Frame counter layout
        frame_counter_layout = QHBoxLayout()
        frame_counter_layout.setSpacing(Layout.SPACING)
        
        self.frame_label = QLabel("Frame: 0 / 0")
        frame_counter_layout.addWidget(self.frame_label)
        
        frame_counter_layout.addStretch()
        
        fps_label = QLabel("FPS:")
        frame_counter_layout.addWidget(fps_label)
        
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setMinimum(Timing.MIN_FPS)
        self.fps_spinbox.setMaximum(Timing.MAX_FPS)
        self.fps_spinbox.setValue(Timing.DEFAULT_FPS)
        self.fps_spinbox.valueChanged.connect(self._on_fps_changed)
        frame_counter_layout.addWidget(self.fps_spinbox)
        
        main_layout.addLayout(frame_counter_layout)
        
        # Control buttons layout
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(Layout.SPACING)
        
        # Skip frames checkbox
        self.skip_frames_checkbox = QCheckBox("Skip 16 frames")
        self.skip_frames_checkbox.setToolTip("Skip 16 frames during playback")
        self.skip_frames_checkbox.setChecked(self.skip_frames)
        self.skip_frames_checkbox.stateChanged.connect(self._on_skip_frames_changed)
        controls_layout.addWidget(self.skip_frames_checkbox)
        
        controls_layout.addStretch()
        
        # Previous button
        self.prev_button = QPushButton()
        self.prev_button.setIcon(QIcon(Icons.PREV))
        self.prev_button.setToolTip("Previous Frame")
        self.prev_button.setFixedHeight(Layout.BUTTON_HEIGHT)
        self.prev_button.clicked.connect(self._on_prev_clicked)
        controls_layout.addWidget(self.prev_button)
        
        # Play button
        self.play_button = QPushButton()
        self.play_button.setIcon(QIcon(Icons.PLAY))
        self.play_button.setToolTip("Play")
        self.play_button.setFixedHeight(Layout.BUTTON_HEIGHT)
        self.play_button.clicked.connect(self._on_play_clicked)
        controls_layout.addWidget(self.play_button)
        
        # Stop button
        self.stop_button = QPushButton()
        self.stop_button.setIcon(QIcon(Icons.STOP))
        self.stop_button.setToolTip("Stop")
        self.stop_button.setFixedHeight(Layout.BUTTON_HEIGHT)
        self.stop_button.clicked.connect(self._on_stop_clicked)
        controls_layout.addWidget(self.stop_button)
        
        # Next button
        self.next_button = QPushButton()
        self.next_button.setIcon(QIcon(Icons.NEXT))
        self.next_button.setToolTip("Next Frame")
        self.next_button.setFixedHeight(Layout.BUTTON_HEIGHT)
        self.next_button.clicked.connect(self._on_next_clicked)
        controls_layout.addWidget(self.next_button)
        
        main_layout.addLayout(controls_layout)
        
        # Set size policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(Layout.CONTROLS_HEIGHT * 2)
    
    def _on_play_clicked(self):
        """Handle play button click."""
        if self.is_playing:
            self.is_playing = False
            self.play_button.setIcon(QIcon(Icons.PLAY))
            self.play_button.setToolTip("Play")
            self.pause_clicked.emit()
        else:
            self.is_playing = True
            self.play_button.setIcon(QIcon(Icons.PAUSE))
            self.play_button.setToolTip("Pause")
            self.play_clicked.emit()
    
    def _on_stop_clicked(self):
        """Handle stop button click."""
        self.is_playing = False
        self.play_button.setIcon(QIcon(Icons.PLAY))
        self.play_button.setToolTip("Play")
        self.stop_clicked.emit()
    
    def _on_next_clicked(self):
        """Handle next button click."""
        self.next_clicked.emit()
    
    def _on_prev_clicked(self):
        """Handle previous button click."""
        self.prev_clicked.emit()
    
    def _on_slider_value_changed(self, value):
        """
        Handle slider value change.
        
        Args:
            value (int): New slider value
        """
        if value != self.current_frame:
            self.current_frame = value
            self.frame_changed.emit(value)
            self._update_frame_label()
    
    def _on_fps_changed(self, value):
        """
        Handle FPS value change.
        
        Args:
            value (int): New FPS value
        """
        self.fps_changed.emit(value)
    
    def _on_skip_frames_changed(self, state):
        """
        Handle skip frames checkbox state change.
        
        Args:
            state (int): Checkbox state (Qt.Checked or Qt.Unchecked)
        """
        is_checked = (state == Qt.Checked)
        if self.skip_frames != is_checked:
            self.skip_frames = is_checked
            self.skip_frames_changed.emit(self.skip_frames)
    
    def _update_frame_label(self):
        """Update the frame label text."""
        self.frame_label.setText(f"Frame: {self.current_frame + 1} / {self.total_frames}")
    
    def set_frame(self, frame_index):
        """
        Set the current frame index.
        
        Args:
            frame_index (int): Frame index
        """
        if 0 <= frame_index < self.total_frames:
            self.current_frame = frame_index
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(frame_index)
            self.frame_slider.blockSignals(False)
            self._update_frame_label()
    
    def set_total_frames(self, total_frames):
        """
        Set the total number of frames.
        
        Args:
            total_frames (int): Total number of frames
        """
        self.total_frames = total_frames
        self.frame_slider.setMaximum(max(0, total_frames - 1))
        self._update_frame_label()
    
    def get_current_fps(self):
        """
        Get the current FPS setting.
        
        Returns:
            int: Current FPS value
        """
        return self.fps_spinbox.value()
    
    def is_skipping_frames(self):
        """
        Get the current state of frame skipping.
        
        Returns:
            bool: True if frames should be skipped, False otherwise
        """
        return self.skip_frames_checkbox.isChecked()
    
    def set_is_playing(self, is_playing):
        """
        Set the playing state.
        
        Args:
            is_playing (bool): True if playing, False otherwise
        """
        if is_playing != self.is_playing:
            self.is_playing = is_playing
            if is_playing:
                self.play_button.setIcon(QIcon(Icons.PAUSE))
                self.play_button.setToolTip("Pause")
            else:
                self.play_button.setIcon(QIcon(Icons.PLAY))
                self.play_button.setToolTip("Play")
    
    def enable_controls(self, enable=True):
        """
        Enable or disable all controls.
        
        Args:
            enable (bool): True to enable, False to disable
        """
        self.frame_slider.setEnabled(enable)
        self.play_button.setEnabled(enable)
        self.stop_button.setEnabled(enable)
        self.next_button.setEnabled(enable)
        self.prev_button.setEnabled(enable)
        self.fps_spinbox.setEnabled(enable)
        self.skip_frames_checkbox.setEnabled(enable) 