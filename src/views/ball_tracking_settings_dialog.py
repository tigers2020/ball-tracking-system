#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ball Tracking Settings Dialog module.
This module contains the BallTrackingSettingsDialog class for configuring HSV mask settings.
"""

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, 
    QPushButton, QGroupBox, QGridLayout, QCheckBox
)

from src.utils.constants import LAYOUT, ROI, HSV, HOUGH, KALMAN
from src.utils.config_manager import ConfigManager
import logging


class BallTrackingSettingsDialog(QDialog):
    """
    Dialog for configuring ball tracking HSV mask settings.
    """
    
    # Signals
    hsv_changed = Signal(dict)
    roi_changed = Signal(dict)
    hough_circle_changed = Signal(dict)
    kalman_changed = Signal(dict)
    
    def __init__(self, parent=None):
        """
        Initialize the ball tracking settings dialog.
        
        Args:
            parent (QWidget, optional): Parent widget
        """
        super(BallTrackingSettingsDialog, self).__init__(parent)
        
        # Set window properties
        self.setWindowTitle("Ball Tracking Settings")
        self.setMinimumWidth(400)
        self.setModal(True)
        
        # Create configuration manager
        self.config_manager = ConfigManager()
        
        # HSV values - load from configuration
        self.hsv_values = self.config_manager.get_hsv_settings()
        
        # Check for missing HSV parameters and set defaults if needed
        hsv_defaults = {
            "h_min": HSV.h_min, "h_max": HSV.h_max,
            "s_min": HSV.s_min, "s_max": HSV.s_max,
            "v_min": HSV.v_min, "v_max": HSV.v_max,
            "blur_size": HSV.blur_size,
            "morph_iterations": HSV.morph_iterations,
            "dilation_iterations": HSV.dilation_iterations
        }
        
        for key, default_value in hsv_defaults.items():
            if key not in self.hsv_values:
                self.hsv_values[key] = default_value
        
        # ROI settings - load from configuration
        self.roi_settings = self.config_manager.get_roi_settings()
        
        # Hough Circle parameters - create default values if not in config
        self.hough_circle_params = self.config_manager.get_hough_circle_settings() if hasattr(self.config_manager, 'get_hough_circle_settings') else {
            "dp": HOUGH.dp,             # Resolution ratio
            "min_dist": HOUGH.min_dist, # Minimum distance between circles
            "param1": HOUGH.param1,     # Higher threshold for edge detection (Canny)
            "param2": HOUGH.param2,     # Threshold for center detection
            "min_radius": HOUGH.min_radius, # Minimum radius
            "max_radius": HOUGH.max_radius  # Maximum radius
        }
        
        # Kalman filter parameters - create default values if not in config
        self.kalman_params = self.config_manager.get_kalman_settings() if hasattr(self.config_manager, 'get_kalman_settings') else {
            "process_noise": KALMAN.process_noise,
            "measurement_noise": KALMAN.measurement_noise
        }
        
        # Check for missing Kalman parameters and set defaults if needed
        kalman_defaults = {
            "process_noise": KALMAN.process_noise,
            "measurement_noise": KALMAN.measurement_noise,
            "max_lost_frames": KALMAN.max_lost_frames,
            "dynamic_process_noise": KALMAN.dynamic_process_noise,
            "adaptive_measurement_noise": KALMAN.adaptive_measurement_noise
        }
        
        for key, default_value in kalman_defaults.items():
            if key not in self.kalman_params:
                self.kalman_params[key] = default_value
        
        # Debounce timer for handling slider changes efficiently
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._update_all_settings)
        
        # Flag to track if settings have changed and need saving
        self._hsv_changed = False
        self._roi_changed = False
        self._hough_changed = False
        self._kalman_changed = False
        
        # Set up UI
        self._setup_ui()
        
    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(LAYOUT.MARGIN, LAYOUT.MARGIN, LAYOUT.MARGIN, LAYOUT.MARGIN)
        main_layout.setSpacing(LAYOUT.SPACING)
        
        # Create HSV sliders group
        hsv_group = QGroupBox("HSV Mask Settings")
        hsv_layout = QGridLayout()
        
        # H sliders (Hue)
        h_min_label = QLabel("H Min:")
        self.h_min_slider = self._create_slider(0, 179, self.hsv_values["h_min"])
        self.h_min_value_label = QLabel(str(self.hsv_values["h_min"]))
        self.h_min_slider.valueChanged.connect(
            lambda v: self._update_hsv_value("h_min", v))
        
        h_max_label = QLabel("H Max:")
        self.h_max_slider = self._create_slider(0, 179, self.hsv_values["h_max"])
        self.h_max_value_label = QLabel(str(self.hsv_values["h_max"]))
        self.h_max_slider.valueChanged.connect(
            lambda v: self._update_hsv_value("h_max", v))
        
        # S sliders (Saturation)
        s_min_label = QLabel("S Min:")
        self.s_min_slider = self._create_slider(0, 255, self.hsv_values["s_min"])
        self.s_min_value_label = QLabel(str(self.hsv_values["s_min"]))
        self.s_min_slider.valueChanged.connect(
            lambda v: self._update_hsv_value("s_min", v))
        
        s_max_label = QLabel("S Max:")
        self.s_max_slider = self._create_slider(0, 255, self.hsv_values["s_max"])
        self.s_max_value_label = QLabel(str(self.hsv_values["s_max"]))
        self.s_max_slider.valueChanged.connect(
            lambda v: self._update_hsv_value("s_max", v))
        
        # V sliders (Value)
        v_min_label = QLabel("V Min:")
        self.v_min_slider = self._create_slider(0, 255, self.hsv_values["v_min"])
        self.v_min_value_label = QLabel(str(self.hsv_values["v_min"]))
        self.v_min_slider.valueChanged.connect(
            lambda v: self._update_hsv_value("v_min", v))
        
        v_max_label = QLabel("V Max:")
        self.v_max_slider = self._create_slider(0, 255, self.hsv_values["v_max"])
        self.v_max_value_label = QLabel(str(self.hsv_values["v_max"]))
        self.v_max_slider.valueChanged.connect(
            lambda v: self._update_hsv_value("v_max", v))
        
        # Blur size slider
        blur_size_label = QLabel("Blur Size:")
        self.blur_size_slider = self._create_slider(1, 15, self.hsv_values.get("blur_size", 3))
        self.blur_size_value_label = QLabel(str(self.hsv_values.get("blur_size", 3)))
        self.blur_size_slider.valueChanged.connect(
            lambda v: self._update_hsv_value("blur_size", v))
        
        hsv_layout.addWidget(h_min_label, 0, 0)
        hsv_layout.addWidget(self.h_min_slider, 0, 1)
        hsv_layout.addWidget(self.h_min_value_label, 0, 2)
        
        hsv_layout.addWidget(h_max_label, 1, 0)
        hsv_layout.addWidget(self.h_max_slider, 1, 1)
        hsv_layout.addWidget(self.h_max_value_label, 1, 2)
        
        hsv_layout.addWidget(s_min_label, 2, 0)
        hsv_layout.addWidget(self.s_min_slider, 2, 1)
        hsv_layout.addWidget(self.s_min_value_label, 2, 2)
        
        hsv_layout.addWidget(s_max_label, 3, 0)
        hsv_layout.addWidget(self.s_max_slider, 3, 1)
        hsv_layout.addWidget(self.s_max_value_label, 3, 2)
        
        hsv_layout.addWidget(v_min_label, 4, 0)
        hsv_layout.addWidget(self.v_min_slider, 4, 1)
        hsv_layout.addWidget(self.v_min_value_label, 4, 2)
        
        hsv_layout.addWidget(v_max_label, 5, 0)
        hsv_layout.addWidget(self.v_max_slider, 5, 1)
        hsv_layout.addWidget(self.v_max_value_label, 5, 2)
        
        # Blur size slider
        blur_size_label = QLabel("Blur Size:")
        self.blur_size_slider = self._create_slider(1, 15, self.hsv_values.get("blur_size", 3))
        self.blur_size_value_label = QLabel(str(self.hsv_values.get("blur_size", 3)))
        self.blur_size_slider.valueChanged.connect(
            lambda v: self._update_hsv_value("blur_size", v))
        
        hsv_layout.addWidget(blur_size_label, 6, 0)
        hsv_layout.addWidget(self.blur_size_slider, 6, 1)
        hsv_layout.addWidget(self.blur_size_value_label, 6, 2)
        
        # Morph iterations slider
        morph_label = QLabel("Morph Iterations:")
        self.morph_slider = self._create_slider(0, 10, self.hsv_values.get("morph_iterations", 2))
        self.morph_value_label = QLabel(str(self.hsv_values.get("morph_iterations", 2)))
        self.morph_slider.valueChanged.connect(
            lambda v: self._update_hsv_value("morph_iterations", v))
        
        hsv_layout.addWidget(morph_label, 7, 0)
        hsv_layout.addWidget(self.morph_slider, 7, 1)
        hsv_layout.addWidget(self.morph_value_label, 7, 2)
        
        # Dilation iterations slider
        dilation_label = QLabel("Dilation Iterations:")
        self.dilation_slider = self._create_slider(0, 10, self.hsv_values.get("dilation_iterations", 1))
        self.dilation_value_label = QLabel(str(self.hsv_values.get("dilation_iterations", 1)))
        self.dilation_slider.valueChanged.connect(
            lambda v: self._update_hsv_value("dilation_iterations", v))
        
        hsv_layout.addWidget(dilation_label, 8, 0)
        hsv_layout.addWidget(self.dilation_slider, 8, 1)
        hsv_layout.addWidget(self.dilation_value_label, 8, 2)
        
        hsv_group.setLayout(hsv_layout)
        main_layout.addWidget(hsv_group)
        
        # Create ROI settings group
        roi_group = QGroupBox("ROI Settings")
        roi_layout = QGridLayout()
        
        # ROI enable checkbox
        self.roi_enabled_checkbox = QCheckBox("Enable ROI")
        self.roi_enabled_checkbox.setChecked(self.roi_settings["enabled"])
        self.roi_enabled_checkbox.stateChanged.connect(
            lambda state: self._update_roi_value("enabled", state == Qt.Checked))
        roi_layout.addWidget(self.roi_enabled_checkbox, 0, 0, 1, 3)
        
        # Auto-center checkbox
        self.auto_center_checkbox = QCheckBox("Auto-center on detected object")
        self.auto_center_checkbox.setChecked(self.roi_settings["auto_center"])
        self.auto_center_checkbox.stateChanged.connect(
            lambda state: self._update_roi_value("auto_center", state == Qt.Checked))
        roi_layout.addWidget(self.auto_center_checkbox, 1, 0, 1, 3)
        
        # ROI width slider
        width_label = QLabel("Width:")
        self.width_slider = self._create_slider(ROI.MIN_SIZE, ROI.MAX_SIZE, self.roi_settings["width"])
        self.width_value_label = QLabel(str(self.roi_settings["width"]))
        self.width_slider.valueChanged.connect(
            lambda v: self._update_roi_value("width", v))
        
        roi_layout.addWidget(width_label, 2, 0)
        roi_layout.addWidget(self.width_slider, 2, 1)
        roi_layout.addWidget(self.width_value_label, 2, 2)
        
        # ROI height slider
        height_label = QLabel("Height:")
        self.height_slider = self._create_slider(ROI.MIN_SIZE, ROI.MAX_SIZE, self.roi_settings["height"])
        self.height_value_label = QLabel(str(self.roi_settings["height"]))
        self.height_slider.valueChanged.connect(
            lambda v: self._update_roi_value("height", v))
        
        roi_layout.addWidget(height_label, 3, 0)
        roi_layout.addWidget(self.height_slider, 3, 1)
        roi_layout.addWidget(self.height_value_label, 3, 2)
        
        roi_group.setLayout(roi_layout)
        main_layout.addWidget(roi_group)
        
        # Create Hough Circle parameters group
        hough_group = QGroupBox("Hough Circle Detection")
        hough_layout = QGridLayout()
        
        # Resolution parameter (dp)
        dp_label = QLabel("Resolution (dp):")
        self.dp_slider = self._create_slider(10, 30, int(self.hough_circle_params["dp"] * 10))
        self.dp_value_label = QLabel(str(self.hough_circle_params["dp"]))
        self.dp_slider.valueChanged.connect(
            lambda v: self._update_hough_value("dp", v / 10.0))
            
        # Min distance parameter
        min_dist_label = QLabel("Min Distance:")
        self.min_dist_slider = self._create_slider(10, 200, self.hough_circle_params["min_dist"])
        self.min_dist_value_label = QLabel(str(self.hough_circle_params["min_dist"]))
        self.min_dist_slider.valueChanged.connect(
            lambda v: self._update_hough_value("min_dist", v))
            
        # Edge threshold parameter (param1)
        param1_label = QLabel("Edge Threshold:")
        self.param1_slider = self._create_slider(10, 300, self.hough_circle_params["param1"])
        self.param1_value_label = QLabel(str(self.hough_circle_params["param1"]))
        self.param1_slider.valueChanged.connect(
            lambda v: self._update_hough_value("param1", v))
            
        # Center threshold parameter (param2)
        param2_label = QLabel("Center Threshold:")
        self.param2_slider = self._create_slider(1, 100, self.hough_circle_params["param2"])
        self.param2_value_label = QLabel(str(self.hough_circle_params["param2"]))
        self.param2_slider.valueChanged.connect(
            lambda v: self._update_hough_value("param2", v))
            
        # Min radius parameter
        min_radius_label = QLabel("Min Radius:")
        self.min_radius_slider = self._create_slider(1, 100, self.hough_circle_params["min_radius"])
        self.min_radius_value_label = QLabel(str(self.hough_circle_params["min_radius"]))
        self.min_radius_slider.valueChanged.connect(
            lambda v: self._update_hough_value("min_radius", v))
            
        # Max radius parameter
        max_radius_label = QLabel("Max Radius:")
        self.max_radius_slider = self._create_slider(10, 300, self.hough_circle_params["max_radius"])
        self.max_radius_value_label = QLabel(str(self.hough_circle_params["max_radius"]))
        self.max_radius_slider.valueChanged.connect(
            lambda v: self._update_hough_value("max_radius", v))
            
        # Add widgets to grid layout
        hough_layout.addWidget(dp_label, 0, 0)
        hough_layout.addWidget(self.dp_slider, 0, 1)
        hough_layout.addWidget(self.dp_value_label, 0, 2)
        
        hough_layout.addWidget(min_dist_label, 1, 0)
        hough_layout.addWidget(self.min_dist_slider, 1, 1)
        hough_layout.addWidget(self.min_dist_value_label, 1, 2)
        
        hough_layout.addWidget(param1_label, 2, 0)
        hough_layout.addWidget(self.param1_slider, 2, 1)
        hough_layout.addWidget(self.param1_value_label, 2, 2)
        
        hough_layout.addWidget(param2_label, 3, 0)
        hough_layout.addWidget(self.param2_slider, 3, 1)
        hough_layout.addWidget(self.param2_value_label, 3, 2)
        
        hough_layout.addWidget(min_radius_label, 4, 0)
        hough_layout.addWidget(self.min_radius_slider, 4, 1)
        hough_layout.addWidget(self.min_radius_value_label, 4, 2)
        
        hough_layout.addWidget(max_radius_label, 5, 0)
        hough_layout.addWidget(self.max_radius_slider, 5, 1)
        hough_layout.addWidget(self.max_radius_value_label, 5, 2)
        
        hough_group.setLayout(hough_layout)
        main_layout.addWidget(hough_group)
        
        # Create Kalman filter parameters group
        kalman_group = QGroupBox("Kalman Filter Settings")
        kalman_layout = QGridLayout()
        
        # Process noise parameter
        process_noise_label = QLabel("Process Noise:")
        self.process_noise_slider = self._create_slider(1, 100, int(self.kalman_params["process_noise"] * 1000))
        self.process_noise_value_label = QLabel(str(self.kalman_params["process_noise"]))
        self.process_noise_slider.valueChanged.connect(
            lambda v: self._update_kalman_value("process_noise", v / 1000.0))
            
        # Measurement noise parameter
        measurement_noise_label = QLabel("Measurement Noise:")
        self.measurement_noise_slider = self._create_slider(1, 100, int(self.kalman_params["measurement_noise"] * 10))
        self.measurement_noise_value_label = QLabel(str(self.kalman_params["measurement_noise"]))
        self.measurement_noise_slider.valueChanged.connect(
            lambda v: self._update_kalman_value("measurement_noise", v / 10.0))
            
        # Max lost frames parameter
        max_lost_frames_label = QLabel("Max Lost Frames:")
        self.max_lost_frames_slider = self._create_slider(1, 50, self.kalman_params.get("max_lost_frames", 20))
        self.max_lost_frames_value_label = QLabel(str(self.kalman_params.get("max_lost_frames", 20)))
        self.max_lost_frames_slider.valueChanged.connect(
            lambda v: self._update_kalman_value("max_lost_frames", v))
        
        # Dynamic process noise checkbox
        self.dynamic_noise_checkbox = QCheckBox("Dynamic Process Noise")
        self.dynamic_noise_checkbox.setChecked(self.kalman_params.get("dynamic_process_noise", True))
        self.dynamic_noise_checkbox.stateChanged.connect(
            lambda state: self._update_kalman_value("dynamic_process_noise", state == Qt.Checked))
        
        # Adaptive measurement noise checkbox
        self.adaptive_measurement_checkbox = QCheckBox("Adaptive Measurement Noise")
        self.adaptive_measurement_checkbox.setChecked(self.kalman_params.get("adaptive_measurement_noise", True))
        self.adaptive_measurement_checkbox.stateChanged.connect(
            lambda state: self._update_kalman_value("adaptive_measurement_noise", state == Qt.Checked))
        
        # Add widgets to grid layout
        kalman_layout.addWidget(process_noise_label, 0, 0)
        kalman_layout.addWidget(self.process_noise_slider, 0, 1)
        kalman_layout.addWidget(self.process_noise_value_label, 0, 2)
        
        kalman_layout.addWidget(measurement_noise_label, 1, 0)
        kalman_layout.addWidget(self.measurement_noise_slider, 1, 1)
        kalman_layout.addWidget(self.measurement_noise_value_label, 1, 2)
        
        kalman_layout.addWidget(max_lost_frames_label, 2, 0)
        kalman_layout.addWidget(self.max_lost_frames_slider, 2, 1)
        kalman_layout.addWidget(self.max_lost_frames_value_label, 2, 2)
        
        kalman_layout.addWidget(self.dynamic_noise_checkbox, 3, 0, 1, 3)
        kalman_layout.addWidget(self.adaptive_measurement_checkbox, 4, 0, 1, 3)
        
        kalman_group.setLayout(kalman_layout)
        main_layout.addWidget(kalman_group)
        
        # Add buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        self.close_button.setDefault(True)  # Set as default button (responds to Enter key)
        button_layout.addWidget(self.close_button)
        
        main_layout.addLayout(button_layout)
    
    def _create_slider(self, min_value, max_value, initial_value):
        """
        Create a slider with the specified range and initial value.
        
        Args:
            min_value (int): Minimum slider value
            max_value (int): Maximum slider value
            initial_value (int): Initial slider value
            
        Returns:
            QSlider: Configured slider
        """
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_value)
        slider.setMaximum(max_value)
        slider.setValue(initial_value)
        slider.setTracking(True)
        return slider
    
    def _update_hsv_value(self, key, value):
        """
        Update HSV value and emit change signal.
        
        Args:
            key (str): HSV parameter key
            value (int): New value
        """
        self.hsv_values[key] = value
        
        # Update the value label
        if key == "h_min":
            self.h_min_value_label.setText(str(value))
        elif key == "h_max":
            self.h_max_value_label.setText(str(value))
        elif key == "s_min":
            self.s_min_value_label.setText(str(value))
        elif key == "s_max":
            self.s_max_value_label.setText(str(value))
        elif key == "v_min":
            self.v_min_value_label.setText(str(value))
        elif key == "v_max":
            self.v_max_value_label.setText(str(value))
        elif key == "blur_size":
            self.blur_size_value_label.setText(str(value))
        elif key == "morph_iterations":
            self.morph_value_label.setText(str(value))
        elif key == "dilation_iterations":
            self.dilation_value_label.setText(str(value))
        
        # Mark HSV settings as changed
        self._hsv_changed = True
        
        # Emit signal with updated HSV values
        self.hsv_changed.emit(self.hsv_values)
        
        # Restart debounce timer
        self._debounce_timer.start(300)  # 300ms debounce
    
    def _update_roi_value(self, key, value):
        """
        Update ROI setting value and emit change signal.
        
        Args:
            key (str): ROI setting key
            value: New value
        """
        self.roi_settings[key] = value
        
        # Update the value label if needed
        if key == "width":
            self.width_value_label.setText(str(value))
        elif key == "height":
            self.height_value_label.setText(str(value))
        
        # Enable/disable other ROI controls based on enabled state
        if key == "enabled":
            self.width_slider.setEnabled(value)
            self.height_slider.setEnabled(value)
            self.auto_center_checkbox.setEnabled(value)
        
        # Mark ROI settings as changed
        self._roi_changed = True
        
        # Emit signal with updated ROI settings
        self.roi_changed.emit(self.roi_settings)
        
        # Restart debounce timer
        self._debounce_timer.start(300)  # 300ms debounce
    
    def _update_hough_value(self, key, value):
        """
        Update Hough Circle parameter value and emit change signal.
        
        Args:
            key (str): Hough Circle parameter key
            value: New value
        """
        self.hough_circle_params[key] = value
        
        # Update the value label
        if key == "dp":
            self.dp_value_label.setText(str(value))
        elif key == "min_dist":
            self.min_dist_value_label.setText(str(value))
        elif key == "param1":
            self.param1_value_label.setText(str(value))
        elif key == "param2":
            self.param2_value_label.setText(str(value))
        elif key == "min_radius":
            self.min_radius_value_label.setText(str(value))
        elif key == "max_radius":
            self.max_radius_value_label.setText(str(value))
        
        # Mark Hough settings as changed
        self._hough_changed = True
        
        # Emit signal with updated Hough Circle parameters
        self.hough_circle_changed.emit(self.hough_circle_params)
        
        # Restart debounce timer
        self._debounce_timer.start(300)  # 300ms debounce
    
    def _update_kalman_value(self, key, value):
        """
        Update Kalman filter parameter value and emit change signal.
        
        Args:
            key (str): Kalman filter parameter key
            value: New value
        """
        self.kalman_params[key] = value
        
        # Update the value label
        if key == "process_noise":
            self.process_noise_value_label.setText(str(value))
        elif key == "measurement_noise":
            self.measurement_noise_value_label.setText(str(value))
        elif key == "max_lost_frames":
            self.max_lost_frames_value_label.setText(str(value))
        elif key == "dynamic_process_noise":
            self.dynamic_noise_checkbox.setChecked(value)
        elif key == "adaptive_measurement_noise":
            self.adaptive_measurement_checkbox.setChecked(value)
        
        # Mark Kalman settings as changed
        self._kalman_changed = True
        
        # Emit signal with updated Kalman filter parameters
        self.kalman_changed.emit(self.kalman_params)
        
        # Restart debounce timer
        self._debounce_timer.start(300)  # 300ms debounce
    
    def _update_all_settings(self):
        """Save all changed settings to configuration after debounce period."""
        logging.debug("Saving settings after slider changes")
        
        if self._hsv_changed:
            self.config_manager.set_hsv_settings(self.hsv_values)
            self._hsv_changed = False
            
        if self._roi_changed:
            self.config_manager.set_roi_settings(self.roi_settings)
            self._roi_changed = False
            
        if self._hough_changed:
            self.config_manager.set_hough_circle_settings(self.hough_circle_params)
            self._hough_changed = False
            
        if self._kalman_changed:
            self.config_manager.set_kalman_settings(self.kalman_params)
            self._kalman_changed = False
    
    def get_hsv_values(self):
        """
        Get the current HSV values.
        
        Returns:
            dict: Current HSV values
        """
        return self.hsv_values
    
    def get_roi_settings(self):
        """
        Get the current ROI settings.
        
        Returns:
            dict: Current ROI settings
        """
        return self.roi_settings
    
    def get_hough_circle_params(self):
        """
        Get the current Hough Circle parameters.
        
        Returns:
            dict: Current Hough Circle parameters
        """
        return self.hough_circle_params
    
    def get_kalman_params(self):
        """
        Get the current Kalman filter parameters.
        
        Returns:
            dict: Current Kalman filter parameters
        """
        return self.kalman_params
    
    def set_hsv_values(self, hsv_values):
        """
        Set HSV values and update sliders.
        
        Args:
            hsv_values (dict): HSV values to set
        """
        for key, value in hsv_values.items():
            if key in self.hsv_values:
                self.hsv_values[key] = value
                
                # Update slider and label
                if key == "h_min":
                    self.h_min_slider.setValue(value)
                    self.h_min_value_label.setText(str(value))
                elif key == "h_max":
                    self.h_max_slider.setValue(value)
                    self.h_max_value_label.setText(str(value))
                elif key == "s_min":
                    self.s_min_slider.setValue(value)
                    self.s_min_value_label.setText(str(value))
                elif key == "s_max":
                    self.s_max_slider.setValue(value)
                    self.s_max_value_label.setText(str(value))
                elif key == "v_min":
                    self.v_min_slider.setValue(value)
                    self.v_min_value_label.setText(str(value))
                elif key == "v_max":
                    self.v_max_slider.setValue(value)
                    self.v_max_value_label.setText(str(value))
                elif key == "blur_size":
                    self.blur_size_slider.setValue(value)
                    self.blur_size_value_label.setText(str(value))
                elif key == "morph_iterations":
                    self.morph_slider.setValue(value)
                    self.morph_value_label.setText(str(value))
                elif key == "dilation_iterations":
                    self.dilation_slider.setValue(value)
                    self.dilation_value_label.setText(str(value))
    
    def set_roi_settings(self, roi_settings):
        """
        Set ROI settings and update UI controls.
        
        Args:
            roi_settings (dict): ROI settings to set
        """
        for key, value in roi_settings.items():
            if key in self.roi_settings:
                self.roi_settings[key] = value
                
                # Update controls
                if key == "enabled":
                    self.roi_enabled_checkbox.setChecked(value)
                    # Update dependent controls state
                    self.width_slider.setEnabled(value)
                    self.height_slider.setEnabled(value)
                    self.auto_center_checkbox.setEnabled(value)
                elif key == "auto_center":
                    self.auto_center_checkbox.setChecked(value)
                elif key == "width":
                    self.width_slider.setValue(value)
                    self.width_value_label.setText(str(value))
                elif key == "height":
                    self.height_slider.setValue(value)
                    self.height_value_label.setText(str(value))
    
    def set_hough_circle_params(self, hough_circle_params):
        """
        Set Hough Circle parameters and update UI controls.
        
        Args:
            hough_circle_params (dict): Hough Circle parameters to set
        """
        for key, value in hough_circle_params.items():
            if key in self.hough_circle_params:
                self.hough_circle_params[key] = value
                
                # Update controls
                if key == "dp":
                    self.dp_slider.setValue(int(value * 10))
                    self.dp_value_label.setText(str(value))
                elif key == "min_dist":
                    self.min_dist_slider.setValue(value)
                    self.min_dist_value_label.setText(str(value))
                elif key == "param1":
                    self.param1_slider.setValue(value)
                    self.param1_value_label.setText(str(value))
                elif key == "param2":
                    self.param2_slider.setValue(value)
                    self.param2_value_label.setText(str(value))
                elif key == "min_radius":
                    self.min_radius_slider.setValue(value)
                    self.min_radius_value_label.setText(str(value))
                elif key == "max_radius":
                    self.max_radius_slider.setValue(value)
                    self.max_radius_value_label.setText(str(value))
    
    def set_kalman_params(self, kalman_params):
        """
        Set Kalman filter parameters and update UI controls.
        
        Args:
            kalman_params (dict): Kalman filter parameters to set
        """
        for key, value in kalman_params.items():
            if key in self.kalman_params:
                self.kalman_params[key] = value
                
                # Update controls
                if key == "process_noise":
                    self.process_noise_slider.setValue(int(value * 1000))
                    self.process_noise_value_label.setText(str(value))
                elif key == "measurement_noise":
                    self.measurement_noise_slider.setValue(int(value * 10))
                    self.measurement_noise_value_label.setText(str(value))
    
    def closeEvent(self, event):
        """
        Handle close event to save settings.
        
        Args:
            event (QCloseEvent): Close event
        """
        # Save current settings to configuration
        self._save_all_settings()
        super(BallTrackingSettingsDialog, self).closeEvent(event)
        
    def accept(self):
        """Handle dialog accept (OK button)."""
        # Save current settings to configuration
        self._save_all_settings()
        super(BallTrackingSettingsDialog, self).accept()
        
    def _save_all_settings(self):
        """Save all settings to configuration."""
        logging.info("Saving all settings to configuration")
        self.config_manager.set_hsv_settings(self.hsv_values)
        self.config_manager.set_roi_settings(self.roi_settings)
        self.config_manager.set_hough_circle_settings(self.hough_circle_params)
        self.config_manager.set_kalman_settings(self.kalman_params) 