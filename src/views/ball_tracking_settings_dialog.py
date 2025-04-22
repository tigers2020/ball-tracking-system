#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ball Tracking Settings Dialog module.
This module contains the BallTrackingSettingsDialog class for configuring HSV mask settings.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, 
    QPushButton, QGroupBox, QGridLayout
)

from src.utils.ui_constants import Layout
from src.utils.config_manager import ConfigManager


class BallTrackingSettingsDialog(QDialog):
    """
    Dialog for configuring ball tracking HSV mask settings.
    """
    
    # Signals
    hsv_changed = Signal(dict)
    
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
        
        # Set up UI
        self._setup_ui()
        
    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(Layout.MARGIN, Layout.MARGIN, Layout.MARGIN, Layout.MARGIN)
        main_layout.setSpacing(Layout.SPACING)
        
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
        
        # Add widgets to grid layout
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
        
        hsv_group.setLayout(hsv_layout)
        main_layout.addWidget(hsv_group)
        
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
        
        # Emit signal with updated HSV values
        self.hsv_changed.emit(self.hsv_values)
    
    def get_hsv_values(self):
        """
        Get the current HSV values.
        
        Returns:
            dict: Current HSV values
        """
        return self.hsv_values
    
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

    def closeEvent(self, event):
        """
        Handle close event to save settings.
        
        Args:
            event (QCloseEvent): Close event
        """
        # Save current HSV values to configuration
        self.config_manager.set_hsv_settings(self.hsv_values)
        super(BallTrackingSettingsDialog, self).closeEvent(event)
        
    def accept(self):
        """Handle dialog accept (OK button)."""
        # Save current HSV values to configuration
        self.config_manager.set_hsv_settings(self.hsv_values)
        super(BallTrackingSettingsDialog, self).accept() 