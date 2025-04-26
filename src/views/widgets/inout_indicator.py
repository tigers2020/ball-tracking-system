#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IN/OUT indicator widget module.
Provides a visual LED indicator for showing if a ball bounce is in or out.
"""

from PySide6.QtWidgets import QLabel, QHBoxLayout, QWidget
from PySide6.QtCore import Slot, Qt, QTimer
from PySide6.QtGui import QFont


class InOutLED(QWidget):
    """
    LED indicator widget for displaying if a ball bounce is in (green) or out (red).
    """
    
    def __init__(self, parent=None):
        """
        Initialize the LED indicator.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Create layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Create LED indicator
        self.led = QLabel(self)
        self.led.setFixedSize(24, 24)
        self.led.setStyleSheet("border-radius: 12px; background-color: #666666;")
        self.led.setToolTip("IN/OUT indicator")
        
        # Create text label
        self.text_label = QLabel("--", self)
        self.text_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.text_label.setFont(QFont("Arial", 10, QFont.Bold))
        
        # Add to layout
        layout.addWidget(self.led)
        layout.addWidget(self.text_label)
        
        # Setup auto-reset timer
        self.reset_timer = QTimer(self)
        self.reset_timer.timeout.connect(self.reset)
        self.reset_timer.setSingleShot(True)
        
        # Initial state
        self.reset()

    @Slot(bool)
    def on_in_out(self, is_in):
        """
        Update the LED color based on the in/out status.
        
        Args:
            is_in: True if the ball bounce is inside the court, False otherwise
        """
        if is_in:
            color = "#00e000"  # Green
            text = "IN"
        else:
            color = "#e00000"  # Red
            text = "OUT"
            
        # Update LED and text
        self.led.setStyleSheet(f"border-radius: 12px; background-color: {color};")
        self.text_label.setText(text)
        self.text_label.setStyleSheet(f"color: {color};")
        
        # Reset after 3 seconds
        self.reset_timer.start(3000)
    
    def reset(self):
        """Reset the indicator to its default state."""
        self.led.setStyleSheet("border-radius: 12px; background-color: #666666;")
        self.text_label.setText("--")
        self.text_label.setStyleSheet("color: #666666;") 