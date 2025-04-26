#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IN/OUT indicator widget module.
Provides a visual LED indicator for showing if a ball bounce is in or out.
"""

from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Slot


class InOutLED(QLabel):
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
        self.setFixedSize(24, 24)
        self.setStyleSheet("border-radius: 12px; background-color: #666666;")
        self.setToolTip("IN/OUT indicator")

    @Slot(bool)
    def on_in_out(self, is_in):
        """
        Update the LED color based on the in/out status.
        
        Args:
            is_in: True if the ball bounce is inside the court, False otherwise
        """
        color = "#00e000" if is_in else "#e00000"
        self.setStyleSheet(f"border-radius: 12px; background-color: {color};") 