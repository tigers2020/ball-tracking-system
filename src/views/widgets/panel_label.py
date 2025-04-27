#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Panel Label Widget.
This module contains a customized QLabel for panel displays.
"""

from PySide6.QtCore import Qt, Property
from PySide6.QtGui import QFont, QColor, QPalette
from PySide6.QtWidgets import QLabel


class PanelLabel(QLabel):
    """
    Enhanced label for panel displays with customizable appearance.
    Features include customizable font, color, alignment, and formatting options.
    """
    
    def __init__(self, text="", parent=None):
        """
        Initialize the panel label.
        
        Args:
            text (str, optional): Initial text content
            parent (QWidget, optional): Parent widget
        """
        super().__init__(text, parent)
        
        # Default font and style
        self._init_default_appearance()
        
        # Value and formatting
        self._value = 0.0
        self._precision = 2
        self._prefix = ""
        self._suffix = ""
        self._format = "{:.2f}"
        
    def _init_default_appearance(self):
        """Initialize default appearance settings."""
        # Set default font (monospaced for alignment)
        font = QFont("Consolas", 10)
        font.setBold(True)
        self.setFont(font)
        
        # Set default alignment
        self.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        # Set default color palette
        palette = self.palette()
        palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
        self.setPalette(palette)
        
        # Text properties
        self.setTextFormat(Qt.PlainText)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        # Size policies
        self.setMinimumWidth(80)
        
    def set_value(self, value, precision=None):
        """
        Set the numeric value and update display.
        
        Args:
            value (float): Numeric value to display
            precision (int, optional): Number of decimal places to show
        """
        self._value = value
        
        if precision is not None:
            self._precision = precision
            self._format = "{:." + str(precision) + "f}"
        
        self._update_display()
    
    def _update_display(self):
        """Update the displayed text based on current settings."""
        try:
            # Format the value
            formatted_value = self._format.format(self._value)
            
            # Apply prefix and suffix
            display_text = f"{self._prefix}{formatted_value}{self._suffix}"
            
            # Set the text
            self.setText(display_text)
        except (ValueError, TypeError) as e:
            # Fallback in case of formatting error
            self.setText(f"{self._prefix}{self._value}{self._suffix}")
    
    def set_format(self, format_str):
        """
        Set custom format string for value display.
        
        Args:
            format_str (str): Python format string (e.g., "{:.3f}")
        """
        self._format = format_str
        self._update_display()
    
    def set_prefix(self, prefix):
        """
        Set text prefix to display before the value.
        
        Args:
            prefix (str): Text to display before the value
        """
        self._prefix = prefix
        self._update_display()
    
    def set_suffix(self, suffix):
        """
        Set text suffix to display after the value.
        
        Args:
            suffix (str): Text to display after the value
        """
        self._suffix = suffix
        self._update_display()
    
    def set_color(self, color):
        """
        Set the text color.
        
        Args:
            color (QColor or str): Color for the text
        """
        palette = self.palette()
        
        if isinstance(color, str):
            color = QColor(color)
            
        palette.setColor(QPalette.WindowText, color)
        self.setPalette(palette)
    
    def get_value(self):
        """
        Get the current numeric value.
        
        Returns:
            float: Current numeric value
        """
        return self._value
    
    # Property definitions for use in Qt Designer
    value = Property(float, get_value, set_value) 