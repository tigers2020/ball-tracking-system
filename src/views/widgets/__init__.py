#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Widgets Package.
This package contains custom UI widgets for the application.
"""

# Import common widgets to make them available directly from the package
from src.views.widgets.panel_label import PanelLabel
from src.views.widgets.inout_indicator import InOutLED

__all__ = ['PanelLabel', 'InOutLED'] 