#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Resources package for the Stereo Image Player application.
This package contains resource modules for the application.
"""

import os
import logging

# Import the icons module to ensure icons are created
from src.resources import icons

# Run the ensure_icons_exist function to create icons on import
if not os.path.exists(os.path.join("src", "resources", "play.png")):
    logging.info("Creating icon resources")
    icons.ensure_icons_exist() 