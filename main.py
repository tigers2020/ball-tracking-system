#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import logging
from pathlib import Path

# Add the src directory to the path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

from src.controllers.app_controller import AppController
from src.utils.ui_theme import ThemeManager

if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    
    # Create the application
    app = QApplication(sys.argv)
    app.setApplicationName("Stereo Image Player")
    
    # Apply theme
    ThemeManager.apply_theme(app)
    
    # Create and show the main controller
    controller = AppController()
    controller.show()
    
    # Run the application
    sys.exit(app.exec())
