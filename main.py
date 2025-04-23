#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import logging
import time
from datetime import datetime
from pathlib import Path

# Add the src directory to the path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

# log Storage Directory generation Function
def create_log_directories():
    log_dirs = ["debug", "info", "warning", "error", "critical"]
    for directory in log_dirs:
        log_path = Path(ROOT_DIR) / "logs" / directory
        log_path.mkdir(parents=True, exist_ok=True)
    return log_dirs

# today hour Foundation Times stamp generation
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# log Directory generation
log_levels = create_log_directories()

# Level log filter class
class LevelFilter(logging.Filter):
    def __init__(self, level):
        super().__init__()
        self.level = level
    
    def filter(self, record):
        return record.levelno == self.level

# basic log setting (Route Ruin)
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)  # every Level treatment As possible setting

# console Handler addition
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # In the console INFO Abnormalities output of power
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
root_logger.addHandler(console_handler)

# Level file Handler definition
level_configs = {
    "debug": (logging.DEBUG, "Debug Level message"),
    "info": (logging.INFO, "information Level message"),
    "warning": (logging.WARNING, "warning Level message"),
    "error": (logging.ERROR, "error Level message"),
    "critical": (logging.CRITICAL, "acute error message")
}

# each Level file Handler generation and addition
for level_name, (level_value, description) in level_configs.items():
    log_file = Path(ROOT_DIR) / "logs" / level_name / f"{level_name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level_value)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # corresponding Level To save filter addition
    level_filter = LevelFilter(level_value)
    file_handler.addFilter(level_filter)
    
    root_logger.addHandler(file_handler)
    
    # beginning log addition
    logger = logging.getLogger(f"setup.{level_name}")
    logger.log(level_value, f"{description} Logging start - file: {log_file}")

# reset complete log
logging.info(f"log System reset complete. Times stamp: {timestamp}")

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
