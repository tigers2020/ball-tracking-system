#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logging Utilities Module
This module contains common logging utility functions to reduce code duplication.
"""

import logging
from typing import Dict, Any, Optional


def log_service_init(service_name: str, settings: Dict[str, Any], log_level: int = logging.INFO) -> None:
    """
    Log service initialization with standardized format.
    
    Args:
        service_name: Name of the service being initialized
        settings: Dictionary containing service settings
        log_level: Logging level (default: logging.INFO)
    """
    logging.log(log_level, f"{service_name} initialized with settings: {settings}")


def log_service_update(service_name: str, settings: Dict[str, Any], log_level: int = logging.INFO) -> None:
    """
    Log service settings update with standardized format.
    
    Args:
        service_name: Name of the service being updated
        settings: Dictionary containing updated service settings
        log_level: Logging level (default: logging.INFO)
    """
    logging.log(log_level, f"{service_name} settings updated: {settings}")


def log_kalman_init(dt: float, process_noise: float, measurement_noise: float, 
                   reset_threshold: float, velocity_decay: float, position_memory: float, 
                   log_level: int = logging.INFO) -> None:
    """
    Log Kalman filter initialization with standardized format.
    
    Args:
        dt: Time step
        process_noise: Process noise parameter
        measurement_noise: Measurement noise parameter
        reset_threshold: Reset threshold parameter
        velocity_decay: Velocity decay factor
        position_memory: Position memory factor
        log_level: Logging level (default: logging.INFO)
    """
    logging.log(
        log_level, 
        f"Kalman processor initialized with dt={dt}, process_noise={process_noise}, "
        f"measurement_noise={measurement_noise}, reset_threshold={reset_threshold}, "
        f"velocity_decay={velocity_decay}, position_memory={position_memory}"
    )


def log_kalman_update(dt: float, process_noise: float, measurement_noise: float, 
                     reset_threshold: float, velocity_decay: float, position_memory: float, 
                     log_level: int = logging.INFO) -> None:
    """
    Log Kalman filter parameters update with standardized format.
    
    Args:
        dt: Updated time step
        process_noise: Updated process noise parameter
        measurement_noise: Updated measurement noise parameter
        reset_threshold: Updated reset threshold parameter
        velocity_decay: Updated velocity decay factor
        position_memory: Updated position memory factor
        log_level: Logging level (default: logging.INFO)
    """
    logging.log(
        log_level, 
        f"Kalman parameters updated: dt={dt}, process_noise={process_noise}, "
        f"measurement_noise={measurement_noise}, reset_threshold={reset_threshold}, "
        f"velocity_decay={velocity_decay}, position_memory={position_memory}"
    ) 