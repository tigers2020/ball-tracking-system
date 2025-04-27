#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Parameter Manager module.
This module contains the ParameterManager class which centralizes parameter management
across the application. It serves as a bridge between the ConfigManager and various
parameter handlers for specific parameter groups.
"""

import logging
from typing import Dict, Any, Optional, Type
from PySide6.QtCore import QObject, Signal

from src.utils.parameter_handler import (
    ParameterHandler, 
    QParameterHandler,
    ROIParameterHandler, 
    HoughParameterHandler,
    MorphologyParameterHandler
)

logger = logging.getLogger(__name__)

class ParameterManager(QObject):
    """
    Centralized parameter manager that handles different parameter groups
    and provides validation through specialized parameter handlers.
    
    This class acts as a registry for parameter handlers and provides
    access to them through a unified interface. It also manages the 
    synchronization of parameter changes between the handlers and the
    configuration manager.
    """
    
    # Signals for parameter updates that can be connected to by controllers
    hsv_parameters_updated = Signal(dict)
    roi_parameters_updated = Signal(dict)
    hough_parameters_updated = Signal(dict)
    kalman_parameters_updated = Signal(dict)
    camera_parameters_updated = Signal(dict)
    
    def __init__(self, config_manager):
        """
        Initialize the parameter manager.
        
        Args:
            config_manager: The configuration manager instance
        """
        super().__init__()
        self.config_manager = config_manager
        
        # Initialize parameter handlers dictionary
        self._handlers = {}
        
        # Initialize parameter groups
        self._initialize_parameter_handlers()
        
        # Load parameters from configuration
        self._load_parameters_from_config()
    
    def _initialize_parameter_handlers(self):
        """
        Initialize the parameter handlers for different parameter groups.
        """
        # Initialize ROI parameter handler
        self._handlers['roi'] = ROIParameterHandler()
        
        # Initialize Hough Circle parameter handler
        self._handlers['hough'] = HoughParameterHandler()
        
        # Initialize Morphology parameter handler for HSV
        self._handlers['morphology'] = MorphologyParameterHandler()
        
        # Connect parameter handlers to their respective signal handlers
        if isinstance(self._handlers['roi'], QParameterHandler):
            self._handlers['roi'].params_updated.connect(
                lambda params: self.roi_parameters_updated.emit(params))
            
        if isinstance(self._handlers['hough'], QParameterHandler):
            self._handlers['hough'].params_updated.connect(
                lambda params: self.hough_parameters_updated.emit(params))
    
    def _load_parameters_from_config(self):
        """
        Load all parameters from configuration into the respective handlers.
        """
        # Load ROI settings
        roi_settings = self.config_manager.get_roi_settings()
        if roi_settings and 'roi' in self._handlers:
            self._handlers['roi'].update_params(roi_settings)
            logger.info("ROI settings loaded from configuration")
        
        # Load Hough Circle settings
        hough_settings = self.config_manager.get_hough_circle_settings()
        if hough_settings and 'hough' in self._handlers:
            self._handlers['hough'].update_params(hough_settings)
            logger.info("Hough Circle settings loaded from configuration")
        
        # Load HSV settings
        hsv_settings = self.config_manager.get_hsv_settings()
        if hsv_settings:
            # HSV settings don't have a dedicated handler yet, but we emit the signal
            self.hsv_parameters_updated.emit(hsv_settings)
            logger.info("HSV settings loaded from configuration")
        
        # Load Kalman settings
        kalman_settings = self.config_manager.get_kalman_settings()
        if kalman_settings:
            # Kalman settings don't have a dedicated handler yet, but we emit the signal
            self.kalman_parameters_updated.emit(kalman_settings)
            logger.info("Kalman settings loaded from configuration")
        
        # Load Camera settings
        camera_settings = self.config_manager.get_camera_settings()
        if camera_settings:
            # Camera settings don't have a dedicated handler yet, but we emit the signal
            self.camera_parameters_updated.emit(camera_settings)
            logger.info("Camera settings loaded from configuration")
    
    def get_handler(self, parameter_group: str) -> Optional[ParameterHandler]:
        """
        Get the parameter handler for a specific parameter group.
        
        Args:
            parameter_group: Name of the parameter group ('roi', 'hough', etc.)
            
        Returns:
            The parameter handler for the specified group or None if not found
        """
        return self._handlers.get(parameter_group)
    
    def update_roi_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update ROI parameters.
        
        Args:
            params: ROI parameters to update
            
        Returns:
            The updated and validated parameters
        """
        if 'roi' not in self._handlers:
            logger.warning("ROI parameter handler not found")
            return {}
            
        # Update parameters in the handler
        self._handlers['roi'].update_params(params)
        
        # Get validated parameters
        validated_params = self._handlers['roi'].get_params()
        
        # Update configuration
        self.config_manager.set_roi_settings(validated_params)
        
        return validated_params
    
    def update_hough_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update Hough Circle parameters.
        
        Args:
            params: Hough Circle parameters to update
            
        Returns:
            The updated and validated parameters
        """
        if 'hough' not in self._handlers:
            logger.warning("Hough Circle parameter handler not found")
            return {}
            
        # Update parameters in the handler
        self._handlers['hough'].update_params(params)
        
        # Get validated parameters
        validated_params = self._handlers['hough'].get_params()
        
        # Update configuration
        self.config_manager.set_hough_circle_settings(validated_params)
        
        return validated_params
    
    def update_hsv_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update HSV parameters.
        
        Args:
            params: HSV parameters to update
            
        Returns:
            The updated parameters
        """
        # We don't have a dedicated handler for HSV yet, so we directly update the config
        self.config_manager.set_hsv_settings(params)
        
        # Emit signal
        self.hsv_parameters_updated.emit(params)
        
        return params
    
    def update_kalman_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update Kalman filter parameters.
        
        Args:
            params: Kalman filter parameters to update
            
        Returns:
            The updated parameters
        """
        # We don't have a dedicated handler for Kalman yet, so we directly update the config
        self.config_manager.set_kalman_settings(params)
        
        # Emit signal
        self.kalman_parameters_updated.emit(params)
        
        return params
    
    def update_camera_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update camera parameters.
        
        Args:
            params: Camera parameters to update
            
        Returns:
            The updated parameters
        """
        # We don't have a dedicated handler for camera settings yet, so we directly update the config
        self.config_manager.set_camera_settings(params)
        
        # Emit signal
        self.camera_parameters_updated.emit(params)
        
        return params
    
    def get_roi_parameters(self) -> Dict[str, Any]:
        """
        Get ROI parameters.
        
        Returns:
            Current ROI parameters
        """
        if 'roi' in self._handlers:
            return self._handlers['roi'].get_params()
        return self.config_manager.get_roi_settings()
    
    def get_hough_parameters(self) -> Dict[str, Any]:
        """
        Get Hough Circle parameters.
        
        Returns:
            Current Hough Circle parameters
        """
        if 'hough' in self._handlers:
            return self._handlers['hough'].get_params()
        return self.config_manager.get_hough_circle_settings()
    
    def get_hsv_parameters(self) -> Dict[str, Any]:
        """
        Get HSV parameters.
        
        Returns:
            Current HSV parameters
        """
        return self.config_manager.get_hsv_settings()
    
    def get_kalman_parameters(self) -> Dict[str, Any]:
        """
        Get Kalman filter parameters.
        
        Returns:
            Current Kalman filter parameters
        """
        return self.config_manager.get_kalman_settings()
    
    def get_camera_parameters(self) -> Dict[str, Any]:
        """
        Get camera parameters.
        
        Returns:
            Current camera parameters
        """
        return self.config_manager.get_camera_settings() 