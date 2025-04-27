#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Parameter Handler module.
This module contains abstract and concrete classes for handling different parameter groups.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Set
from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)


class ParameterHandler(ABC):
    """
    Abstract base class for parameter handlers.
    Provides a common interface for updating and validating parameters.
    """
    
    def __init__(self, initial_params: Dict[str, Any] = None):
        """
        Initialize the parameter handler.
        
        Args:
            initial_params: Initial parameter values
        """
        self.params = {}
        if initial_params:
            self.update_params(initial_params)
    
    @abstractmethod
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parameters before updating.
        
        Args:
            params: Parameters to validate
            
        Returns:
            Validated parameters (possibly modified)
        """
        pass
    
    def update_params(self, params: Dict[str, Any]) -> None:
        """
        Update parameters after validation.
        
        Args:
            params: Parameters to update
        """
        validated_params = self.validate_params(params)
        self.params.update(validated_params)
        self.log_params_update(validated_params)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get current parameters.
        
        Returns:
            Current parameters
        """
        return self.params.copy()
    
    def log_params_update(self, updated_params: Dict[str, Any]) -> None:
        """
        Log parameters update.
        
        Args:
            updated_params: Updated parameters
        """
        handler_name = self.__class__.__name__
        param_str = ", ".join([f"{k}={v}" for k, v in updated_params.items()])
        logger.info(f"{handler_name} parameters updated: {param_str}")


# 메타클래스 충돌 해결을 위한 중간 클래스
class _ParameterHandlerMixin:
    """
    Mixin class for parameter handlers.
    This class exists to resolve metaclass conflicts between QObject and ABC.
    """
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Placeholder implementation to be overridden by subclasses.
        
        Args:
            params: Parameters to validate
            
        Returns:
            Validated parameters (possibly modified)
        """
        return params


class QParameterHandler(QObject, _ParameterHandlerMixin):
    """
    Qt-enabled parameter handler with signals for parameter changes.
    """
    
    params_updated = Signal(dict)
    
    def __init__(self, initial_params: Dict[str, Any] = None):
        """
        Initialize the Qt parameter handler.
        
        Args:
            initial_params: Initial parameter values
        """
        QObject.__init__(self)
        self.params = {}
        if initial_params:
            self.update_params(initial_params)
    
    def update_params(self, params: Dict[str, Any]) -> None:
        """
        Update parameters and emit signal.
        
        Args:
            params: Parameters to update
        """
        validated_params = self.validate_params(params)
        self.params.update(validated_params)
        self.log_params_update(validated_params)
        
        # Emit signal with updated params
        self.params_updated.emit(self.get_params())
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get current parameters.
        
        Returns:
            Current parameters
        """
        return self.params.copy()
    
    def log_params_update(self, updated_params: Dict[str, Any]) -> None:
        """
        Log parameters update.
        
        Args:
            updated_params: Updated parameters
        """
        handler_name = self.__class__.__name__
        param_str = ", ".join([f"{k}={v}" for k, v in updated_params.items()])
        logger.info(f"{handler_name} parameters updated: {param_str}")


class ROIParameterHandler(QParameterHandler):
    """
    Handler for ROI (Region of Interest) parameters.
    """
    
    DEFAULT_PARAMS = {
        "width": 100,
        "height": 100,
        "auto_center": True,
        "enabled": True,
    }
    
    REQUIRED_KEYS = {"width", "height"}
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate ROI parameters.
        
        Args:
            params: ROI parameters to validate
            
        Returns:
            Validated ROI parameters
        """
        validated = {}
        
        # Handle width parameter
        if "width" in params:
            width = params["width"]
            if not isinstance(width, (int, float)) or width <= 0:
                logger.warning(f"Invalid ROI width: {width}, using default: {self.DEFAULT_PARAMS['width']}")
                validated["width"] = self.DEFAULT_PARAMS["width"]
            else:
                validated["width"] = int(width)
        
        # Handle height parameter
        if "height" in params:
            height = params["height"]
            if not isinstance(height, (int, float)) or height <= 0:
                logger.warning(f"Invalid ROI height: {height}, using default: {self.DEFAULT_PARAMS['height']}")
                validated["height"] = self.DEFAULT_PARAMS["height"]
            else:
                validated["height"] = int(height)
        
        # Handle auto_center parameter
        if "auto_center" in params:
            validated["auto_center"] = bool(params["auto_center"])
        
        # Handle enabled parameter
        if "enabled" in params:
            validated["enabled"] = bool(params["enabled"])
        
        # Handle center coordinates if manual centering is used
        if "center_x" in params or "center_y" in params:
            if "center_x" in params:
                validated["center_x"] = int(params["center_x"])
            if "center_y" in params:
                validated["center_y"] = int(params["center_y"])
        
        return validated


class HoughParameterHandler(QParameterHandler):
    """
    Handler for Hough Circle parameters.
    """
    
    DEFAULT_PARAMS = {
        "dp": 1.5,
        "min_dist": 20,
        "param1": 100,
        "param2": 30,
        "min_radius": 10,
        "max_radius": 100,
        "adaptive": True
    }
    
    REQUIRED_KEYS = {"dp", "min_dist", "param1", "param2", "min_radius", "max_radius"}
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate Hough Circle parameters.
        
        Args:
            params: Hough parameters to validate
            
        Returns:
            Validated Hough parameters
        """
        validated = {}
        
        # Validate dp (resolution ratio)
        if "dp" in params:
            dp = params["dp"]
            if not isinstance(dp, (int, float)) or dp <= 0:
                logger.warning(f"Invalid dp: {dp}, using default: {self.DEFAULT_PARAMS['dp']}")
                validated["dp"] = self.DEFAULT_PARAMS["dp"]
            else:
                validated["dp"] = float(dp)
        
        # Validate min_dist (minimum distance between circles)
        if "min_dist" in params:
            min_dist = params["min_dist"]
            if not isinstance(min_dist, (int, float)) or min_dist < 1:
                logger.warning(f"Invalid min_dist: {min_dist}, using default: {self.DEFAULT_PARAMS['min_dist']}")
                validated["min_dist"] = self.DEFAULT_PARAMS["min_dist"]
            else:
                validated["min_dist"] = int(min_dist)
        
        # Validate param1 (higher threshold for edge detection)
        if "param1" in params:
            param1 = params["param1"]
            if not isinstance(param1, (int, float)) or param1 < 1:
                logger.warning(f"Invalid param1: {param1}, using default: {self.DEFAULT_PARAMS['param1']}")
                validated["param1"] = self.DEFAULT_PARAMS["param1"]
            else:
                validated["param1"] = int(param1)
        
        # Validate param2 (threshold for center detection)
        if "param2" in params:
            param2 = params["param2"]
            if not isinstance(param2, (int, float)) or param2 < 1:
                logger.warning(f"Invalid param2: {param2}, using default: {self.DEFAULT_PARAMS['param2']}")
                validated["param2"] = self.DEFAULT_PARAMS["param2"]
            else:
                validated["param2"] = int(param2)
        
        # Validate min_radius
        if "min_radius" in params:
            min_radius = params["min_radius"]
            if not isinstance(min_radius, (int, float)) or min_radius < 0:
                logger.warning(f"Invalid min_radius: {min_radius}, using default: {self.DEFAULT_PARAMS['min_radius']}")
                validated["min_radius"] = self.DEFAULT_PARAMS["min_radius"]
            else:
                validated["min_radius"] = int(min_radius)
        
        # Validate max_radius
        if "max_radius" in params:
            max_radius = params["max_radius"]
            if not isinstance(max_radius, (int, float)) or max_radius < 1:
                logger.warning(f"Invalid max_radius: {max_radius}, using default: {self.DEFAULT_PARAMS['max_radius']}")
                validated["max_radius"] = self.DEFAULT_PARAMS["max_radius"]
            else:
                validated["max_radius"] = int(max_radius)
        
        # Validate adaptive
        if "adaptive" in params:
            validated["adaptive"] = bool(params["adaptive"])
        
        # Ensure min_radius < max_radius
        if "min_radius" in validated and "max_radius" in validated:
            if validated["min_radius"] >= validated["max_radius"]:
                logger.warning(f"min_radius ({validated['min_radius']}) >= max_radius ({validated['max_radius']}), adjusting")
                validated["min_radius"] = min(validated["min_radius"], validated["max_radius"] - 1)
        
        return validated


class MorphologyParameterHandler(QParameterHandler):
    """
    Handler for morphological operation parameters.
    """
    
    DEFAULT_PARAMS = {
        "kernel_size": (5, 5),
        "iterations": 2,
        "operation": "dilate"  # 'erode', 'dilate', 'open', 'close'
    }
    
    VALID_OPERATIONS = {"erode", "dilate", "open", "close"}
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate morphological operation parameters.
        
        Args:
            params: Morphology parameters to validate
            
        Returns:
            Validated morphology parameters
        """
        validated = {}
        
        # Validate kernel_size
        if "kernel_size" in params:
            kernel_size = params["kernel_size"]
            if isinstance(kernel_size, (list, tuple)) and len(kernel_size) == 2:
                try:
                    ks_x, ks_y = int(kernel_size[0]), int(kernel_size[1])
                    if ks_x > 0 and ks_y > 0:
                        # Ensure kernel size is odd (for symmetry)
                        if ks_x % 2 == 0:
                            ks_x += 1
                        if ks_y % 2 == 0:
                            ks_y += 1
                        validated["kernel_size"] = (ks_x, ks_y)
                    else:
                        logger.warning(f"Invalid kernel_size: {kernel_size}, using default: {self.DEFAULT_PARAMS['kernel_size']}")
                        validated["kernel_size"] = self.DEFAULT_PARAMS["kernel_size"]
                except (ValueError, TypeError):
                    logger.warning(f"Invalid kernel_size: {kernel_size}, using default: {self.DEFAULT_PARAMS['kernel_size']}")
                    validated["kernel_size"] = self.DEFAULT_PARAMS["kernel_size"]
            else:
                logger.warning(f"Invalid kernel_size: {kernel_size}, using default: {self.DEFAULT_PARAMS['kernel_size']}")
                validated["kernel_size"] = self.DEFAULT_PARAMS["kernel_size"]
        
        # Validate iterations
        if "iterations" in params:
            iterations = params["iterations"]
            if not isinstance(iterations, int) or iterations < 1:
                logger.warning(f"Invalid iterations: {iterations}, using default: {self.DEFAULT_PARAMS['iterations']}")
                validated["iterations"] = self.DEFAULT_PARAMS["iterations"]
            else:
                validated["iterations"] = iterations
        
        # Validate operation
        if "operation" in params:
            operation = params["operation"].lower() if isinstance(params["operation"], str) else ""
            if operation not in self.VALID_OPERATIONS:
                logger.warning(f"Invalid operation: {operation}, using default: {self.DEFAULT_PARAMS['operation']}")
                validated["operation"] = self.DEFAULT_PARAMS["operation"]
            else:
                validated["operation"] = operation
        
        return validated 