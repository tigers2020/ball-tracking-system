#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Triangulation factory module.

This module provides a factory class for creating triangulator instances
based on specified methods and parameters.
"""

import logging
from typing import Dict, Any, Optional, Union

from src.core.geometry.triangulation.base import AbstractTriangulator
from src.core.geometry.triangulation.linear import LinearTriangulator
from src.core.geometry.triangulation.nonlinear import NonlinearTriangulator

logger = logging.getLogger(__name__)

class TriangulationFactory:
    """
    Factory class for creating triangulator instances.
    
    This class provides static methods to create different types of triangulators
    with appropriate parameters.
    """
    
    @staticmethod
    def create_triangulator(method: str = 'linear', 
                           sub_method: str = 'dlt',
                           camera_params: Optional[Dict[str, Any]] = None,
                           **kwargs) -> AbstractTriangulator:
        """
        Create a triangulator instance based on the specified method.
        
        Args:
            method: Triangulation method ('linear', 'nonlinear')
            sub_method: Specific algorithm implementation
                       - For linear: 'dlt', 'midpoint', 'eigen'
                       - For nonlinear: 'lm', 'trf', 'dogbox'
            camera_params: Camera parameters to initialize the triangulator
            **kwargs: Additional parameters for specific triangulator types
            
        Returns:
            AbstractTriangulator instance
        """
        if method == 'linear':
            return TriangulationFactory.create_linear_triangulator(sub_method, camera_params, **kwargs)
        elif method == 'nonlinear':
            return TriangulationFactory.create_nonlinear_triangulator(sub_method, camera_params, **kwargs)
        else:
            logger.error(f"Unknown triangulation method: {method}")
            # Default to DLT triangulation
            return LinearTriangulator(camera_params)
    
    @staticmethod
    def create_linear_triangulator(method: str = 'dlt',
                                  camera_params: Optional[Dict[str, Any]] = None,
                                  **kwargs) -> LinearTriangulator:
        """
        Create a linear triangulator with the specified method.
        
        Args:
            method: Linear triangulation method ('dlt', 'midpoint', 'eigen')
            camera_params: Camera parameters to initialize the triangulator
            **kwargs: Additional parameters
            
        Returns:
            LinearTriangulator instance
        """
        valid_methods = ['dlt', 'midpoint', 'eigen']
        if method not in valid_methods:
            logger.warning(f"Unknown linear method: {method}. Using 'dlt' instead.")
            method = 'dlt'
            
        return LinearTriangulator(camera_params, method=method)
    
    @staticmethod
    def create_nonlinear_triangulator(method: str = 'lm',
                                     camera_params: Optional[Dict[str, Any]] = None,
                                     linear_method: str = 'dlt',
                                     **kwargs) -> NonlinearTriangulator:
        """
        Create a nonlinear triangulator with the specified method.
        
        Args:
            method: Nonlinear optimization method ('lm', 'trf', 'dogbox')
            camera_params: Camera parameters to initialize the triangulator
            linear_method: Linear method for initial estimation
            **kwargs: Additional parameters
            
        Returns:
            NonlinearTriangulator instance
        """
        valid_methods = ['lm', 'trf', 'dogbox']
        if method not in valid_methods:
            logger.warning(f"Unknown nonlinear method: {method}. Using 'lm' instead.")
            method = 'lm'
            
        return NonlinearTriangulator(camera_params, 
                                    linear_method=linear_method, 
                                    optimization_method=method)
                                    
    @staticmethod
    def create_triangulator_from_config(config: Dict[str, Any], 
                                       camera_params: Optional[Dict[str, Any]] = None) -> AbstractTriangulator:
        """
        Create a triangulator from a configuration dictionary.
        
        Args:
            config: Configuration dictionary with triangulation parameters
            camera_params: Camera parameters to initialize the triangulator
            
        Returns:
            AbstractTriangulator instance
        """
        method = config.get('method', 'linear')
        sub_method = config.get('sub_method', 'dlt')
        
        # Additional parameters for specific triangulator types
        kwargs = {}
        
        if method == 'nonlinear':
            kwargs['linear_method'] = config.get('linear_method', 'dlt')
            
        # Create triangulator
        triangulator = TriangulationFactory.create_triangulator(
            method=method,
            sub_method=sub_method,
            camera_params=camera_params,
            **kwargs
        )
        
        return triangulator 