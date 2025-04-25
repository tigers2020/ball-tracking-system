#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Service Manager module.
This module contains the ServiceManager class which manages service registration and access.
"""

import logging
from typing import Dict, Any, Type, Optional

logger = logging.getLogger(__name__)


class ServiceManager:
    """
    Manager for registering and accessing application services.
    Implements a singleton pattern for global access to services.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ServiceManager, cls).__new__(cls)
            cls._instance._services = {}
            logger.debug("ServiceManager instance created")
        return cls._instance
    
    def register_service(self, service_name: str, service_instance: Any) -> None:
        """
        Register a service instance with a given name.
        
        Args:
            service_name (str): Name to register the service under
            service_instance (Any): The service instance to register
        """
        if service_name in self._services:
            logger.warning(f"Service '{service_name}' already registered, replacing")
        
        self._services[service_name] = service_instance
        logger.debug(f"Service '{service_name}' registered")
    
    def get_service(self, service_name: str) -> Optional[Any]:
        """
        Get a registered service by name.
        
        Args:
            service_name (str): Name of the service to retrieve
            
        Returns:
            Optional[Any]: The service instance, or None if not found
        """
        if service_name not in self._services:
            logger.warning(f"Service '{service_name}' not found")
            return None
        
        return self._services[service_name]
    
    def has_service(self, service_name: str) -> bool:
        """
        Check if a service is registered.
        
        Args:
            service_name (str): Name of the service to check
            
        Returns:
            bool: True if the service is registered
        """
        return service_name in self._services
    
    def remove_service(self, service_name: str) -> bool:
        """
        Remove a registered service.
        
        Args:
            service_name (str): Name of the service to remove
            
        Returns:
            bool: True if the service was removed, False if not found
        """
        if service_name not in self._services:
            logger.warning(f"Cannot remove service '{service_name}', not found")
            return False
        
        del self._services[service_name]
        logger.debug(f"Service '{service_name}' removed")
        return True
    
    def get_all_services(self) -> Dict[str, Any]:
        """
        Get all registered services.
        
        Returns:
            Dict[str, Any]: Dictionary of service names to instances
        """
        return self._services.copy()
    
    def clear_services(self) -> None:
        """
        Clear all registered services.
        """
        self._services.clear()
        logger.debug("All services cleared") 