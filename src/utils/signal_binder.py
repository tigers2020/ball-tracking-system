#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Signal Binder module.
This module provides utilities for managing connections between Qt signals and slots.
"""

import logging
from typing import Dict, Any, Callable, List, Union, Optional, Tuple

logger = logging.getLogger(__name__)


class SignalBinder:
    """
    Utility class for managing signal connections between Qt objects.
    Provides methods to easily connect and disconnect signals between objects.
    """
    
    @staticmethod
    def bind(sender: Any, signal_name: str, receiver: Any, slot_name: str) -> bool:
        """
        Connect a signal from a sender to a slot in a receiver.
        
        Args:
            sender: Object that emits the signal
            signal_name: Name of the signal to connect
            receiver: Object that receives the signal
            slot_name: Name of the slot method to connect
            
        Returns:
            bool: True if connection was successful, False otherwise
        """
        if not sender or not receiver:
            logger.warning(f"Cannot bind signal: sender or receiver is None")
            return False
            
        try:
            # Get the signal by name
            signal = getattr(sender, signal_name, None)
            if not signal:
                logger.warning(f"Signal '{signal_name}' not found in {sender.__class__.__name__}")
                return False
                
            # Get the slot by name
            slot = getattr(receiver, slot_name, None)
            if not slot:
                logger.warning(f"Slot '{slot_name}' not found in {receiver.__class__.__name__}")
                return False
                
            # Connect the signal to the slot
            connected = signal.connect(slot)
            
            if connected:
                logger.debug(f"Connected {sender.__class__.__name__}.{signal_name} to {receiver.__class__.__name__}.{slot_name}")
            else:
                logger.warning(f"Failed to connect {sender.__class__.__name__}.{signal_name} to {receiver.__class__.__name__}.{slot_name}")
                
            return connected
            
        except Exception as e:
            logger.error(f"Error connecting signal {signal_name} to slot {slot_name}: {e}")
            return False
    
    @staticmethod
    def bind_lambda(sender: Any, signal_name: str, lambda_func: Callable) -> bool:
        """
        Connect a signal from a sender to a lambda function.
        
        Args:
            sender: Object that emits the signal
            signal_name: Name of the signal to connect
            lambda_func: Lambda function to connect
            
        Returns:
            bool: True if connection was successful, False otherwise
        """
        if not sender:
            logger.warning(f"Cannot bind signal to lambda: sender is None")
            return False
            
        try:
            # Get the signal by name
            signal = getattr(sender, signal_name, None)
            if not signal:
                logger.warning(f"Signal '{signal_name}' not found in {sender.__class__.__name__}")
                return False
                
            # Connect the signal to the lambda function
            connected = signal.connect(lambda_func)
            
            if connected:
                logger.debug(f"Connected {sender.__class__.__name__}.{signal_name} to lambda function")
            else:
                logger.warning(f"Failed to connect {sender.__class__.__name__}.{signal_name} to lambda function")
                
            return connected
            
        except Exception as e:
            logger.error(f"Error connecting signal {signal_name} to lambda function: {e}")
            return False
    
    @staticmethod
    def unbind(sender: Any, signal_name: str, receiver: Any = None, slot_name: str = None) -> bool:
        """
        Disconnect a signal from a sender to a slot in a receiver.
        If receiver or slot_name is None, disconnects all connections from the signal.
        
        Args:
            sender: Object that emits the signal
            signal_name: Name of the signal to disconnect
            receiver: Object that receives the signal (optional)
            slot_name: Name of the slot method to disconnect (optional)
            
        Returns:
            bool: True if disconnection was successful, False otherwise
        """
        if not sender:
            logger.warning(f"Cannot unbind signal: sender is None")
            return False
            
        try:
            # Get the signal by name
            signal = getattr(sender, signal_name, None)
            if not signal:
                logger.warning(f"Signal '{signal_name}' not found in {sender.__class__.__name__}")
                return False
                
            # Disconnect the signal based on parameters
            if receiver and slot_name:
                # Get the slot by name
                slot = getattr(receiver, slot_name, None)
                if not slot:
                    logger.warning(f"Slot '{slot_name}' not found in {receiver.__class__.__name__}")
                    return False
                    
                # Disconnect the specific connection
                try:
                    signal.disconnect(slot)
                    logger.debug(f"Disconnected {sender.__class__.__name__}.{signal_name} from {receiver.__class__.__name__}.{slot_name}")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to disconnect {sender.__class__.__name__}.{signal_name} from {receiver.__class__.__name__}.{slot_name}: {e}")
                    return False
            elif receiver:
                # Disconnect all connections to the receiver
                try:
                    signal.disconnect(receiver)
                    logger.debug(f"Disconnected all connections from {sender.__class__.__name__}.{signal_name} to {receiver.__class__.__name__}")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to disconnect all connections from {sender.__class__.__name__}.{signal_name} to {receiver.__class__.__name__}: {e}")
                    return False
            else:
                # Disconnect all connections from the signal
                try:
                    signal.disconnect()
                    logger.debug(f"Disconnected all connections from {sender.__class__.__name__}.{signal_name}")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to disconnect all connections from {sender.__class__.__name__}.{signal_name}: {e}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error disconnecting signal {signal_name}: {e}")
            return False
    
    @staticmethod
    def bind_all(sender: Any, receiver: Any, mappings: Dict[str, str]) -> Dict[str, bool]:
        """
        Connect multiple signals from a sender to slots in a receiver using a dictionary mapping.
        
        Args:
            sender: Object that emits the signals
            receiver: Object that receives the signals
            mappings: Dictionary mapping signal names to slot names
            
        Returns:
            Dict[str, bool]: Dictionary with signal names as keys and connection status as values
        """
        results = {}
        for signal_name, slot_name in mappings.items():
            results[signal_name] = SignalBinder.bind(sender, signal_name, receiver, slot_name)
        return results
    
    @staticmethod
    def unbind_all(sender: Any, receiver: Any, signal_names: List[str] = None) -> Dict[str, bool]:
        """
        Disconnect multiple signals from a sender to a receiver.
        
        Args:
            sender: Object that emits the signals
            receiver: Object that receives the signals
            signal_names: List of signal names to disconnect (if None, tries to disconnect all signals)
            
        Returns:
            Dict[str, bool]: Dictionary with signal names as keys and disconnection status as values
        """
        results = {}
        
        # If no signal names provided, try to get all signals from sender
        if signal_names is None:
            try:
                # Get all attributes that might be signals (those that have connect/disconnect methods)
                signal_names = [attr for attr in dir(sender) 
                                if not attr.startswith('_') and 
                                hasattr(getattr(sender, attr), 'connect') and 
                                hasattr(getattr(sender, attr), 'disconnect')]
            except Exception as e:
                logger.error(f"Error getting signal names from {sender.__class__.__name__}: {e}")
                return results
        
        # Disconnect each signal
        for signal_name in signal_names:
            results[signal_name] = SignalBinder.unbind(sender, signal_name, receiver)
            
        return results


def connect_controllers(view, controller_mappings: Dict[str, Dict[str, str]]) -> None:
    """
    Connect a view to multiple controllers using mapping dictionaries.
    
    Args:
        view: View object to connect controllers to
        controller_mappings: Dictionary mapping controller attributes to signal mapping dictionaries
            {
                'controller_attr': {
                    'signal_name': 'slot_name',
                    ...
                },
                ...
            }
    """
    for controller_attr, signal_mappings in controller_mappings.items():
        # Get the controller from the view
        controller = getattr(view, controller_attr, None)
        if not controller:
            logger.warning(f"Controller '{controller_attr}' not found in view")
            continue
            
        # Connect all signals
        results = SignalBinder.bind_all(controller, view, signal_mappings)
        
        # Log results
        success_count = sum(1 for connected in results.values() if connected)
        logger.info(f"Connected {success_count}/{len(results)} signals from {controller_attr} to view") 