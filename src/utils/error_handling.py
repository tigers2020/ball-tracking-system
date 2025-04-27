#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Error handling utilities.
This module provides decorators and utilities for consistent error handling
across the application.
"""

import functools
import logging
import enum
import traceback
from typing import Any, Callable, TypeVar, Optional, Union, Dict, ContextManager
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Type variable for function return type
T = TypeVar('T')


class ErrorAction(enum.Enum):
    """Enum defining actions to take when an error occurs."""
    RETURN_DEFAULT = 'return_default'
    RETURN_FALSE = 'return_false'  # Specifically for returning False
    RAISE = 'raise'
    LOG_ONLY = 'log_only'


def handle_errors(action: ErrorAction = ErrorAction.RETURN_DEFAULT,
                  default_return: Any = None,
                  message: str = "An error occurred: {error}",
                  log_level: int = logging.ERROR,
                  log_traceback: bool = False) -> Callable:
    """
    Decorator that provides consistent error handling.
    
    Args:
        action: Action to take when an exception occurs
        default_return: Value to return if action is RETURN_DEFAULT
        message: Message template for the error log (can use {error} placeholder)
        log_level: Logging level to use
        log_traceback: Whether to log the full traceback
        
    Returns:
        Decorated function with error handling
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Union[T, Any]]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Union[T, Any]:
            try:
                return func(*args, **kwargs)
            except Exception as error:
                # Format error message
                error_message = message.format(error=str(error))
                
                # Log the error
                if log_traceback:
                    logger.log(log_level, f"{error_message}\n{traceback.format_exc()}")
                else:
                    logger.log(log_level, error_message)
                
                # Handle error according to action
                if action == ErrorAction.RAISE:
                    raise
                elif action == ErrorAction.RETURN_DEFAULT:
                    return default_return
                elif action == ErrorAction.RETURN_FALSE:
                    return False
                # For LOG_ONLY, continue execution - though this is generally not recommended
                
        return wrapper
    return decorator


def with_logging(func: Optional[Callable] = None,
                 level: int = logging.DEBUG) -> Callable:
    """
    Decorator that logs function entry and exit.
    
    Args:
        func: Function to decorate
        level: Logging level to use
        
    Returns:
        Decorated function with logging
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            func_name = fn.__name__
            logger.log(level, f"Entering {func_name}")
            result = fn(*args, **kwargs)
            logger.log(level, f"Exiting {func_name}")
            return result
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func) 


@contextmanager
def error_handler(
    message: str = "An error occurred: {error}",
    action: ErrorAction = ErrorAction.LOG_ONLY,
    default_return: Any = None,
    log_level: int = logging.ERROR,
    log_traceback: bool = False,
    exception_types: tuple = (Exception,)
) -> ContextManager[None]:
    """
    Context manager for handling errors in a block of code.
    
    This provides a cleaner alternative to repetitive try-except blocks.
    
    Args:
        message: Message template for the error log (can use {error} placeholder)
        action: Action to take when an exception occurs
        default_return: Value to return if action is RETURN_DEFAULT
        log_level: Logging level to use
        log_traceback: Whether to log the full traceback
        exception_types: Tuple of exception types to catch
        
    Yields:
        None: The context manager doesn't provide any value
        
    Example:
        >>> with error_handler("Failed to process data: {error}"):
        ...     result = process_data()
        
        >>> with error_handler(
        ...     message="Failed to save file: {error}",
        ...     action=ErrorAction.RETURN_DEFAULT,
        ...     default_return=False
        ... ) as handled:
        ...     save_file()
        ...     return True
        ... return handled.result  # Returns False if an error occurred
    """
    class ResultContainer:
        def __init__(self):
            self.result = default_return
            self.error_occurred = False
    
    handler = ResultContainer()
    
    try:
        yield handler
    except exception_types as error:
        handler.error_occurred = True
        
        # Format error message
        error_message = message.format(error=str(error))
        
        # Log the error
        if log_traceback:
            logger.log(log_level, f"{error_message}\n{traceback.format_exc()}")
        else:
            logger.log(log_level, error_message)
        
        # Handle error according to action
        if action == ErrorAction.RAISE:
            raise
        elif action == ErrorAction.RETURN_DEFAULT:
            handler.result = default_return
        elif action == ErrorAction.RETURN_FALSE:
            handler.result = False
        # For LOG_ONLY, continue execution 