#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Examples demonstrating the use of error handling utilities.
"""

import logging
import os
from src.utils.error_handling import (
    handle_errors,
    error_handler,
    ErrorAction,
    with_logging
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_error_handler_context_manager():
    """Demonstrate the use of error_handler context manager."""
    logger.info("Example 1: Basic error handling with context manager")
    
    # Simple usage - just log the error
    with error_handler("Error occurred while dividing: {error}"):
        result = 10 / 0  # This will raise ZeroDivisionError
        logger.info(f"This won't be executed due to the error above")
    
    logger.info("Execution continues after the error")
    
    # Using with return value
    logger.info("\nExample 2: Using error_handler with default return value")
    
    def divide(a, b):
        with error_handler(
            message="Error dividing {a} by {b}: {error}".format(a=a, b=b, error="{error}"),
            action=ErrorAction.RETURN_DEFAULT,
            default_return=0
        ) as handler:
            result = a / b
            return result
        return handler.result
    
    result = divide(10, 2)
    logger.info(f"Result of 10/2: {result}")
    
    result = divide(10, 0)
    logger.info(f"Result of 10/0: {result} (used default value)")
    
    # Using to check for errors
    logger.info("\nExample 3: Checking if an error occurred")
    
    def save_file(content, path):
        with error_handler(
            message="Failed to save file at {}: {error}".format(path, error="{error}"),
            action=ErrorAction.LOG_ONLY
        ) as handler:
            with open(path, 'w') as f:
                f.write(content)
            logger.info(f"Successfully saved file to {path}")
            return True
        
        if handler.error_occurred:
            logger.warning("Handling the file save error")
            return False
        return True
    
    # Try to save to an invalid path
    success = save_file("test content", "/invalid/path/file.txt")
    logger.info(f"Save operation success: {success}")
    
    # Try to save to a valid path
    temp_path = "temp_example.txt"
    success = save_file("test content", temp_path)
    logger.info(f"Save operation success: {success}")
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    # Multiple exception types
    logger.info("\nExample 4: Handling specific exception types")
    
    def process_data(data):
        with error_handler(
            message="Error processing data: {error}",
            exception_types=(ValueError, TypeError, KeyError)
        ):
            # This might raise various errors
            if isinstance(data, dict):
                return data["key"]
            elif isinstance(data, str):
                return int(data)
            else:
                raise TypeError("Unsupported data type")
    
    process_data({})  # KeyError
    process_data("not_a_number")  # ValueError
    process_data(None)  # TypeError
    
    try:
        with error_handler(
            message="This will only catch ValueError: {error}",
            exception_types=(ValueError,)
        ):
            raise RuntimeError("This error type won't be caught")
    except RuntimeError:
        logger.info("Caught RuntimeError outside the context manager")


if __name__ == "__main__":
    example_error_handler_context_manager() 