#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Error handling decorators usage examples.

This module demonstrates how to use the error handling decorators
in different scenarios.
"""

import logging
from typing import Dict, List, Optional, Any

from src.utils.error_handling import (
    handle_errors, log_errors, ignore_errors, retry,
    ErrorAction, ErrorHandlingMixin
)

# Configure logging for examples
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


# Basic error handling examples
@handle_errors()
def divide(a: float, b: float) -> float:
    """
    Divide a by b, re-raising any exceptions after logging them.
    Uses the default error handling with ErrorAction.RERAISE.
    """
    return a / b


@handle_errors(action=ErrorAction.RETURN_DEFAULT, default_return=0)
def safe_divide(a: float, b: float) -> float:
    """
    Safely divide a by b, returning 0 if an error occurs.
    """
    return a / b


@log_errors(message="Failed to process data: {error}")
def process_data(data: List[Dict]) -> List[Any]:
    """
    Process a list of data, logging and re-raising any errors.
    Uses a custom error message format.
    """
    results = []
    for item in data:
        # Process the item (could raise various exceptions)
        result = item['value'] * 2
        results.append(result)
    return results


@ignore_errors(return_value=[])
def parse_data(data_str: str) -> List[Dict]:
    """
    Parse a string into a list of dictionaries.
    Returns an empty list if parsing fails.
    """
    import json
    return json.loads(data_str)


@retry(max_attempts=3, delay=1, backoff_factor=2)
def fetch_data(url: str) -> Dict:
    """
    Fetch data from a URL, retrying up to 3 times with exponential backoff.
    """
    import random
    
    # Simulate a network request that might fail
    if random.random() < 0.7:  # 70% chance of failure for demo purposes
        raise ConnectionError("Network error")
    
    return {"status": "success", "data": "example data"}


# Class with error handling mixin
class DataProcessor(ErrorHandlingMixin):
    """Example class using the ErrorHandlingMixin."""
    
    def __init__(self, data_source: str):
        self.data_source = data_source
    
    @ErrorHandlingMixin.handle_method_errors(
        action=ErrorAction.RETURN_NONE,
        message="Error processing {self.data_source}: {error}"
    )
    def process(self, item_id: int) -> Optional[Dict]:
        """Process an item, returning None if an error occurs."""
        if item_id < 0:
            raise ValueError("Item ID cannot be negative")
        
        # Simulate processing
        return {"id": item_id, "source": self.data_source, "processed": True}


# Demonstration of usage
def demonstrate_error_handling():
    """Demonstrate the usage of error handling decorators."""
    logger.info("Starting error handling demonstration")
    
    # Test basic error handling
    try:
        result = divide(10, 2)
        logger.info(f"10 / 2 = {result}")
        
        result = divide(10, 0)  # This will raise and log ZeroDivisionError
        logger.info("This line won't be reached")
    except ZeroDivisionError:
        logger.info("Caught expected ZeroDivisionError from divide()")
    
    # Test safe error handling with default return
    result = safe_divide(10, 2)
    logger.info(f"safe_divide(10, 2) = {result}")
    
    result = safe_divide(10, 0)  # This will return 0 instead of raising
    logger.info(f"safe_divide(10, 0) = {result}")
    
    # Test with custom error message
    try:
        data = [{"value": 1}, {"value": 2}, {"not_value": 3}]  # Will raise KeyError
        result = process_data(data)
        logger.info(f"process_data result: {result}")
    except KeyError:
        logger.info("Caught expected KeyError from process_data()")
    
    # Test ignore_errors
    invalid_json = "{invalid: json}"
    result = parse_data(invalid_json)  # Will return [] instead of raising
    logger.info(f"parse_data result: {result}")
    
    valid_json = '[{"key": "value"}]'
    result = parse_data(valid_json)
    logger.info(f"parse_data result: {result}")
    
    # Test retry
    try:
        data = fetch_data("https://example.com/api")
        logger.info(f"fetch_data result: {data}")
    except ConnectionError:
        logger.info("fetch_data failed after 3 retries")
    
    # Test class with mixin
    processor = DataProcessor("example_source")
    
    result = processor.process(1)
    logger.info(f"processor.process(1) result: {result}")
    
    result = processor.process(-1)  # Will return None due to ValueError
    logger.info(f"processor.process(-1) result: {result}")
    
    logger.info("Error handling demonstration completed")


if __name__ == "__main__":
    demonstrate_error_handling() 