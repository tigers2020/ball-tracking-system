#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Format utilities module.
Provides utility functions for string formatting.
"""

from datetime import timedelta


def format_time_delta(delta):
    """
    Format a timedelta object into a readable string (HH:MM:SS).
    
    Args:
        delta (datetime.timedelta): The time delta to format
        
    Returns:
        str: Formatted time string in the format HH:MM:SS
    """
    if not isinstance(delta, timedelta):
        return "00:00:00"
    
    # Calculate total seconds
    total_seconds = int(delta.total_seconds())
    
    # Extract hours, minutes, seconds
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Format as HH:MM:SS
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_float(value, precision=2):
    """
    Format a float value with specified precision.
    
    Args:
        value (float): The float value to format
        precision (int, optional): Number of decimal places. Defaults to 2.
        
    Returns:
        str: Formatted float string
    """
    if not isinstance(value, (int, float)):
        return "0.00"
    
    format_str = f"{{:.{precision}f}}"
    return format_str.format(value)


def format_coordinates(x, y, z=None, precision=2):
    """
    Format coordinates as a string.
    
    Args:
        x (float): X coordinate
        y (float): Y coordinate
        z (float, optional): Z coordinate if 3D
        precision (int, optional): Number of decimal places. Defaults to 2.
        
    Returns:
        str: Formatted coordinates string
    """
    if z is None:
        return f"({format_float(x, precision)}, {format_float(y, precision)})"
    else:
        return f"({format_float(x, precision)}, {format_float(y, precision)}, {format_float(z, precision)})" 