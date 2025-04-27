#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Net Zone module.
This module contains the NetZone enum for tennis court side identification.
"""

from enum import Enum


class NetZone(Enum):
    """
    Enum representing the sides of the tennis court relative to the net.
    Used for determining which side of the court the ball is on.
    """
    LEFT = 0
    RIGHT = 1 