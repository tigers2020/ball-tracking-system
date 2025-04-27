#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Triangulation package initialization.

This package provides modules for 3D triangulation from multiple camera views,
including linear and nonlinear triangulation methods.
"""

from src.core.geometry.triangulation.base import AbstractTriangulator
from src.core.geometry.triangulation.linear import LinearTriangulator
from src.core.geometry.triangulation.nonlinear import NonlinearTriangulator
from src.core.geometry.triangulation.factory import TriangulationFactory
import src.core.geometry.triangulation.utils as triangulation_utils

__all__ = [
    'AbstractTriangulator',
    'LinearTriangulator',
    'NonlinearTriangulator',
    'TriangulationFactory',
    'triangulation_utils',
]

# These imports will be added as we implement each file
# from .base import AbstractTriangulator
# from .stereo import StereoTriangulator
# from .multi_view import MultiViewTriangulator 