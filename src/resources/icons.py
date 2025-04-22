#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Icons resource module.
This module provides access to icon resources for the application.
"""

import os
import logging
from pathlib import Path

from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtCore import QBuffer, QByteArray, QIODevice

# Icon data as base64 strings
ICON_DATA = {
    "play": """
    iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAA7AAAAOwBeShxvQAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAHvSURBVFiF7ZY/aBNhGIef907atEbS9sBEQTo00MG/iwiiCEIHXQRd3BxcRRBEcLC7oLiJk7iJooubIA5dVHBw0FVBiohDS0X8Q6xJ89793jokhoa0uXxfu/jMe7w/eLjv4/vg00pK8j+j/A7IGNrXHGZiwDjqo9VHLVB+5fszwqgWTtJ+M4g2tK85tOslfH2H9/xbfbR6WDqnDWpDuw5ae8jMIvF9zLUXzEX3EEnmAF9czp1I6D+0UkH8BPkyyMDmOLg9SdQP6I3iFKE3AqDVReJBqI0SkJpFDNt+FBBoT15JnNjAiXKJXJf7YEsdJGzDvxIAOIdBK0iuEQAgg0Wk5CJmFpWeBzPSLwjLKKt3vMbsSWcAcWPkyw848SDUorvXH1A5f4Tw4hH8Fze3D+AZFCtrBJDkxM6gHvIlSJxE3B0+HQog7yCl2u6wvvIzKGdvA4g7TPTL59Hv10GqPQHUxAUKN2dwXt7vDdCvpO7rkTubzRVXssMVwDl+Fn/hOdHdC6g/9f4A6sRFwkvHMLMrgAJu3MkVl3Ojl4BtaTq+QvHTBul2nL0bQCuijx8QDG0F5/RlnFM3RgrQrNSdLDbjB0BuSdyY/O3ZkeYBm7F8MZsrrpgBLjM9M7yh1Q6A3MrM+PXMuI8Z9VFLkkj9BCryP3YR3mzcAAAAAElFTkSuQmCC
    """,
    "pause": """
    iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAA7AAAAOwBeShxvQAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAADWSURBVFiF7ZdNCsIwEIW/iT/HcOEJPJvgUkGvoKB7L+INXIh4g4JeQMSNZyo0TdKUSTsB6YOBkvfymJeZQC5lGIZ3EbkBGFMFVmLmZVk+iMgO+ORKIAHsAYQQKCLmqgCmB+YsZVRY8jWNGc/LVeuBdcRcBwAPQFVV4Jx7AQhgkysBVXUAVFU3WRNI+oLFbM7Eee7X9mXL6+E48eYE2g7ktNZEFqcFKIqCTxu5XRDCGHMFLsBSVZ3tgLX2mRKgV4B6+HHdAcQbUg+t3dB9+vt/Qk6gsANfJiiS+qDnUyUAAAAASUVORK5CYII=
    """,
    "stop": """
    iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAA7AAAAOwBeShxvQAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAADPSURBVFiF7ZdLCsIwFEXT0CrOXJRrceZQXIJD9+DCLejQgbtwJw4EEYuS5JmkJYV64JHS3HffI58mkUgkEr+P9p1QVdWeXU1TVxcM4Lqu670FLgPQS2cJ2LZdAVgDKAC8bdveAPRZyVVVlQDIr7yfrutaAaAHwPu+nwMAVlUlY+MIAJimeUwpoJ9QW8Dlj4iHEMjewEfgWIGHgE8Cp4BLAQYB1wIvBMYQuBHoEzAV8EXgLhAiECQQYxeM9p+w67pGtP1TGDJ6PwnEY/S/QskE3p/JVHSObdQlAAAAAElFTkSuQmCC
    """,
    "next": """
    iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAA7AAAAOwBeShxvQAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAGqSURBVFiF7Za9ThtBFIW/u8YiQUokJEjUkTsU0aWgQhS8A8/A61BSJCnS0KdL6qQhhQ2SJSSBAC5wumvvzCL/JJYXZjFo90hXu3vvnfOdOzMrNgIkTQMP8iXQSo8B9oDuMJO8Yf0K/AbWbW9KWgA+AfettY8kfUXyIR3jQGcYAElx+4p55wAmkiKgcV4Qtp+nCyPcaGVZliXA0hVDLNVFGAEYx3nS/wxA0jhQXASgG7C9GtOO+ZeBGKn1DmZr3p+Lg2e3F54BQc+BpFvAF2A2PdoHnkn6cQTBzIak2cTXI9uvLrR3jVXwoO5+SQ+Br8BsvdB2r9FofAYWJW3FKnhg+/WFBRKAnVQFtfhLFTSbzQWgBeTAN0kz9vEJ2EsP1+JXgIa0/SFdnwLN0+JXgEbj11TVn7a7/VsBvVMAtvsVbwKPG4Cfx+IpfizvBMA4FW6m+I9hIk8GSPFHw8QdA5gYJt7PAAnkKXCnDvAfAI7i1Wr+yKfyArGWj4CdqvpXfTruvk5lK4yHUTNV0PcG45t/U9IysHuJENuSPgJv4/M+uI4DxVX15Ar0F0eBRNVMsw8cAAAAAElFTkSuQmCC
    """,
    "prev": """
    iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAA7AAAAOwBeShxvQAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAHISURBVFiF7ZY9bhNBFMd/M14HIQpDgUODkKyIgoKecAJzBE7CEbgAVRokCuiQuEFoKBCJFAllpIQoJst63zs0SGvLduKZtRMa/qXRjt/M/N+bj51ZUVWMMQvAZ2AdaOauCMwL/yZwbIz5AHwVkZ3sHVUtEpGfwFmFnACHInKVe6o+l1NVVJXZxtbxQpCzKoUYATBGWlWhOgDdA67KBDQkIMSUlNYL9xagXgm8j+RCjk9uPQqgaWZ0/lhEdnMxn1X1CfBJRC5Ffby+6+vDm6j9J4De9R9w7zq1aecxOqnYHATwJZWe6wn4niCHubHJQwCwkAUeU+6GhPc+Vgaw5DUxvIY48QbvfawMACZ/+VKa2RJbqBN6FUDW91+WJ9YiBVgdFtAohYAYYwQqA5gGVeEKICyZInl9MiugCgSKfFAGVYEtgtO0xVkBGkMVeJ3oP0iJb04KCIeUZ4l5FoNNqaAKqopzDoD7iUXrwFPgdQHgG7CfufcMH9Y8iEnZu0+Ap4UCst3tYSp5H3iTivkKvAPO89jWWluKawAfJ5RXqGqnUMCYCh4B+8aYNRE5AHrF5CJyC7wAukXnACKyC3SKzgs9A/4CezWZ3QLngD8DSuRfwEg2s/ByLoEAAAAASUVORK5CYII=
    """,
    "open": """
    iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAA7AAAAOwBeShxvQAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAJlSURBVFiF7ZdNiE5RGMd/533vO2Y+3hnkY0yNkGSBJptZWGAhC1lYUFZKFrJQs7CZJJuyQcxOslGzQ0mzkY2PbLCRjwVmPsYM73vvcY7FfcddvO/1vs2C/9S955zn+f/OPc9zn3OViBDEVlR1GbAOWAYsAeYC04EpQA9wF7gBXAFOicj3XqA1gC2r6irgGLCg4LwXgN0icrYQgKrOAY4A6wtOfCLlwGERGckFoKoG2APs7zWhmX0EtgFt/9VZRYYz+xrYVWDyAaBDRPqstUOquixJkjuNQptgk36gvd/s8cxasw3YBMwAdgJNwFlV7TDG9KTr+rkqPEtm3wC2ichgPefM4sB6EVkLtIrIZmCtqrb7c7F4QFWbgL3AVhF5G4H8WxFpFRHr3YJJOahqC3AcWCUi1wugHfP7ik4BuBnYUTA9fwYmPPsLEdmeJMnnIEBXV9dwfX39JqALmBsAEOCOqs4SkTG/T5J08cRMwOvXwM4kSd75ZTDZ4ORwOBFLg0BnLpcySwLuB7KllYvAJ+Bj3sSqOu3PPz3q6uouwClVAFwF1olIpqOJlRnVCiiH3CDQ5Jw7opRqA/aJyKEA4DXgkNZ6P9AJ1OU5V1pB04EnwJw8sBlzG2MeW2vb/bKNxNIkSbLvQXTuR6HVWnu/VqvF/i5sFJF7uQCVUi3APWpAG2POWGvrgLR0eI+I1ODzEdgjIqfTNB3yCWSMaVbKNQGDSrl+pYb6RQbfAY2+VL9ZaxuBaUClt/clSXLPL9WBaQAewzXgkdYMa+1+gIoIqGoj8BaYHQH/GfjVKzIC3AD+ACP1a6sJIm/nAAAAAElFTkSuQmCC
    """,
    "save": """
    iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAA7AAAAOwBeShxvQAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAG/SURBVFiF7ZcxSxxBFMd/b3eFWIRUUYv7BNfZWVhcpBDLfIV0KZNvsE3KbRPwvkC6FBYpUp1VildYrIX3pxARQYznwZvNW9jb3Z0RQfzDMDO8+c//N+/N7KyEELiP08xKZrYKvAZeAiVgLj5eAxrgHPhuZsdFQQpJwMwqwC7wpaDJDXAEvA8h/MmZeO6umVUD9oCW6/Ev0AX2gU+u2QrwDihHuQ4cAAvu2RnQDiEcZkUQYvKdBPwgxnEUQvgZO38JfHUyT0IIP5LnJ5nk7zM6uQwhvEuInxHlTeBR1oBRygZJvpMDsO/kN1JQM6sBS1GsSWY/U2ZmVTN7BuxG8XoULQNn0eJt5mkZFUzmVwjhc5SruOsq8D32bv9PArPRejtFdirxXZ6AGmC5iUwJxLnbAZYS0f9cIF4BD4E/d509D5xC88ATd2+U8gTUYgJ5G1FCAqDtF5GZPY6JrI5JYKLzK8o1YDt2cmJmhxPIP3b/7zq2HwFHsXwrhNBPCGFmj4DXUfQihHCeEUHPzGpZm5OZVcxsB/gWRRtZnc3EZdTMFvnXvBqZ/g1wAHzxL6qUVoEmMBvFf49K3m/8Hd87/QXcnAxURq2N3wAAAABJRU5ErkJggg==
    """,
    "exit": """
    iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAA7AAAAOwBeShxvQAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAJJSURBVFiF7ZdPiE1RGMafc+/MjOIPKSnlz9iQsrAQNQtbSzYWk42ysLJko5SVlIUyUmQzshEbc/fuLBCllDJJYWr+zOJ9zb3OPd+5dzQWHqfO4rvvOef3fO8957aoKnNJS3Pqzj8A841HZYRkd0EwC7wGCqC9+Vz1A5jTCJSZ73gKPADuAyNAH7A+2yIHwJxGIGa+X1Vbsb2JdgOjlXzPAeA/jcD8mYfPnTYBA6lrDgDVbKCXdbk6sdfJlhwAxQfq2nXgKHAeGA72+oH9wIR/7hKaV1VE5CJQiDLuqerh0J46C8aYPWG+L+q9P94IrPSFNRYtRdtLDV5qA23Armi5u8DZhH8L6I18tqnqp6ZV4EQC/Wdi79GfdR43RSXAN5M+1TfAuVj1JRAzf6uq39IZoGJMRTcROhWYj13GnHLm0QicFJGlqZ6LMd+ZGl0v8EpVZ7KGlpl7f4TbKj6IXcBEavRB5+ZKnXk0gBJYFcZ5ILZPC6CqnwMpD8TOV2d8VoQ9HwFvHKoXQFQs6F5VnepmXgPA+VcuE4pNdY98KuIL4FJnnnhZbDlzmLdVdVuQbwC4oap7IwDngPeuVbqZpwA4mX+o6pHI/zAwLOF5oDKYj+t0RNdcdvLcMoQqDgJ3nXl5FIgBzKX5N+CgMYbrATCoqtPdzFMA5jQCP4CDxpjh2GF/jPk8BDBnEVDVaWKDMWZLyrzSwCWd14c9/p9UZr4leBxY65C+r3QqlVYJx/GbIvJihuNzT78BcWXuUOWJF6YAAAAASUVORK5CYII=
    """
}


def create_icon_file(name, icon_data, target_dir="src/resources"):
    """
    Create an icon file from base64 data.
    
    Args:
        name (str): Icon name
        icon_data (str): Base64 encoded icon data
        target_dir (str): Target directory
        
    Returns:
        str: Path to the created icon file
    """
    # Ensure the directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Create the pixmap from base64 data
    icon_data = icon_data.strip()
    byte_array = QByteArray.fromBase64(icon_data.encode('utf-8'))
    pixmap = QPixmap()
    pixmap.loadFromData(byte_array)
    
    # Save the pixmap to a file
    file_path = os.path.join(target_dir, f"{name}.png")
    if not pixmap.save(file_path):
        logging.error(f"Failed to save icon: {file_path}")
        return None
    
    return file_path


def ensure_icons_exist():
    """
    Ensure that all icon files exist, creating them if necessary.
    
    Returns:
        bool: True if all icons exist or were created successfully, False otherwise
    """
    success = True
    resource_dir = "src/resources"
    
    for name, data in ICON_DATA.items():
        icon_path = os.path.join(resource_dir, f"{name}.png")
        if not os.path.exists(icon_path):
            if not create_icon_file(name, data, resource_dir):
                success = False
                logging.error(f"Failed to create icon: {name}")
    
    return success


# Ensure icons exist when this module is imported
if __name__ != "__main__":
    ensure_icons_exist() 