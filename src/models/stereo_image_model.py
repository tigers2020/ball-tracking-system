#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stereo Image Model module.
This module contains the StereoImageModel class which handles the data model for stereo images.
"""

import os
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
import concurrent.futures

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

from src.utils.ui_constants import XML, Messages


class StereoFrame:
    """
    Represents a single stereo frame with left and right images.
    """
    
    def __init__(self, index, left_image_path=None, right_image_path=None):
        """
        Initialize a stereo frame.
        
        Args:
            index (int): Frame index
            left_image_path (str, optional): Path to the left image
            right_image_path (str, optional): Path to the right image
        """
        self.index = index
        self.left_image_path = left_image_path
        self.right_image_path = right_image_path
        self._left_image = None
        self._right_image = None
    
    def load_images(self):
        """
        Load both left and right images using OpenCV.
        
        Returns:
            tuple: (left_image, right_image) as numpy arrays, or (None, None) if loading fails
        """
        try:
            if self.left_image_path and os.path.exists(self.left_image_path):
                self._left_image = cv2.imread(self.left_image_path)
            
            if self.right_image_path and os.path.exists(self.right_image_path):
                self._right_image = cv2.imread(self.right_image_path)
                
            return self._left_image, self._right_image
        except Exception as e:
            logging.error(f"Error loading images for frame {self.index}: {str(e)}")
            return None, None
    
    def get_left_image(self):
        """
        Get the left image, loading it if necessary.
        
        Returns:
            numpy.ndarray: The left image as a numpy array, or None if loading fails
        """
        if self._left_image is None and self.left_image_path:
            try:
                self._left_image = cv2.imread(self.left_image_path)
            except Exception as e:
                logging.error(f"Error loading left image for frame {self.index}: {str(e)}")
                return None
        return self._left_image
    
    def get_right_image(self):
        """
        Get the right image, loading it if necessary.
        
        Returns:
            numpy.ndarray: The right image as a numpy array, or None if loading fails
        """
        if self._right_image is None and self.right_image_path:
            try:
                self._right_image = cv2.imread(self.right_image_path)
            except Exception as e:
                logging.error(f"Error loading right image for frame {self.index}: {str(e)}")
                return None
        return self._right_image
    
    def release_images(self):
        """Release loaded images to free memory."""
        self._left_image = None
        self._right_image = None


class StereoImageModel(QObject):
    """
    Model class for handling stereo images and XML data.
    """
    
    # Signals
    loading_progress = Signal(int, int)  # current, total
    loading_complete = Signal()
    loading_error = Signal(str)
    frame_changed = Signal(int)
    
    def __init__(self):
        """Initialize the stereo image model."""
        super(StereoImageModel, self).__init__()
        self.frames = []
        self.current_frame_index = 0
        self.total_frames = 0
        self.base_folder = None
        self.xml_path = None
        self.is_playing = False
    
    def load_from_folder(self, folder_path):
        """
        Load stereo images from a folder. Checks for frames_info.xml or creates it if it doesn't exist.
        
        Args:
            folder_path (str): Path to the folder containing the stereo images
            
        Returns:
            bool: True if loading is successful, False otherwise
        """
        self.base_folder = folder_path
        self.xml_path = os.path.join(folder_path, XML.FRAMES_INFO_FILENAME)
        
        # Clear current frames
        self.frames = []
        self.current_frame_index = 0
        
        # Check if XML file exists
        if os.path.exists(self.xml_path):
            return self._load_from_xml()
        else:
            # Generate XML from folder contents
            return self._generate_xml_from_folder()
    
    def _load_from_xml(self):
        """
        Load frames information from the XML file.
        
        Returns:
            bool: True if loading is successful, False otherwise
        """
        try:
            tree = ET.parse(self.xml_path)
            root = tree.getroot()
            
            # Support both 'frames' and 'stereo_frames' as root elements
            if root.tag != XML.ROOT_ELEMENT and root.tag != "frames":
                logging.error(f"Invalid XML root element: {root.tag}")
                self.loading_error.emit(Messages.ERROR_PARSING_XML)
                return False
            
            # Find all frame elements
            frame_elements = root.findall("frame")
            self.total_frames = len(frame_elements)
            
            if self.total_frames == 0:
                logging.warning("No frame elements found in XML")
                self.loading_error.emit(Messages.NO_IMAGES_FOUND)
                return False
            
            for i, frame_elem in enumerate(frame_elements):
                try:
                    # Try to get index/id attribute
                    if "id" in frame_elem.attrib:
                        index = int(frame_elem.get("id"))
                    elif XML.INDEX_ATTR in frame_elem.attrib:
                        index = int(frame_elem.get(XML.INDEX_ATTR))
                    else:
                        index = i
                    
                    # Different XML structures require different parsing
                    left_path = None
                    right_path = None
                    
                    # Structure 1: <left_image path="..."/> and <right_image path="..."/>
                    left_elem = frame_elem.find(XML.LEFT_IMAGE_ELEMENT)
                    right_elem = frame_elem.find(XML.RIGHT_IMAGE_ELEMENT)
                    
                    if left_elem is not None and right_elem is not None:
                        left_path = left_elem.get(XML.PATH_ATTR)
                        right_path = right_elem.get(XML.PATH_ATTR)
                    else:
                        # Structure 2: <camera type="LeftCamera"><raw_path>...</raw_path></camera>
                        for camera_elem in frame_elem.findall("camera"):
                            camera_type = camera_elem.get("type")
                            
                            if camera_type == "LeftCamera":
                                # Try raw_path first, then resize_path
                                raw_path_elem = camera_elem.find("raw_path")
                                resize_path_elem = camera_elem.find("resize_path")
                                
                                if raw_path_elem is not None and raw_path_elem.text:
                                    left_path = raw_path_elem.text
                                elif resize_path_elem is not None and resize_path_elem.text:
                                    left_path = resize_path_elem.text
                            
                            elif camera_type == "RightCamera":
                                # Try raw_path first, then resize_path
                                raw_path_elem = camera_elem.find("raw_path")
                                resize_path_elem = camera_elem.find("resize_path")
                                
                                if raw_path_elem is not None and raw_path_elem.text:
                                    right_path = raw_path_elem.text
                                elif resize_path_elem is not None and resize_path_elem.text:
                                    right_path = resize_path_elem.text
                    
                    # Skip if missing either path
                    if not left_path or not right_path:
                        logging.warning(f"Missing image path for frame {index}")
                        continue
                    
                    # Make paths absolute if they are relative
                    if not os.path.isabs(left_path):
                        left_path = os.path.join(self.base_folder, left_path)
                    if not os.path.isabs(right_path):
                        right_path = os.path.join(self.base_folder, right_path)
                    
                    # Check if files exist
                    if not os.path.exists(left_path) or not os.path.exists(right_path):
                        logging.warning(f"Missing image file for frame {index}")
                        continue
                    
                    frame = StereoFrame(index, left_path, right_path)
                    self.frames.append(frame)
                    
                    # Emit progress
                    self.loading_progress.emit(i + 1, self.total_frames)
                
                except Exception as e:
                    logging.error(f"Error parsing frame {i}: {str(e)}")
                    continue
            
            if not self.frames:
                logging.warning("No valid frames found")
                self.loading_error.emit(Messages.NO_IMAGES_FOUND)
                return False
            
            self.loading_complete.emit()
            return True
            
        except ET.ParseError as e:
            logging.error(f"Error parsing XML: {str(e)}")
            self.loading_error.emit(Messages.ERROR_PARSING_XML)
            return False
        except Exception as e:
            logging.error(f"Error loading XML: {str(e)}")
            self.loading_error.emit(Messages.ERROR_LOADING_XML)
            return False
    
    def _generate_xml_from_folder(self):
        """
        Generate frames_info.xml from folder contents, looking for stereo image pairs.
        
        Returns:
            bool: True if generation is successful, False otherwise
        """
        try:
            # Find all image files in the directory
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(list(Path(self.base_folder).glob(f"*{ext}")))
                image_files.extend(list(Path(self.base_folder).glob(f"*{ext.upper()}")))
            
            if not image_files:
                logging.warning("No image files found in the selected folder")
                self.loading_error.emit(Messages.NO_IMAGES_FOUND)
                return False
            
            # Sort files by name
            image_files.sort()
            
            # Group images into left/right pairs (basic implementation)
            # This assumes files are named with _L and _R suffixes, or left/right in the name
            left_images = [f for f in image_files if '_L' in f.name or 'left' in f.name.lower()]
            right_images = [f for f in image_files if '_R' in f.name or 'right' in f.name.lower()]
            
            # If no explicit left/right naming, try to pair them by order (even/odd indices)
            if not left_images and not right_images and len(image_files) >= 2:
                left_images = image_files[::2]  # Even indices
                right_images = image_files[1::2]  # Odd indices
            
            # Create XML structure
            root = ET.Element(XML.ROOT_ELEMENT)
            
            self.total_frames = min(len(left_images), len(right_images))
            for i in range(self.total_frames):
                frame_elem = ET.SubElement(root, XML.FRAME_ELEMENT)
                frame_elem.set(XML.INDEX_ATTR, str(i))
                
                left_elem = ET.SubElement(frame_elem, XML.LEFT_IMAGE_ELEMENT)
                left_elem.set(XML.PATH_ATTR, str(left_images[i].relative_to(self.base_folder)))
                
                right_elem = ET.SubElement(frame_elem, XML.RIGHT_IMAGE_ELEMENT)
                right_elem.set(XML.PATH_ATTR, str(right_images[i].relative_to(self.base_folder)))
                
                # Create and add frame to the model
                left_path = str(left_images[i])
                right_path = str(right_images[i])
                frame = StereoFrame(i, left_path, right_path)
                self.frames.append(frame)
                
                # Emit progress
                self.loading_progress.emit(i + 1, self.total_frames)
            
            if not self.frames:
                logging.warning("No stereo image pairs found")
                self.loading_error.emit(Messages.NO_IMAGES_FOUND)
                return False
            
            # Save XML file for future use
            self.save_xml()
            
            self.loading_complete.emit()
            return True
            
        except Exception as e:
            logging.error(f"Error generating XML: {str(e)}")
            self.loading_error.emit(str(e))
            return False
    
    def get_current_frame(self):
        """
        Get the current frame.
        
        Returns:
            StereoFrame: The current frame, or None if no frames are loaded
        """
        if not self.frames or self.current_frame_index < 0 or self.current_frame_index >= len(self.frames):
            return None
        
        return self.frames[self.current_frame_index]
    
    def get_frame(self, index):
        """
        Get a frame by index.
        
        Args:
            index (int): Frame index
            
        Returns:
            StereoFrame: The frame at the specified index, or None if the index is out of bounds
        """
        if not self.frames or index < 0 or index >= len(self.frames):
            return None
        
        return self.frames[index]
    
    def set_current_frame_index(self, index):
        """
        Set the current frame index and emit the frame_changed signal.
        
        Args:
            index (int): The new frame index
        """
        if 0 <= index < len(self.frames):
            self.current_frame_index = index
            self.frame_changed.emit(index)
    
    def next_frame(self):
        """
        Move to the next frame and return it.
        
        Returns:
            StereoFrame: The next frame, or None if at the end
        """
        if self.current_frame_index < len(self.frames) - 1:
            self.current_frame_index += 1
            self.frame_changed.emit(self.current_frame_index)
            return self.frames[self.current_frame_index]
        return None
    
    def prev_frame(self):
        """
        Move to the previous frame and return it.
        
        Returns:
            StereoFrame: The previous frame, or None if at the beginning
        """
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.frame_changed.emit(self.current_frame_index)
            return self.frames[self.current_frame_index]
        return None
    
    def preload_frames(self, num_frames=5):
        """
        Preload a number of frames ahead of the current frame.
        
        Args:
            num_frames (int): Number of frames to preload
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i in range(self.current_frame_index, min(self.current_frame_index + num_frames, len(self.frames))):
                futures.append(executor.submit(self.frames[i].load_images))
            
            # Wait for all threads to complete
            concurrent.futures.wait(futures)
    
    def save_xml(self):
        """
        Save the current frames information to the XML file.
        
        Returns:
            bool: True if saving is successful, False otherwise
        """
        try:
            root = ET.Element(XML.ROOT_ELEMENT)
            
            for frame in self.frames:
                frame_elem = ET.SubElement(root, XML.FRAME_ELEMENT)
                frame_elem.set(XML.INDEX_ATTR, str(frame.index))
                
                if frame.left_image_path:
                    left_elem = ET.SubElement(frame_elem, XML.LEFT_IMAGE_ELEMENT)
                    # Convert to relative path if it's inside the base folder
                    rel_path = os.path.relpath(frame.left_image_path, self.base_folder)
                    left_elem.set(XML.PATH_ATTR, rel_path)
                
                if frame.right_image_path:
                    right_elem = ET.SubElement(frame_elem, XML.RIGHT_IMAGE_ELEMENT)
                    # Convert to relative path if it's inside the base folder
                    rel_path = os.path.relpath(frame.right_image_path, self.base_folder)
                    right_elem.set(XML.PATH_ATTR, rel_path)
            
            tree = ET.ElementTree(root)
            tree.write(self.xml_path, encoding='utf-8', xml_declaration=True)
            return True
            
        except Exception as e:
            logging.error(f"Error saving XML: {str(e)}")
            return False
    
    def release_all_frames(self):
        """Release all loaded frames to free memory."""
        for frame in self.frames:
            frame.release_images() 