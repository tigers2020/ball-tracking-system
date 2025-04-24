#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
XML Logger module.
This module provides functionality for logging tracking data in XML format.
"""

import logging
import os
import time
import xml.etree.ElementTree as ET
import xml.dom.minidom
from typing import Dict, Any, Optional, Union, List


class XMLLogger:
    """
    Logger class for tracking data in XML format.
    Manages a session for recording frames and writing to disk.
    """
    
    def __init__(self, flush_interval: int = 100):
        """
        Initialize XML Logger.
        
        Args:
            flush_interval (int): Number of frames after which to flush to disk
        """
        self.root = None
        self.file_path = None
        self.frame_count = 0
        self.flush_interval = flush_interval
        self.is_session_active = False
    
    def start_session(self, folder: str, output_path: Optional[str] = None) -> bool:
        """
        Start a new tracking session.
        
        Args:
            folder (str): Name of the folder/session being tracked
            output_path (str, optional): Output directory for XML file
                                        Defaults to "tracking_data/{folder}"
                                        
        Returns:
            bool: Success or failure
        """
        try:
            # Set up file path
            if output_path is None:
                output_path = os.path.join(os.getcwd(), "tracking_data", folder)
            
            # Create directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)
            
            # Set file path
            self.file_path = os.path.join(output_path, "tracking_data.xml")
            
            # Check if the file already exists and load it if possible
            if os.path.exists(self.file_path):
                try:
                    # Try to parse the existing file
                    tree = ET.parse(self.file_path)
                    self.root = tree.getroot()
                    
                    # Get existing frame count
                    image_elements = self.root.findall("Image")
                    self.frame_count = len(image_elements)
                    
                    logging.info(f"Loaded existing XML tracking data from {self.file_path} with {self.frame_count} frames")
                    
                    # Update timestamp to indicate this is a resumed session
                    self.root.set("resumed", str(time.time()))
                    self.root.set("resume_time", time.strftime("%Y-%m-%d %H:%M:%S"))
                except Exception as e:
                    logging.warning(f"Could not load existing XML file, creating new one: {e}")
                    # If loading fails, create a new root element
                    self.root = ET.Element("TrackingData")
                    self.root.set("folder", folder)
                    self.root.set("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
                    self.root.set("created", str(time.time()))
                    self.frame_count = 0
            else:
                # Create a new root element if file doesn't exist
                self.root = ET.Element("TrackingData")
                self.root.set("folder", folder)
                self.root.set("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
                self.root.set("created", str(time.time()))
                self.frame_count = 0
            
            self.is_session_active = True
            
            logging.info(f"Started XML tracking session for folder: {folder}")
            logging.info(f"XML output will be saved to: {self.file_path}")
            
            # Flush immediately to disk to ensure file exists
            self.flush()
            
            return True
            
        except Exception as e:
            logging.error(f"Error starting XML tracking session: {e}")
            return False
    
    def log_frame(self, frame_number: int, data: Dict[str, Any], frame_name: Optional[str] = None) -> bool:
        """
        Log frame data to the XML structure.
        
        Args:
            frame_number (int): Frame number
            data (dict): Dictionary with tracking data
            frame_name (str, optional): Frame filename or identifier
            
        Returns:
            bool: Success or failure
        """
        if not self.is_session_active or self.root is None:
            logging.error("Cannot log frame: No active session. Call start_session first.")
            return False
            
        try:
            # Create image element
            image_elem = ET.SubElement(self.root, "Image")
            image_elem.set("number", str(frame_number))
            
            if frame_name:
                image_elem.set("name", frame_name)
                
            image_elem.set("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
            
            # Add tracking status if available
            if "tracking_active" in data:
                image_elem.set("tracking_active", str(data["tracking_active"]))
            
            # Process left camera data if available
            if "left" in data and data["left"]:
                left_data = data["left"]
                left_elem = ET.SubElement(image_elem, "Left")
                
                self._add_point_data(left_elem, left_data, "hsv_center", "HSV")
                self._add_point_data(left_elem, left_data, "hough_center", "Hough", include_radius=True)
                self._add_velocity_data(left_elem, left_data, "kalman_prediction", "Kalman")
                self._add_point_data(left_elem, left_data, "fused_center", "Fused")
            
            # Process right camera data if available
            if "right" in data and data["right"]:
                right_data = data["right"]
                right_elem = ET.SubElement(image_elem, "Right")
                
                self._add_point_data(right_elem, right_data, "hsv_center", "HSV")
                self._add_point_data(right_elem, right_data, "hough_center", "Hough", include_radius=True)
                self._add_velocity_data(right_elem, right_data, "kalman_prediction", "Kalman")
                self._add_point_data(right_elem, right_data, "fused_center", "Fused")
            
            # Increment frame count
            self.frame_count += 1
            
            # Always flush to disk after every frame to ensure real-time updates
            self.flush()
                
            return True
            
        except Exception as e:
            logging.error(f"Error logging frame {frame_number} to XML: {e}")
            return False
    
    def _add_point_data(self, parent_elem: ET.Element, data: Dict[str, Any], 
                        data_key: str, elem_name: str, include_radius: bool = False) -> None:
        """
        Add point data to an XML element.
        
        Args:
            parent_elem: Parent XML element
            data: Dictionary containing the data
            data_key: Key for the point data in the dictionary
            elem_name: Name for the XML element
            include_radius: Whether to include radius data
        """
        if data_key in data and data[data_key] is not None:
            point_data = data[data_key]
            point_elem = ET.SubElement(parent_elem, elem_name)
            
            # Add x, y coordinates
            point_elem.set("x", str(float(point_data["x"])))
            point_elem.set("y", str(float(point_data["y"])))
            
            # Add radius if available and requested
            if include_radius and "radius" in point_data:
                point_elem.set("radius", str(float(point_data["radius"])))
    
    def _add_velocity_data(self, parent_elem: ET.Element, data: Dict[str, Any], 
                           data_key: str, elem_name: str) -> None:
        """
        Add point data with velocity to an XML element.
        
        Args:
            parent_elem: Parent XML element
            data: Dictionary containing the data
            data_key: Key for the data in the dictionary
            elem_name: Name for the XML element
        """
        if data_key in data and data[data_key] is not None:
            kalman_data = data[data_key]
            # Skip invalid zero predictions
            try:
                x_val = float(kalman_data.get("x", 0))
                y_val = float(kalman_data.get("y", 0))
            except (ValueError, TypeError):
                return
            if x_val == 0.0 and y_val == 0.0:
                return
            kalman_elem = ET.SubElement(parent_elem, elem_name)
            
            # Add x, y coordinates
            kalman_elem.set("x", str(x_val))
            kalman_elem.set("y", str(y_val))
            
            # Add velocity if available
            if "vx" in kalman_data and "vy" in kalman_data:
                kalman_elem.set("vx", str(float(kalman_data["vx"])))
                kalman_elem.set("vy", str(float(kalman_data["vy"])))
    
    def add_statistics(self, stats: Dict[str, Any]) -> bool:
        """
        Add summary statistics to the XML.
        
        Args:
            stats (dict): Dictionary with statistics data
            
        Returns:
            bool: Success or failure
        """
        if not self.is_session_active or self.root is None:
            logging.error("Cannot add statistics: No active session. Call start_session first.")
            return False
            
        try:
            # Create stats element
            stats_elem = ET.SubElement(self.root, "Statistics")
            
            # Add statistics as attributes
            for key, value in stats.items():
                stats_elem.set(key, str(value))
                
            logging.info("Added statistics to XML tracking data")
            return True
            
        except Exception as e:
            logging.error(f"Error adding statistics to XML: {e}")
            return False
    
    def finalize_xml(self, processing_stats: Optional[Dict[str, Any]] = None) -> bool:
        """
        Finalize the XML file by adding total tracking time and processing statistics.
        
        Args:
            processing_stats (dict, optional): Dictionary with processing statistics
                                              such as processing_time, fps, etc.
            
        Returns:
            bool: Success or failure
        """
        if not self.is_session_active or self.root is None:
            logging.error("Cannot finalize XML: No active session. Call start_session first.")
            return False
            
        try:
            # Calculate tracking duration if the created attribute exists
            if "created" in self.root.attrib:
                try:
                    start_time = float(self.root.attrib["created"])
                    end_time = time.time()
                    duration_seconds = end_time - start_time
                    
                    # Add duration information
                    self.root.set("duration_seconds", str(round(duration_seconds, 2)))
                    self.root.set("duration_formatted", str(time.strftime(
                        "%H:%M:%S", time.gmtime(duration_seconds))))
                except (ValueError, TypeError):
                    logging.warning("Could not calculate tracking duration: invalid created timestamp")
            
            # Update final frame count
            self.root.set("total_frames", str(self.frame_count))
            self.root.set("finalized", str(time.time()))
            self.root.set("finalized_time", time.strftime("%Y-%m-%d %H:%M:%S"))
            
            # Add processing statistics if provided
            if processing_stats:
                # Create or get existing processing stats element
                stats_elems = self.root.findall("ProcessingStats")
                if stats_elems:
                    # Update existing element
                    stats_elem = stats_elems[0]
                else:
                    # Create new element
                    stats_elem = ET.SubElement(self.root, "ProcessingStats")
                
                # Add all stats as attributes
                for key, value in processing_stats.items():
                    stats_elem.set(key, str(value))
            
            # Flush to disk
            return self.flush()
            
        except Exception as e:
            logging.error(f"Error finalizing XML: {e}")
            return False
    
    def flush(self) -> bool:
        """
        Write current XML data to disk without closing the session.
        
        Returns:
            bool: Success or failure
        """
        if not self.is_session_active or self.root is None or self.file_path is None:
            logging.error("Cannot flush: No active session. Call start_session first.")
            return False
            
        try:
            # Create XML tree
            tree = ET.ElementTree(self.root)
            
            # Use minidom to pretty print the XML
            rough_string = ET.tostring(self.root, 'utf-8')
            reparsed = xml.dom.minidom.parseString(rough_string)
            pretty_xml = reparsed.toprettyxml(indent="  ")
            
            # Write to file
            with open(self.file_path, 'w', encoding='utf-8') as f:
                f.write(pretty_xml)
                
            logging.debug(f"Flushed XML tracking data to disk after {self.frame_count} frames")
            return True
            
        except Exception as e:
            logging.error(f"Error flushing XML tracking data: {e}")
            return False
    
    def close(self) -> bool:
        """
        Close the session and write final XML to disk.
        
        Returns:
            bool: Success or failure
        """
        if not self.is_session_active:
            logging.warning("No active session to close.")
            return False
            
        try:
            # Add final timestamp
            if self.root is not None:
                self.root.set("closed", str(time.time()))
                self.root.set("close_time", time.strftime("%Y-%m-%d %H:%M:%S"))
                
                # Ensure frame_count is an integer and set it as a string
                try:
                    frame_count = int(self.frame_count)
                    self.root.set("total_frames", str(frame_count))
                except (ValueError, TypeError):
                    logging.warning("Invalid frame count, setting total_frames to 0")
                    self.root.set("total_frames", "0")
            
            # Flush to disk
            result = self.flush()
            
            # Reset session state
            self.is_session_active = False
            
            logging.info(f"Closed XML tracking session with {self.frame_count} frames")
            return result
            
        except Exception as e:
            logging.error(f"Error closing XML tracking session: {e}")
            return False


class XMLLoggerAdapter:
    """
    Adapter class to provide backward compatibility with the previous frame-by-frame logging.
    This class wraps the XMLLogger to maintain the same interface as the previous JSON logger.
    """
    
    def __init__(self, xml_logger: XMLLogger = None):
        """
        Initialize the adapter.
        
        Args:
            xml_logger: XMLLogger instance to use, or create a new one if None
        """
        self.logger = xml_logger if xml_logger is not None else XMLLogger()
        self.current_folder = None
    
    def save_tracking_data_for_frame(self, frame_number: int, folder_path: Optional[str] = None) -> Optional[str]:
        """
        Save tracking data for a specific frame using XMLLogger.
        This method is designed to be compatible with the previous JSON-based interface.
        
        Args:
            frame_number: Current frame number
            folder_path: Path to the output folder
            
        Returns:
            str: Path to the XML file or None if failed
        """
        # Extract folder name from path if provided
        folder_name = os.path.basename(folder_path) if folder_path else "default"
        
        # Start a new session if needed
        if not self.logger.is_session_active or self.current_folder != folder_name:
            self.logger.start_session(folder_name, folder_path)
            self.current_folder = folder_name
        
        # Build data structure for the frame (to be implemented by derived class)
        frame_data = self._build_frame_data(frame_number)
        
        # Check if this frame already exists in the XML
        if self.logger.root is not None:
            existing_frames = self.logger.root.findall(f"Image[@number='{frame_number}']")
            if existing_frames:
                # Remove existing frame element to avoid duplicates
                for existing_frame in existing_frames:
                    self.logger.root.remove(existing_frame)
                    logging.debug(f"Removed existing frame {frame_number} from XML to avoid duplicates")
        
        # Log the frame
        frame_name = f"frame_{frame_number:06d}.png"
        success = self.logger.log_frame(frame_number, frame_data, frame_name)
        
        if success:
            return self.logger.file_path
        else:
            return None
    
    def _build_frame_data(self, frame_number: int) -> Dict[str, Any]:
        """
        Base implementation to build frame data dictionary.
        This should be overridden by derived classes to provide custom data.
        
        Args:
            frame_number: Frame number
            
        Returns:
            Dict containing basic frame data
        """
        # Default implementation returns an empty data structure
        return {
            "tracking_active": False,
            "left": {},
            "right": {}
        } 