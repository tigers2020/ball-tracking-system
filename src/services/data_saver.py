#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Saver module.
This module contains the DataSaver class for saving tracking data to files.
"""

import logging
import json
import os
import time
import xml.etree.ElementTree as ET
import xml.dom.minidom
from typing import Dict, List, Tuple, Optional, Any


class DataSaver:
    """
    Service class for saving tracking data to files.
    """
    
    def __init__(self):
        """Initialize the data saver."""
        self.xml_root = None
        self.current_folder = None
    
    def save_json_frame(self, frame_number: int, frame_data: Dict[str, Any], 
                       folder_path: Optional[str] = None) -> Optional[str]:
        """
        Save tracking data for a specific frame as JSON.
        
        Args:
            frame_number: Current frame number
            frame_data: Dictionary containing frame tracking data
            folder_path: Path to the output folder
            
        Returns:
            Path to the saved file or None if failed
        """
        try:
            # Set default folder path if not provided
            if folder_path is None:
                folder_path = os.path.join(os.getcwd(), "tracking_data")
            
            # Create folder if it doesn't exist
            os.makedirs(folder_path, exist_ok=True)
            
            # Create filename using frame number to ensure overwrite of same frame data
            filename = f"frame_{frame_number:06d}.json"
            
            # Combine folder path and filename
            file_path = os.path.join(folder_path, filename)
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(frame_data, f, indent=2)
            
            logging.debug(f"Frame {frame_number} tracking data saved to {file_path}")
            return file_path
            
        except Exception as e:
            logging.error(f"Error saving frame tracking data: {e}")
            return None
    
    def save_json_summary(self, tracking_data: Dict[str, Any], 
                         folder_path: Optional[str] = None, 
                         filename: Optional[str] = None) -> Optional[str]:
        """
        Save summary tracking data to a JSON file.
        
        Args:
            tracking_data: Dictionary containing tracking data
            folder_path: Path to the output folder
            filename: Base filename without extension
        
        Returns:
            Path to the saved file or None if failed
        """
        try:
            # Set default folder path if not provided
            if folder_path is None:
                folder_path = os.path.join(os.getcwd(), "tracking_data")
            
            # Create folder if it doesn't exist
            os.makedirs(folder_path, exist_ok=True)
            
            # Set default filename if not provided (use timestamp)
            if filename is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"tracking_data_{timestamp}.json"
            elif not filename.endswith('.json'):
                filename = f"{filename}.json"
            
            # Combine folder path and filename
            file_path = os.path.join(folder_path, filename)
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(tracking_data, f, indent=2)
            
            logging.info(f"Tracking data saved to {file_path}")
            return file_path
            
        except Exception as e:
            logging.error(f"Error saving tracking data: {e}")
            return None
    
    def initialize_xml_tracking(self, folder_name: str) -> bool:
        """
        Initialize XML tracking data structure for a new folder.
        If an existing XML file exists, it will be loaded to continue tracking.
        
        Args:
            folder_name: Name of the folder being processed
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Set up file path
            tracking_folder = os.path.join(os.getcwd(), "tracking_data", folder_name)
            os.makedirs(tracking_folder, exist_ok=True)
            xml_file_path = os.path.join(tracking_folder, "tracking_data.xml")
            
            # Check if the file already exists
            if os.path.exists(xml_file_path):
                try:
                    # Try to parse the existing file
                    tree = ET.parse(xml_file_path)
                    self.xml_root = tree.getroot()
                    
                    # Get existing frame count
                    image_elements = self.xml_root.findall("Image")
                    frame_count = len(image_elements)
                    
                    # Store the current folder
                    self.current_folder = folder_name
                    
                    logging.info(f"Loaded existing XML tracking data for folder '{folder_name}' with {frame_count} frames")
                    
                    # Update timestamp to indicate this is a resumed session
                    self.xml_root.set("resumed", str(time.time()))
                    self.xml_root.set("resume_time", time.strftime("%Y-%m-%d %H:%M:%S"))
                    
                    return True
                except Exception as e:
                    logging.warning(f"Failed to load existing XML file, creating new one: {e}")
                    # Fall through to create new file
            
            # Create new root element
            self.xml_root = ET.Element("TrackingData")
            self.xml_root.set("folder", folder_name)
            self.xml_root.set("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
            self.xml_root.set("created", str(time.time()))
            
            # Store the current folder
            self.current_folder = folder_name
            
            # Immediately save to disk to ensure file exists
            self.save_xml_tracking_data()
            
            logging.info(f"Initialized new XML tracking data for folder: {folder_name}")
            return True
            
        except Exception as e:
            logging.error(f"Error initializing XML tracking: {e}")
            return False
    
    def save_frame_to_xml(self, frame_number: int, frame_data: Dict[str, Any], 
                         frame_name: Optional[str] = None) -> bool:
        """
        Save the current frame's tracking data to the XML structure.
        
        Args:
            frame_number: Frame number
            frame_data: Dictionary containing frame data
            frame_name: Frame filename if available
            
        Returns:
            Success or failure
        """
        if self.xml_root is None:
            logging.error("XML tracking not initialized. Call initialize_xml_tracking first.")
            return False
            
        try:
            # Create frame/image element
            image_elem = ET.SubElement(self.xml_root, "Image")
            image_elem.set("number", str(frame_number))
            if frame_name:
                image_elem.set("name", frame_name)
            image_elem.set("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
            image_elem.set("tracking_active", str(frame_data.get("tracking_active", False)))
            
            # Process left camera data
            left_elem = ET.SubElement(image_elem, "Left")
            left_data = frame_data.get("left", {})
            
            # HSV mask centroid
            if left_data.get("hsv_center"):
                hsv_elem = ET.SubElement(left_elem, "HSV")
                hsv_elem.set("x", str(float(left_data["hsv_center"]["x"])))
                hsv_elem.set("y", str(float(left_data["hsv_center"]["y"])))
            
            # Hough circle
            if left_data.get("hough_center"):
                hough_elem = ET.SubElement(left_elem, "Hough")
                hough_elem.set("x", str(float(left_data["hough_center"]["x"])))
                hough_elem.set("y", str(float(left_data["hough_center"]["y"])))
                if "radius" in left_data["hough_center"]:
                    hough_elem.set("radius", str(float(left_data["hough_center"]["radius"])))
            
            # Kalman prediction
            if left_data.get("kalman_prediction"):
                kalman_elem = ET.SubElement(left_elem, "Kalman")
                kalman_elem.set("x", str(left_data["kalman_prediction"]["x"]))
                kalman_elem.set("y", str(left_data["kalman_prediction"]["y"]))
                if "vx" in left_data["kalman_prediction"]:
                    kalman_elem.set("vx", str(left_data["kalman_prediction"]["vx"]))
                    kalman_elem.set("vy", str(left_data["kalman_prediction"]["vy"]))
            
            # Fused coordinate
            if left_data.get("fused_center"):
                fused_elem = ET.SubElement(left_elem, "Fused")
                fused_elem.set("x", str(float(left_data["fused_center"]["x"])))
                fused_elem.set("y", str(float(left_data["fused_center"]["y"])))
            
            # Process right camera data
            right_elem = ET.SubElement(image_elem, "Right")
            right_data = frame_data.get("right", {})
            
            # HSV mask centroid
            if right_data.get("hsv_center"):
                hsv_elem = ET.SubElement(right_elem, "HSV")
                hsv_elem.set("x", str(float(right_data["hsv_center"]["x"])))
                hsv_elem.set("y", str(float(right_data["hsv_center"]["y"])))
            
            # Hough circle
            if right_data.get("hough_center"):
                hough_elem = ET.SubElement(right_elem, "Hough")
                hough_elem.set("x", str(float(right_data["hough_center"]["x"])))
                hough_elem.set("y", str(float(right_data["hough_center"]["y"])))
                if "radius" in right_data["hough_center"]:
                    hough_elem.set("radius", str(float(right_data["hough_center"]["radius"])))
            
            # Kalman prediction
            if right_data.get("kalman_prediction"):
                kalman_elem = ET.SubElement(right_elem, "Kalman")
                kalman_elem.set("x", str(right_data["kalman_prediction"]["x"]))
                kalman_elem.set("y", str(right_data["kalman_prediction"]["y"]))
                if "vx" in right_data["kalman_prediction"]:
                    kalman_elem.set("vx", str(right_data["kalman_prediction"]["vx"]))
                    kalman_elem.set("vy", str(right_data["kalman_prediction"]["vy"]))
            
            # Fused coordinate
            if right_data.get("fused_center"):
                fused_elem = ET.SubElement(right_elem, "Fused")
                fused_elem.set("x", str(float(right_data["fused_center"]["x"])))
                fused_elem.set("y", str(float(right_data["fused_center"]["y"])))
            
            logging.debug(f"Added frame {frame_number} to XML tracking data")
            
            # Real-time update: write XML tracking data file after each frame
            self.save_xml_tracking_data()
            return True
            
        except Exception as e:
            logging.error(f"Error saving frame to XML: {e}")
            return False
    
    def save_xml_tracking_data(self, folder_path: Optional[str] = None) -> Optional[str]:
        """
        Save the XML tracking data to a file.
        
        Args:
            folder_path: Path to the output folder. 
                        Default is 'tracking_data/{current_folder}'.
            
        Returns:
            Path to the saved file or None if failed
        """
        if self.xml_root is None:
            logging.error("XML tracking not initialized. Call initialize_xml_tracking first.")
            return None
            
        try:
            # Set default folder path if not provided
            if folder_path is None:
                if self.current_folder is None:
                    folder_path = os.path.join(os.getcwd(), "tracking_data", "default")
                else:
                    folder_path = os.path.join(os.getcwd(), "tracking_data", self.current_folder)
            
            # Create folder if it doesn't exist
            os.makedirs(folder_path, exist_ok=True)
            
            # Create the output file path
            output_file = os.path.join(folder_path, "tracking_data.xml")
            
            # Remove any existing Statistics elements to avoid duplicates
            for existing_stats in self.xml_root.findall("Statistics"):
                self.xml_root.remove(existing_stats)
            
            # Add summary statistics (optional - would come from model)
            if hasattr(self, 'stats') and isinstance(getattr(self, 'stats'), dict):
                stats_elem = ET.SubElement(self.xml_root, "Statistics")
                stats = getattr(self, 'stats')
                for key, value in stats.items():
                    stats_elem.set(key, str(value))
            
            # Create XML tree and write to file
            tree = ET.ElementTree(self.xml_root)
            
            # Use minidom to pretty print the XML
            rough_string = ET.tostring(self.xml_root, 'utf-8')
            reparsed = xml.dom.minidom.parseString(rough_string)
            pretty_xml = reparsed.toprettyxml(indent="  ")
            
            # Remove blank lines to avoid excessive empty lines
            pretty_lines = [line for line in pretty_xml.split('\n') if line.strip()]
            pretty_xml = '\n'.join(pretty_lines)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(pretty_xml)
            
            logging.info(f"Saved XML tracking data to {output_file}")
            return output_file
            
        except Exception as e:
            logging.error(f"Error saving XML tracking data: {e}")
            return None 