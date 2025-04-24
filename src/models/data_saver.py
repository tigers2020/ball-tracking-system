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
import io
import xml.sax.saxutils
import atexit
from typing import Dict, List, Tuple, Optional, Any


class DataSaver:
    """
    Service class for saving tracking data to files.
    """
    
    def __init__(self):
        """Initialize the data saver."""
        self.xml_root = None
        self.current_folder = None
        self._log_fp: Optional[io.TextIOWrapper] = None
        self._last_frame = -1
    
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
            
            # Close any existing file handler
            if self._log_fp:
                self.finalize_xml()
            
            # Check if the file already exists
            if os.path.exists(xml_file_path):
                # 기존 파일이 있으면 먼저 복구를 시도합니다
                self._repair_xml(xml_file_path)
                
                # 존재하는 파일에 추가 모드로 열기
                try:
                    # 파일의 마지막 행을 확인하여 닫는 태그가 이미 있는지 체크
                    with open(xml_file_path, 'r', encoding='utf-8') as check_file:
                        content = check_file.readlines()
                        
                    # 닫는 태그가 이미 존재하면 제거 (이후 추가하기 위해)
                    if content and content[-1].strip() == "</TrackingData>":
                        with open(xml_file_path, 'w', encoding='utf-8') as truncate_file:
                            truncate_file.writelines(content[:-1])
                    
                    # 기존 XML을 파싱하여 구조 파악
                    tree = ET.parse(xml_file_path)
                    self.xml_root = tree.getroot()
                    
                    # 마지막 프레임 번호 찾기
                    image_elements = self.xml_root.findall("Image")
                    if image_elements:
                        last_frame_nums = [int(img.get("number", "-1")) for img in image_elements]
                        self._last_frame = max(last_frame_nums) if last_frame_nums else -1
                    
                    # 파일을 추가 모드로 열기
                    self._log_fp = open(xml_file_path, 'a', buffering=1, encoding='utf-8')
                    
                    # 현재 폴더 저장
                    self.current_folder = folder_name
                    
                    logging.info(f"Resumed XML tracking for '{folder_name}' with last frame {self._last_frame}")
                    
                    # XML 루트 객체에 세션 재개 정보 추가
                    self.xml_root.set("resumed", str(time.time()))
                    self.xml_root.set("resume_time", time.strftime("%Y-%m-%d %H:%M:%S"))
                    
                    return True
                except Exception as e:
                    logging.warning(f"Failed to load existing XML file, creating new one: {e}")
                    # Fall through to create new file
            
            # 새 파일 생성 (라인 버퍼링 활성화)
            self._log_fp = open(xml_file_path, 'w', buffering=1, encoding='utf-8')
            
            # 헤더와 루트 요소 작성
            header = '<?xml version="1.0" encoding="utf-8"?>\n'
            root_start = f'<TrackingData folder="{folder_name}" timestamp="{time.strftime("%Y-%m-%d %H:%M:%S")}" created="{time.time()}">\n'
            self._log_fp.write(header)
            self._log_fp.write(root_start)
            self._log_fp.flush()  # 즉시 디스크에 기록
            
            # XML 루트 객체도 초기화 (메모리 내 작업용)
            self.xml_root = ET.Element("TrackingData")
            self.xml_root.set("folder", folder_name)
            self.xml_root.set("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
            self.xml_root.set("created", str(time.time()))
            
            # 현재 폴더 저장
            self.current_folder = folder_name
            
            # Register finalize method to be called on program exit
            atexit.register(self.finalize_xml)
            
            logging.info(f"Initialized new XML tracking data stream for folder: {folder_name}")
            return True
            
        except Exception as e:
            logging.error(f"Error initializing XML tracking: {e}")
            return False
    
    def _build_image_elem(self, frame_number: int, frame_data: Dict[str, Any], 
                         frame_name: Optional[str] = None) -> str:
        """
        Build an XML Image element string for a frame without adding it to the root.
        
        Args:
            frame_number: Frame number
            frame_data: Dictionary containing frame data
            frame_name: Frame filename if available
            
        Returns:
            String representation of the Image element
        """
        try:
            # Space indentation for readability
            indent = "  "
            
            # Start Image element
            image_elem = f"{indent}<Image number=\"{frame_number}\""
            if frame_name:
                image_elem += f" name=\"{xml.sax.saxutils.escape(frame_name)}\""
            image_elem += f" timestamp=\"{time.strftime('%Y-%m-%d %H:%M:%S')}\""
            image_elem += f" tracking_active=\"{str(frame_data.get('tracking_active', False)).lower()}\""
            
            # Check if there is any data to add
            has_data = False
            
            # Process left camera data
            left_data = frame_data.get("left", {})
            left_elem = ""
            
            # HSV mask centroid
            if left_data.get("hsv_center"):
                has_data = True
                left_elem += f"\n{indent*2}<HSV"
                left_elem += f" x=\"{float(left_data['hsv_center']['x'])}\""
                left_elem += f" y=\"{float(left_data['hsv_center']['y'])}\""
                left_elem += "/>"
            
            # Hough circle
            if left_data.get("hough_center"):
                has_data = True
                left_elem += f"\n{indent*2}<Hough"
                left_elem += f" x=\"{float(left_data['hough_center']['x'])}\""
                left_elem += f" y=\"{float(left_data['hough_center']['y'])}\""
                if "radius" in left_data["hough_center"]:
                    left_elem += f" radius=\"{float(left_data['hough_center']['radius'])}\""
                left_elem += "/>"
            
            # Kalman prediction
            if left_data.get("kalman_prediction"):
                has_data = True
                left_elem += f"\n{indent*2}<Kalman"
                left_elem += f" x=\"{left_data['kalman_prediction']['x']}\""
                left_elem += f" y=\"{left_data['kalman_prediction']['y']}\""
                if "vx" in left_data["kalman_prediction"]:
                    left_elem += f" vx=\"{left_data['kalman_prediction']['vx']}\""
                    left_elem += f" vy=\"{left_data['kalman_prediction']['vy']}\""
                left_elem += "/>"
            
            # Fused coordinate
            if left_data.get("fused_center"):
                has_data = True
                left_elem += f"\n{indent*2}<Fused"
                left_elem += f" x=\"{float(left_data['fused_center']['x'])}\""
                left_elem += f" y=\"{float(left_data['fused_center']['y'])}\""
                left_elem += "/>"
            
            # Process right camera data
            right_data = frame_data.get("right", {})
            right_elem = ""
            
            # HSV mask centroid
            if right_data.get("hsv_center"):
                has_data = True
                right_elem += f"\n{indent*2}<HSV"
                right_elem += f" x=\"{float(right_data['hsv_center']['x'])}\""
                right_elem += f" y=\"{float(right_data['hsv_center']['y'])}\""
                right_elem += "/>"
            
            # Hough circle
            if right_data.get("hough_center"):
                has_data = True
                right_elem += f"\n{indent*2}<Hough"
                right_elem += f" x=\"{float(right_data['hough_center']['x'])}\""
                right_elem += f" y=\"{float(right_data['hough_center']['y'])}\""
                if "radius" in right_data["hough_center"]:
                    right_elem += f" radius=\"{float(right_data['hough_center']['radius'])}\""
                right_elem += "/>"
            
            # Kalman prediction
            if right_data.get("kalman_prediction"):
                has_data = True
                right_elem += f"\n{indent*2}<Kalman"
                right_elem += f" x=\"{right_data['kalman_prediction']['x']}\""
                right_elem += f" y=\"{right_data['kalman_prediction']['y']}\""
                if "vx" in right_data["kalman_prediction"]:
                    right_elem += f" vx=\"{right_data['kalman_prediction']['vx']}\""
                    right_elem += f" vy=\"{right_data['kalman_prediction']['vy']}\""
                right_elem += "/>"
            
            # Fused coordinate
            if right_data.get("fused_center"):
                has_data = True
                right_elem += f"\n{indent*2}<Fused"
                right_elem += f" x=\"{float(right_data['fused_center']['x'])}\""
                right_elem += f" y=\"{float(right_data['fused_center']['y'])}\""
                right_elem += "/>"
            
            # If we have data, create full nested elements
            if has_data:
                full_elem = image_elem + ">"
                if left_elem:
                    full_elem += f"\n{indent*1}<Left>{left_elem}\n{indent*1}</Left>"
                if right_elem:
                    full_elem += f"\n{indent*1}<Right>{right_elem}\n{indent*1}</Right>"
                full_elem += f"\n{indent}</Image>\n"
            else:
                # Self-closing tag if no data
                full_elem = image_elem + "/>\n"
            
            return full_elem
            
        except Exception as e:
            logging.error(f"Error building Image element: {e}")
            return ""
    
    def save_frame_to_xml(self, frame_number: int, frame_data: Dict[str, Any], 
                         frame_name: Optional[str] = None, overwrite: bool = False) -> bool:
        """
        Save the current frame's tracking data to the XML stream.
        
        Args:
            frame_number: Frame number
            frame_data: Dictionary containing frame data
            frame_name: Frame filename if available
            overwrite: Whether to overwrite if the frame already exists
            
        Returns:
            Success or failure
        """
        if self._log_fp is None:
            logging.error("XML stream not open. Call initialize_xml_tracking first.")
            return False
        
        # 프레임 중복 방지
        if not overwrite and frame_number <= self._last_frame:
            logging.debug(f"Skipping duplicate frame {frame_number} (last saved: {self._last_frame})")
            return True
            
        try:
            # Build the Image element string
            image_elem_str = self._build_image_elem(frame_number, frame_data, frame_name)
            
            # Write to the file stream
            self._log_fp.write(image_elem_str)
            self._log_fp.flush()  # Force buffer to disk
            
            # XML 메모리 표현에도 추가 (통계나 다른 용도로 사용)
            if self.xml_root is not None:
                image_elem = ET.SubElement(self.xml_root, "Image")
                image_elem.set("number", str(frame_number))
                if frame_name:
                    image_elem.set("name", frame_name)
                image_elem.set("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
                image_elem.set("tracking_active", str(frame_data.get("tracking_active", False)))
                
                # Process left camera data
                left_data = frame_data.get("left", {})
                if left_data:
                    left_elem = ET.SubElement(image_elem, "Left")
                    
                    # HSV mask centroid
                    if left_data.get("hsv_center"):
                        hsv_elem = ET.SubElement(left_elem, "HSV")
                        hsv_elem.set("x", str(float(left_data["hsv_center"]["x"])))
                        hsv_elem.set("y", str(float(left_data["hsv_center"]["y"])))
                    
                    # Hough circle (다른 요소들도 유사하게 처리...)
                    if left_data.get("hough_center"):
                        hough_elem = ET.SubElement(left_elem, "Hough")
                        hough_elem.set("x", str(float(left_data["hough_center"]["x"])))
                        hough_elem.set("y", str(float(left_data["hough_center"]["y"])))
                        if "radius" in left_data["hough_center"]:
                            hough_elem.set("radius", str(float(left_data["hough_center"]["radius"])))
                
                # 나머지 요소들을 메모리 XML에 추가...
            
            # 마지막 프레임 번호 업데이트
            self._last_frame = max(self._last_frame, frame_number)
            
            logging.debug(f"Added frame {frame_number} to XML tracking data stream")
            return True
            
        except Exception as e:
            logging.error(f"Error saving frame to XML: {e}")
            return False
    
    def save_xml_tracking_data(self, folder_path: Optional[str] = None) -> Optional[str]:
        """
        Save a complete copy of the in-memory XML tracking data to a file.
        This is only needed for special cases, as the streaming XML is
        already being written incrementally.
        
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
            output_file = os.path.join(folder_path, "tracking_data_snapshot.xml")
            
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
            
            logging.info(f"Saved XML tracking data snapshot to {output_file}")
            return output_file
            
        except Exception as e:
            logging.error(f"Error saving XML tracking data snapshot: {e}")
            return None
    
    def finalize_xml(self):
        """
        Finalize XML file by adding the closing tag and closing the file.
        Should be called when tracking is complete or application is exiting.
        """
        if self._log_fp is None:
            return
            
        try:
            # Add any final statistics if needed
            if hasattr(self, 'stats') and isinstance(getattr(self, 'stats'), dict):
                stats_elem = "  <Statistics"
                stats = getattr(self, 'stats')
                for key, value in stats.items():
                    stats_elem += f" {key}=\"{str(value)}\""
                stats_elem += "/>\n"
                self._log_fp.write(stats_elem)
            
            # Add closing tag
            self._log_fp.write("</TrackingData>\n")
            self._log_fp.flush()
            self._log_fp.close()
            self._log_fp = None
            
            logging.info("Finalized XML tracking data file")
            
        except Exception as e:
            logging.error(f"Error finalizing XML file: {e}")
    
    def _repair_xml(self, file_path: str) -> bool:
        """
        Repair XML file if it was not properly closed.
        
        Args:
            file_path: Path to the XML file to repair
            
        Returns:
            True if repaired or no repair needed, False on error
        """
        if not os.path.exists(file_path):
            return False
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check if file has proper XML header
            if not content.strip().startswith('<?xml'):
                logging.warning(f"XML file {file_path} does not have XML header. Adding header.")
                content = '<?xml version="1.0" encoding="utf-8"?>\n' + content
                needs_repair = True
            else:
                needs_repair = False
                
            # Check if file has closing tag
            if not content.strip().endswith('</TrackingData>'):
                logging.warning(f"XML file {file_path} not properly closed. Adding closing tag.")
                content = content.rstrip() + "\n</TrackingData>\n"
                needs_repair = True
                
            # Write repaired file if needed
            if needs_repair:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logging.info(f"Repaired XML file: {file_path}")
                
            return True
            
        except Exception as e:
            logging.error(f"Error repairing XML file: {e}")
            return False 