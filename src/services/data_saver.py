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
from collections import deque
import threading


class DataSaver:
    """
    Service class for saving tracking data to files.
    """
    
    def __init__(self, queue_size=30, batch_size=5):
        """
        Initialize the data saver.
        
        Args:
            queue_size: Maximum number of frames to keep in the queue
            batch_size: Number of frames to process in a batch
        """
        self.xml_root = None
        self.current_folder = None
        
        # 비동기 저장을 위한 큐 및 설정
        self.frame_queue = deque(maxlen=queue_size)
        self.batch_size = batch_size
        self.queue_lock = threading.Lock()
        self.save_thread = None
        self.is_saving = False
        self.exit_flag = False
        
        # XML 저장 최적화를 위한 변수
        self._frames_since_xml_save = 0
        self._xml_save_interval = 30  # 30 프레임마다 XML 파일 저장 (1초에 30 FPS 가정 시 1초마다 저장)
        self._last_xml_save_time = 0
        self._xml_save_time_interval = 5.0  # 최소 5초 간격으로 XML 저장
        self._xml_modified = False  # XML이 변경되었는지 여부
        self._last_frame = -1  # 마지막으로 처리된 프레임 번호 초기화
    
    def enqueue_frame(self, frame_number: int, frame_data: Dict[str, Any], 
                      folder_path: Optional[str] = None, auto_flush: bool = True):
        """
        추가 프레임 데이터를 큐에 추가하고 필요시 저장을 트리거합니다.
        
        Args:
            frame_number: 프레임 번호
            frame_data: 저장할 프레임 데이터
            folder_path: 저장 폴더 경로 (없으면 기본값 사용)
            auto_flush: 큐가 배치 크기에 도달하면 자동으로 저장
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # 큐에 항목 추가
            with self.queue_lock:
                self.frame_queue.append((frame_number, frame_data, folder_path))
                queue_size = len(self.frame_queue)
            
            logging.debug(f"Frame {frame_number} added to queue. Queue size: {queue_size}")
            
            # 큐가 배치 크기에 도달하면 저장 시작
            if auto_flush and queue_size >= self.batch_size:
                self.flush_queue()
                
            return True
        except Exception as e:
            logging.error(f"Error enqueueing frame data: {e}")
            return False
    
    def flush_queue(self):
        """
        큐에 있는 모든 프레임을 비동기적으로 저장합니다.
        
        Returns:
            저장 작업이 시작되었는지 여부
        """
        # 이미 저장 중이거나 큐가 비어있으면 건너뜀
        if self.is_saving or len(self.frame_queue) == 0:
            return False
            
        # 이미 실행 중인 스레드가 있으면 종료를 기다림
        if self.save_thread and self.save_thread.is_alive():
            logging.debug("Waiting for previous save thread to complete...")
            self.save_thread.join(timeout=1.0)
            
        # 새 저장 스레드 시작
        self.is_saving = True
        self.save_thread = threading.Thread(target=self._process_queue)
        self.save_thread.daemon = True
        self.save_thread.start()
        
        logging.debug(f"Started queue processing thread with {len(self.frame_queue)} items")
        return True
    
    def _process_queue(self):
        """
        큐에서 프레임 데이터를 처리하는 백그라운드 메서드
        """
        try:
            frames_to_process = []
            
            # 현재 큐의 모든 항목을 복사하고 큐 비우기
            with self.queue_lock:
                frames_to_process = list(self.frame_queue)
                self.frame_queue.clear()
            
            logging.debug(f"Processing {len(frames_to_process)} frames from queue")
            
            # 모든 프레임 처리
            for frame_number, frame_data, folder_path in frames_to_process:
                if self.exit_flag:
                    break
                    
                # 파일에 저장
                self.save_json_frame(frame_number, frame_data, folder_path)
                
                # XML에 저장 (프레임 이름은 숫자 기반으로 생성)
                frame_name = f"frame_{frame_number:06d}.png"
                self.save_frame_to_xml(frame_number, frame_data, frame_name)
            
            logging.debug(f"Completed processing {len(frames_to_process)} frames")
            
        except Exception as e:
            logging.error(f"Error processing queue: {e}")
        finally:
            self.is_saving = False
    
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
            
            # 평균 좌표 계산을 위한 모든 좌표 저장
            all_points = []
            
            # Process left camera data
            left_elem = ET.SubElement(image_elem, "Left")
            left_data = frame_data.get("left", {})
            
            # HSV mask centroid
            if left_data.get("hsv_center"):
                hsv_elem = ET.SubElement(left_elem, "HSV")
                hsv_x = float(left_data["hsv_center"]["x"])
                hsv_y = float(left_data["hsv_center"]["y"])
                hsv_elem.set("x", str(hsv_x))
                hsv_elem.set("y", str(hsv_y))
                all_points.append((hsv_x, hsv_y))
            
            # Hough circle
            if left_data.get("hough_center"):
                hough_elem = ET.SubElement(left_elem, "Hough")
                hough_x = float(left_data["hough_center"]["x"])
                hough_y = float(left_data["hough_center"]["y"])
                hough_elem.set("x", str(hough_x))
                hough_elem.set("y", str(hough_y))
                if "radius" in left_data["hough_center"]:
                    hough_elem.set("radius", str(float(left_data["hough_center"]["radius"])))
                all_points.append((hough_x, hough_y))
            
            # Kalman prediction
            if left_data.get("kalman_prediction"):
                kalman_elem = ET.SubElement(left_elem, "Kalman")
                kalman_x = float(left_data["kalman_prediction"]["x"])
                kalman_y = float(left_data["kalman_prediction"]["y"])
                kalman_elem.set("x", str(kalman_x))
                kalman_elem.set("y", str(kalman_y))
                if "vx" in left_data["kalman_prediction"]:
                    kalman_elem.set("vx", str(left_data["kalman_prediction"]["vx"]))
                    kalman_elem.set("vy", str(left_data["kalman_prediction"]["vy"]))
                all_points.append((kalman_x, kalman_y))
            
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
                hsv_x = float(right_data["hsv_center"]["x"])
                hsv_y = float(right_data["hsv_center"]["y"])
                hsv_elem.set("x", str(hsv_x))
                hsv_elem.set("y", str(hsv_y))
                all_points.append((hsv_x, hsv_y))
            
            # Hough circle
            if right_data.get("hough_center"):
                hough_elem = ET.SubElement(right_elem, "Hough")
                hough_x = float(right_data["hough_center"]["x"])
                hough_y = float(right_data["hough_center"]["y"])
                hough_elem.set("x", str(hough_x))
                hough_elem.set("y", str(hough_y))
                if "radius" in right_data["hough_center"]:
                    hough_elem.set("radius", str(float(right_data["hough_center"]["radius"])))
                all_points.append((hough_x, hough_y))
            
            # Kalman prediction
            if right_data.get("kalman_prediction"):
                kalman_elem = ET.SubElement(right_elem, "Kalman")
                kalman_x = float(right_data["kalman_prediction"]["x"])
                kalman_y = float(right_data["kalman_prediction"]["y"])
                kalman_elem.set("x", str(kalman_x))
                kalman_elem.set("y", str(kalman_y))
                if "vx" in right_data["kalman_prediction"]:
                    kalman_elem.set("vx", str(right_data["kalman_prediction"]["vx"]))
                    kalman_elem.set("vy", str(right_data["kalman_prediction"]["vy"]))
                all_points.append((kalman_x, kalman_y))
            
            # Fused coordinate
            if right_data.get("fused_center"):
                fused_elem = ET.SubElement(right_elem, "Fused")
                fused_elem.set("x", str(float(right_data["fused_center"]["x"])))
                fused_elem.set("y", str(float(right_data["fused_center"]["y"])))
            
            # World coordinate
            if "world" in frame_data:
                world_elem = ET.SubElement(image_elem, "World")
                world_data = frame_data["world"]
                for key, value in world_data.items():
                    world_elem.set(key, str(value))
                    
            # 평균 좌표 계산 및 추가
            if all_points:
                mean_x = sum(p[0] for p in all_points) / len(all_points)
                mean_y = sum(p[1] for p in all_points) / len(all_points)
                
                # Mean 요소 추가
                mean_elem = ET.SubElement(image_elem, "Mean")
                mean_elem.set("x", f"{mean_x:.1f}")
                mean_elem.set("y", f"{mean_y:.1f}")
                
                logging.debug(f"Added Mean coordinates ({mean_x:.1f}, {mean_y:.1f}) from {len(all_points)} points")
            
            # 마지막 프레임 번호 업데이트
            self._last_frame = max(self._last_frame, frame_number)
            
            logging.debug(f"Added frame {frame_number} to XML tracking data stream")
            return True
            
        except Exception as e:
            logging.error(f"Error saving frame to XML: {e}")
            return False
    
    def save_xml_tracking_data(self, folder_path: Optional[str] = None, force: bool = False) -> Optional[str]:
        """
        Save the XML tracking data to a file.
        
        Args:
            folder_path: Path to the output folder. 
                        Default is 'tracking_data/{current_folder}'.
            force: Whether to force save even if throttling conditions aren't met
            
        Returns:
            Path to the saved file or None if failed
        """
        if self.xml_root is None:
            logging.error("XML tracking not initialized. Call initialize_xml_tracking first.")
            return None
            
        # 강제 저장이 아니고 충분한 시간이 지나지 않은 경우 저장하지 않음
        current_time = time.time()
        if not force and (current_time - self._last_xml_save_time < 1.0):
            logging.debug("Throttling XML save operation (too soon)")
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
            
            # 저장 완료 후 타임스탬프 업데이트
            self._last_xml_save_time = current_time
            
            logging.info(f"Saved XML tracking data to {output_file}")
            return output_file
            
        except Exception as e:
            logging.error(f"Error saving XML tracking data: {e}")
            return None
    
    def cleanup(self, wait=True, timeout=3.0):
        """
        Complete all saving tasks and clean up resources.
        
        Args:
            wait: Whether to wait for ongoing save operations to complete
            timeout: Maximum wait time (seconds)
            
        Returns:
            Task completion status
        """
        try:
            # Process all remaining queue items
            if len(self.frame_queue) > 0:
                logging.info(f"Saving remaining {len(self.frame_queue)} items in queue before cleanup")
                self.flush_queue()
                
            # Set exit flag
            self.exit_flag = True
            
            # Wait for save thread to complete
            if wait and self.save_thread and self.save_thread.is_alive():
                logging.debug(f"Waiting for save thread to complete (timeout: {timeout}s)")
                self.save_thread.join(timeout=timeout)
                
            # Save XML file
            if self.xml_root is not None:
                self.save_xml_tracking_data()
                
            return not (self.save_thread and self.save_thread.is_alive())
            
        except Exception as e:
            logging.error(f"Error during DataSaver cleanup: {e}")
            return False
    
    def finalize_xml(self, processing_stats=None):
        """
        Finalize the XML tracking data and save to file.
        
        Args:
            processing_stats: Optional processing statistics to add
        """
        if self.xml_root is None:
            logging.debug("No XML tracking data to finalize")
            return
            
        try:
            # If processing stats provided, add to XML
            if processing_stats is not None and isinstance(processing_stats, dict):
                stats_elem = ET.SubElement(self.xml_root, "Statistics")
                
                # Add each statistic as an attribute
                for key, value in processing_stats.items():
                    stats_elem.set(key, str(value))
                    
                logging.debug(f"Added processing statistics to XML: {processing_stats}")
            
            # Force save and update timestamp
            self.xml_root.set("finalized", time.strftime("%Y-%m-%d %H:%M:%S"))
            self.xml_root.set("total_frames", str(len(self.xml_root.findall("Image"))))
            
            # 저장이 실패해도 강제로 한 번 더 시도
            self.save_xml_tracking_data(force=True)
            logging.info("XML tracking data finalized and saved")
            
        except Exception as e:
            logging.error(f"Error finalizing XML tracking data: {e}")
            
        # Remove XML root from memory
        self.xml_root = None
        self.current_folder = None 