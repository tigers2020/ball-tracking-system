import os
import json
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import csv
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from abc import ABC, abstractmethod

from PySide6.QtWidgets import QFileDialog

logger = logging.getLogger(__name__)

class IFileIO(ABC):
    """Interface for file I/O operations."""
    
    @abstractmethod
    def save(self, data: Any, file_path: Union[str, Path]) -> bool:
        """
        Save data to a file.
        
        Args:
            data: Data to save
            file_path: Path to save the file
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def load(self, file_path: Union[str, Path]) -> Any:
        """
        Load data from a file.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            Any: Loaded data or None if load failed
        """
        pass


class JsonFileIO(IFileIO):
    """Implementation of IFileIO for JSON files."""
    
    def save(self, data: Any, file_path: Union[str, Path]) -> bool:
        """
        Save data to a JSON file.
        
        Args:
            data: Data to save (must be JSON serializable)
            file_path: Path to save the file
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Convert Path object to string if necessary
            if isinstance(file_path, Path):
                file_path = str(file_path)
                
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write data to file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
                
            logger.info(f"Data saved to JSON file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving data to JSON file: {e}")
            return False
    
    def load(self, file_path: Union[str, Path]) -> Any:
        """
        Load data from a JSON file.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            Any: Loaded data or None if load failed
        """
        try:
            # Convert Path object to string if necessary
            if isinstance(file_path, Path):
                file_path = str(file_path)
                
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None
                
            # Read data from file
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            logger.info(f"Data loaded from JSON file: {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading data from JSON file: {e}")
            return None


class XmlFileIO(IFileIO):
    """Implementation of IFileIO for XML files."""
    
    def save(self, data: Dict, file_path: Union[str, Path]) -> bool:
        """
        Save data to an XML file.
        
        Args:
            data: Dictionary to convert to XML and save
            file_path: Path to save the file
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Convert Path object to string if necessary
            if isinstance(file_path, Path):
                file_path = str(file_path)
                
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Create root element
            root = ET.Element(data.get('root_tag', 'data'))
            
            # Helper function to convert dict to XML
            def dict_to_xml(parent, data_dict):
                for key, value in data_dict.items():
                    # Skip root_tag key
                    if key == 'root_tag':
                        continue
                        
                    if isinstance(value, dict):
                        # Create subelement for dictionary
                        subelem = ET.SubElement(parent, key)
                        dict_to_xml(subelem, value)
                    elif isinstance(value, list):
                        # Create subelement for each item in list
                        for item in value:
                            item_elem = ET.SubElement(parent, key)
                            if isinstance(item, dict):
                                dict_to_xml(item_elem, item)
                            else:
                                item_elem.text = str(item)
                    else:
                        # Create subelement for simple value
                        elem = ET.SubElement(parent, key)
                        elem.text = str(value)
            
            # Convert data to XML
            dict_to_xml(root, data)
            
            # Create XML tree
            tree = ET.ElementTree(root)
            
            # Write XML to file with pretty formatting
            xml_str = ET.tostring(root, encoding='utf-8')
            dom = minidom.parseString(xml_str)
            pretty_xml = dom.toprettyxml(indent="  ")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(pretty_xml)
                
            logger.info(f"Data saved to XML file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving data to XML file: {e}")
            return False
    
    def load(self, file_path: Union[str, Path]) -> Optional[Dict]:
        """
        Load data from an XML file.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            Dict: Dictionary containing the parsed XML data or None if load failed
        """
        try:
            # Convert Path object to string if necessary
            if isinstance(file_path, Path):
                file_path = str(file_path)
                
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None
                
            # Parse XML file
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Helper function to convert XML to dict
            def xml_to_dict(element):
                result = {}
                
                # Process child elements
                for child in element:
                    child_dict = xml_to_dict(child)
                    
                    if child.tag in result:
                        # If tag already exists, convert to list or append to existing list
                        if isinstance(result[child.tag], list):
                            result[child.tag].append(child_dict if child_dict else child.text or '')
                        else:
                            result[child.tag] = [result[child.tag], child_dict if child_dict else child.text or '']
                    else:
                        # Add new tag
                        result[child.tag] = child_dict if child_dict else child.text or ''
                        
                return result if result else None
            
            # Convert XML to dict
            data = {root.tag: xml_to_dict(root)}
            
            logger.info(f"Data loaded from XML file: {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading data from XML file: {e}")
            return None


class CsvFileIO(IFileIO):
    """Implementation of IFileIO for CSV files."""
    
    def save(self, data: List[List], file_path: Union[str, Path], headers: Optional[List[str]] = None) -> bool:
        """
        Save data to a CSV file.
        
        Args:
            data: List of rows to save
            file_path: Path to save the file
            headers: Optional list of column headers
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Convert Path object to string if necessary
            if isinstance(file_path, Path):
                file_path = str(file_path)
                
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write data to file
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write headers if provided
                if headers:
                    writer.writerow(headers)
                    
                # Write data rows
                writer.writerows(data)
                
            logger.info(f"Data saved to CSV file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving data to CSV file: {e}")
            return False
    
    def load(self, file_path: Union[str, Path], has_headers: bool = False) -> Optional[Union[List[List], List[Dict]]]:
        """
        Load data from a CSV file.
        
        Args:
            file_path: Path to the file to load
            has_headers: Whether the CSV file has headers
            
        Returns:
            Union[List[List], List[Dict]]: List of rows or list of dictionaries if has_headers is True
            None if load failed
        """
        try:
            # Convert Path object to string if necessary
            if isinstance(file_path, Path):
                file_path = str(file_path)
                
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None
                
            # Read data from file
            with open(file_path, 'r', newline='') as f:
                reader = csv.reader(f)
                
                if has_headers:
                    # Read headers
                    headers = next(reader)
                    
                    # Read data rows
                    rows = []
                    for row in reader:
                        # Convert row to dictionary using headers
                        row_dict = {headers[i]: row[i] for i in range(min(len(headers), len(row)))}
                        rows.append(row_dict)
                        
                    return rows
                else:
                    # Read all rows
                    return list(reader)
        except Exception as e:
            logger.error(f"Error loading data from CSV file: {e}")
            return None


class FileIOFactory:
    """Factory class for creating file I/O instances based on file extension."""
    
    @staticmethod
    def create(file_path: Union[str, Path]) -> Optional[IFileIO]:
        """
        Create a file I/O instance based on file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            IFileIO: File I/O instance or None if extension is not supported
        """
        # Get file extension
        if isinstance(file_path, Path):
            extension = file_path.suffix.lower()
        else:
            extension = os.path.splitext(file_path)[1].lower()
            
        # Create instance based on extension
        if extension == '.json':
            return JsonFileIO()
        elif extension == '.xml':
            return XmlFileIO()
        elif extension == '.csv':
            return CsvFileIO()
        else:
            logger.error(f"Unsupported file extension: {extension}")
            return None


class FileDialogHelper:
    """Helper class for Qt file dialogs."""
    
    @staticmethod
    def get_save_path(parent, title: str, directory: str, filter_str: str) -> Optional[str]:
        """
        Show a save file dialog and return the selected path.
        
        Args:
            parent: Parent widget
            title: Dialog title
            directory: Initial directory
            filter_str: File type filter (e.g., "JSON Files (*.json)")
            
        Returns:
            str: Selected file path or None if canceled
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(directory), exist_ok=True)
            
            # Show dialog
            file_path, _ = QFileDialog.getSaveFileName(
                parent,
                title,
                directory,
                filter_str
            )
            
            return file_path if file_path else None
        except Exception as e:
            logger.error(f"Error showing save file dialog: {e}")
            return None
    
    @staticmethod
    def get_open_path(parent, title: str, directory: str, filter_str: str) -> Optional[str]:
        """
        Show an open file dialog and return the selected path.
        
        Args:
            parent: Parent widget
            title: Dialog title
            directory: Initial directory
            filter_str: File type filter (e.g., "JSON Files (*.json)")
            
        Returns:
            str: Selected file path or None if canceled
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(directory), exist_ok=True)
            
            # Show dialog
            file_path, _ = QFileDialog.getOpenFileName(
                parent,
                title,
                directory,
                filter_str
            )
            
            return file_path if file_path else None
        except Exception as e:
            logger.error(f"Error showing open file dialog: {e}")
            return None


class PathUtils:
    """Utility class for path operations."""
    
    @staticmethod
    def ensure_dir(directory: Union[str, Path]) -> bool:
        """
        Ensure directory exists, creating it if necessary.
        
        Args:
            directory: Directory path
            
        Returns:
            bool: True if directory exists or was created, False otherwise
        """
        try:
            if isinstance(directory, str):
                directory = Path(directory)
                
            directory.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error creating directory: {e}")
            return False
    
    @staticmethod
    def get_default_save_dir(base_dir: Union[str, Path], subdir: str = "") -> Path:
        """
        Get default save directory.
        
        Args:
            base_dir: Base directory
            subdir: Subdirectory name
            
        Returns:
            Path: Default save directory
        """
        if isinstance(base_dir, str):
            base_dir = Path(base_dir)
            
        save_dir = base_dir
        if subdir:
            save_dir = save_dir / subdir
            
        # Ensure directory exists
        save_dir.mkdir(parents=True, exist_ok=True)
        
        return save_dir 

# Re-export classes for easier imports
__all__ = ['IFileIO', 'JsonFileIO', 'XmlFileIO', 'CsvFileIO', 'FileIOFactory', 'FileDialogHelper', 'PathUtils'] 