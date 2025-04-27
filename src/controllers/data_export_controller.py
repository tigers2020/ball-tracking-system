import os
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import QMessageBox

from src.utils.file_io import FileIOFactory, FileDialogHelper, PathUtils, CsvFileIO
from src.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class DataExportController(QObject):
    """
    Controller for data export/import functionality.
    Provides methods for exporting and importing various types of data.
    """
    
    # Signals
    export_successful = Signal(str)  # file_path
    import_successful = Signal(str)  # file_path
    
    def __init__(self, config_manager: Optional[ConfigManager] = None, parent=None):
        """
        Initialize the data export controller.
        
        Args:
            config_manager: Optional config manager instance
            parent: Optional parent QObject
        """
        super().__init__(parent)
        self.config_manager = config_manager
        
        # Default export/import directories
        self.default_export_dir = Path(os.path.expanduser("~")) / "Documents" / "BallTrackingSystem" / "exports"
        self.default_import_dir = Path(os.path.expanduser("~")) / "Documents" / "BallTrackingSystem" / "imports"
        
        # Create default directories
        PathUtils.ensure_dir(self.default_export_dir)
        PathUtils.ensure_dir(self.default_import_dir)
        
        # Load paths from config if available
        if self.config_manager:
            # Create app_settings section if it doesn't exist
            app_settings = self.config_manager.get("app_settings", {})
            
            export_dir = app_settings.get("export_directory")
            if export_dir:
                self.default_export_dir = Path(export_dir)
                
            import_dir = app_settings.get("import_directory")
            if import_dir:
                self.default_import_dir = Path(import_dir)
    
    def _save_directory_to_config(self, key, directory_path):
        """
        Helper method to save directory paths to config manager.
        
        Args:
            key: Key to save under in app_settings section
            directory_path: Path to save
        """
        if not self.config_manager:
            return
            
        # Get current app_settings or create new dict if not exists
        app_settings = self.config_manager.get("app_settings", {})
        
        # Update the setting
        app_settings[key] = str(directory_path)
        
        # Save back to config
        self.config_manager.set("app_settings", app_settings)
        self.config_manager.save_config()

    def export_data(self, data: Any, parent_widget, title: str, file_type: str, 
                    default_filename: str, file_extension: str) -> Optional[str]:
        """
        Export data to a file.
        
        Args:
            data: Data to export
            parent_widget: Parent widget for the file dialog
            title: Dialog title
            file_type: File type description
            default_filename: Default filename
            file_extension: File extension (e.g., ".json")
            
        Returns:
            str: Path to the exported file or None if export was canceled or failed
        """
        try:
            # Ensure directory exists
            PathUtils.ensure_dir(self.default_export_dir)
            
            # Ensure default filename has the correct extension
            if not default_filename.endswith(file_extension):
                default_filename += file_extension
                
            # Get save path
            file_path = FileDialogHelper.get_save_path(
                parent_widget,
                title,
                str(self.default_export_dir / default_filename),
                f"{file_type} (*{file_extension})"
            )
            
            if not file_path:
                logger.info("Export operation canceled by user")
                return None
                
            # Create file I/O instance
            file_io = FileIOFactory.create(file_path)
            if not file_io:
                QMessageBox.warning(
                    parent_widget,
                    "Export Failed",
                    f"Unsupported file format. Please use {file_extension} extension."
                )
                return None
                
            # Save data
            success = file_io.save(data, file_path)
            
            if success:
                # Update default export directory
                self.default_export_dir = Path(os.path.dirname(file_path))
                
                # Save to config if available
                self._save_directory_to_config("export_directory", self.default_export_dir)
                    
                # Emit signal
                self.export_successful.emit(file_path)
                
                # Show success message
                if parent_widget:
                    QMessageBox.information(
                        parent_widget,
                        "Export Successful",
                        f"Data exported to:\n{file_path}"
                    )
                
                return file_path
            else:
                # Show error message
                if parent_widget:
                    QMessageBox.warning(
                        parent_widget,
                        "Export Failed",
                        "Failed to export data. See log for details."
                    )
                
                return None
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            if parent_widget:
                QMessageBox.critical(
                    parent_widget,
                    "Export Error",
                    f"An error occurred while exporting: {str(e)}"
                )
            return None
    
    def import_data(self, parent_widget, title: str, file_type: str, 
                   file_extension: str) -> Optional[Any]:
        """
        Import data from a file.
        
        Args:
            parent_widget: Parent widget for the file dialog
            title: Dialog title
            file_type: File type description
            file_extension: File extension (e.g., ".json")
            
        Returns:
            Any: Imported data or None if import was canceled or failed
        """
        try:
            # Ensure directory exists
            PathUtils.ensure_dir(self.default_import_dir)
            
            # Get open path
            file_path = FileDialogHelper.get_open_path(
                parent_widget,
                title,
                str(self.default_import_dir),
                f"{file_type} (*{file_extension})"
            )
            
            if not file_path:
                logger.info("Import operation canceled by user")
                return None
                
            # Create file I/O instance
            file_io = FileIOFactory.create(file_path)
            if not file_io:
                QMessageBox.warning(
                    parent_widget,
                    "Import Failed",
                    f"Unsupported file format. Please use {file_extension} extension."
                )
                return None
                
            # Load data
            data = file_io.load(file_path)
            
            if data is not None:
                # Update default import directory
                self.default_import_dir = Path(os.path.dirname(file_path))
                
                # Save to config if available
                self._save_directory_to_config("import_directory", self.default_import_dir)
                    
                # Emit signal
                self.import_successful.emit(file_path)
                
                # Show success message
                if parent_widget:
                    QMessageBox.information(
                        parent_widget,
                        "Import Successful",
                        f"Data imported from:\n{file_path}"
                    )
                
                return data
            else:
                # Show error message
                if parent_widget:
                    QMessageBox.warning(
                        parent_widget,
                        "Import Failed",
                        "Failed to import data. See log for details."
                    )
                
                return None
        except Exception as e:
            logger.error(f"Error importing data: {e}")
            if parent_widget:
                QMessageBox.critical(
                    parent_widget,
                    "Import Error",
                    f"An error occurred while importing: {str(e)}"
                )
            return None
    
    def export_json(self, data: Any, parent_widget, title: str = "Export JSON Data", 
                    default_filename: str = "data") -> Optional[str]:
        """
        Export data to a JSON file.
        
        Args:
            data: Data to export
            parent_widget: Parent widget for the file dialog
            title: Dialog title
            default_filename: Default filename
            
        Returns:
            str: Path to the exported file or None if export was canceled or failed
        """
        return self.export_data(
            data, 
            parent_widget, 
            title, 
            "JSON Files", 
            default_filename, 
            ".json"
        )
    
    def import_json(self, parent_widget, title: str = "Import JSON Data") -> Optional[Any]:
        """
        Import data from a JSON file.
        
        Args:
            parent_widget: Parent widget for the file dialog
            title: Dialog title
            
        Returns:
            Any: Imported data or None if import was canceled or failed
        """
        return self.import_data(
            parent_widget, 
            title, 
            "JSON Files", 
            ".json"
        )
    
    def export_xml(self, data: Dict, parent_widget, title: str = "Export XML Data", 
                  default_filename: str = "data") -> Optional[str]:
        """
        Export data to an XML file.
        
        Args:
            data: Data to export
            parent_widget: Parent widget for the file dialog
            title: Dialog title
            default_filename: Default filename
            
        Returns:
            str: Path to the exported file or None if export was canceled or failed
        """
        return self.export_data(
            data, 
            parent_widget, 
            title, 
            "XML Files", 
            default_filename, 
            ".xml"
        )
    
    def import_xml(self, parent_widget, title: str = "Import XML Data") -> Optional[Dict]:
        """
        Import data from an XML file.
        
        Args:
            parent_widget: Parent widget for the file dialog
            title: Dialog title
            
        Returns:
            Dict: Imported data or None if import was canceled or failed
        """
        return self.import_data(
            parent_widget, 
            title, 
            "XML Files", 
            ".xml"
        )
    
    def export_csv(self, data: List[List], parent_widget, title: str = "Export CSV Data", 
                  default_filename: str = "data", headers: Optional[List[str]] = None) -> Optional[str]:
        """
        Export data to a CSV file.
        
        Args:
            data: Data to export
            parent_widget: Parent widget for the file dialog
            title: Dialog title
            default_filename: Default filename
            headers: Optional list of column headers
            
        Returns:
            str: Path to the exported file or None if export was canceled or failed
        """
        try:
            # Ensure directory exists
            PathUtils.ensure_dir(self.default_export_dir)
            
            # Ensure default filename has the correct extension
            if not default_filename.endswith(".csv"):
                default_filename += ".csv"
                
            # Get save path
            file_path = FileDialogHelper.get_save_path(
                parent_widget,
                title,
                str(self.default_export_dir / default_filename),
                "CSV Files (*.csv)"
            )
            
            if not file_path:
                logger.info("Export operation canceled by user")
                return None
                
            # Create file I/O instance
            file_io = FileIOFactory.create(file_path)
            if not file_io:
                if parent_widget:
                    QMessageBox.warning(
                        parent_widget,
                        "Export Failed",
                        "Unsupported file format. Please use .csv extension."
                    )
                return None
                
            # Save data with custom save method for CSV
            if isinstance(file_io, CsvFileIO):
                success = file_io.save(data, file_path, headers)
            else:
                # Fallback for unexpected instance type
                success = file_io.save(data, file_path)
            
            if success:
                # Update default export directory
                self.default_export_dir = Path(os.path.dirname(file_path))
                
                # Save to config if available
                self._save_directory_to_config("export_directory", self.default_export_dir)
                    
                # Emit signal
                self.export_successful.emit(file_path)
                
                # Show success message
                if parent_widget:
                    QMessageBox.information(
                        parent_widget,
                        "Export Successful",
                        f"Data exported to:\n{file_path}"
                    )
                
                return file_path
            else:
                # Show error message
                if parent_widget:
                    QMessageBox.warning(
                        parent_widget,
                        "Export Failed",
                        "Failed to export data. See log for details."
                    )
                
                return None
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            if parent_widget:
                QMessageBox.critical(
                    parent_widget,
                    "Export Error",
                    f"An error occurred while exporting: {str(e)}"
                )
            return None
    
    def import_csv(self, parent_widget, title: str = "Import CSV Data", 
                  has_headers: bool = False) -> Optional[Any]:
        """
        Import data from a CSV file.
        
        Args:
            parent_widget: Parent widget for the file dialog
            title: Dialog title
            has_headers: Whether the CSV file has headers
            
        Returns:
            Any: Imported data or None if import was canceled or failed
        """
        try:
            # Ensure directory exists
            PathUtils.ensure_dir(self.default_import_dir)
            
            # Get open path
            file_path = FileDialogHelper.get_open_path(
                parent_widget,
                title,
                str(self.default_import_dir),
                "CSV Files (*.csv)"
            )
            
            if not file_path:
                logger.info("Import operation canceled by user")
                return None
                
            # Create file I/O instance
            file_io = FileIOFactory.create(file_path)
            if not file_io:
                if parent_widget:
                    QMessageBox.warning(
                        parent_widget,
                        "Import Failed",
                        "Unsupported file format. Please use .csv extension."
                    )
                return None
                
            # Load data with custom parameter for CSV
            if isinstance(file_io, CsvFileIO):
                data = file_io.load(file_path, has_headers)
            else:
                # Fallback for unexpected instance type
                data = file_io.load(file_path)
            
            if data is not None:
                # Update default import directory
                self.default_import_dir = Path(os.path.dirname(file_path))
                
                # Save to config if available
                self._save_directory_to_config("import_directory", self.default_import_dir)
                    
                # Emit signal
                self.import_successful.emit(file_path)
                
                # Show success message
                if parent_widget:
                    QMessageBox.information(
                        parent_widget,
                        "Import Successful",
                        f"Data imported from:\n{file_path}"
                    )
                
                return data
            else:
                # Show error message
                if parent_widget:
                    QMessageBox.warning(
                        parent_widget,
                        "Import Failed",
                        "Failed to import data. See log for details."
                    )
                
                return None
        except Exception as e:
            logger.error(f"Error importing data: {e}")
            if parent_widget:
                QMessageBox.critical(
                    parent_widget,
                    "Import Error",
                    f"An error occurred while importing: {str(e)}"
                )
            return None
            
    def export_tracking_data(self, tracking_controller, parent_widget=None, 
                           folder_path=None, default_filename=None) -> Optional[str]:
        """
        Export tracking data to a JSON file.
        
        Args:
            tracking_controller: Ball tracking controller instance
            parent_widget: Parent widget for the file dialog
            folder_path: Optional custom folder path
            default_filename: Optional default filename
            
        Returns:
            str: Path to the exported file or None if export was canceled or failed
        """
        try:
            if not tracking_controller:
                logger.error("No tracking controller provided")
                return None
                
            # Prepare tracking data
            tracking_data = {
                "coordinate_history": tracking_controller.get_coordinate_history(),
                "detection_settings": tracking_controller.get_detection_settings_summary(),
                "detection_rate": tracking_controller.get_detection_rate(),
                "export_timestamp": {
                    "unix": time.time(),
                    "formatted": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            
            # Generate default filename if not provided
            if not default_filename:
                detection_rate = tracking_controller.get_detection_rate()
                detection_count = tracking_controller.detection_stats["detection_count"]
                total_frames = tracking_controller.detection_stats["total_frames"]
                default_filename = f"tracking_data_rate{int(detection_rate*100)}_frames{total_frames}_detections{detection_count}"
            
            # Use provided folder path or default
            export_dir = folder_path if folder_path else self.default_export_dir
            PathUtils.ensure_dir(export_dir)
            
            # Export data
            if parent_widget:
                # Show dialog if parent widget is provided
                return self.export_json(
                    tracking_data,
                    parent_widget,
                    "Export Tracking Data",
                    default_filename
                )
            else:
                # Silent export with fixed path if no parent widget
                file_path = os.path.join(export_dir, default_filename + ".json")
                file_io = FileIOFactory.create(file_path)
                if file_io and file_io.save(tracking_data, file_path):
                    logger.info(f"Tracking data exported to: {file_path}")
                    self.export_successful.emit(file_path)
                    return file_path
                else:
                    logger.error("Failed to export tracking data")
                    return None
        except Exception as e:
            logger.error(f"Error exporting tracking data: {e}")
            if parent_widget:
                QMessageBox.critical(
                    parent_widget,
                    "Export Error",
                    f"An error occurred while exporting tracking data: {str(e)}"
                )
            return None 