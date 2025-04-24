import logging
import cv2
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFileDialog, QMessageBox, QGroupBox, QRadioButton, QProgressBar
)
from PyQt5.QtGui import QImage, QPixmap, QMouseEvent
from PyQt5.QtCore import Qt, pyqtSignal, QPoint

from src.controllers.calibration import CourtCalibrationController


class ClickableImageLabel(QLabel):
    """
    Custom QLabel that emits a signal when clicked with the click position.
    """
    clicked = pyqtSignal(QPoint)
    
    def __init__(self, text="", parent=None):
        """
        Initialize the clickable image label.
        
        Args:
            text (str, optional): Label text
            parent (QWidget, optional): Parent widget
        """
        super(ClickableImageLabel, self).__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("border: 1px solid #cccccc; background-color: #f5f5f5;")
        
    def mousePressEvent(self, event: QMouseEvent):
        """
        Handle mouse press events.
        
        Args:
            event (QMouseEvent): Mouse event
        """
        if event.button() == Qt.LeftButton and self.pixmap() is not None:
            # Calculate position relative to the image, accounting for scaling
            img_rect = self.pixmap().rect()
            label_rect = self.rect()
            
            # Calculate scale factors
            x_scale = img_rect.width() / label_rect.width()
            y_scale = img_rect.height() / label_rect.height()
            
            # Calculate position within image
            img_x = int(event.x() * x_scale)
            img_y = int(event.y() * y_scale)
            
            # Emit signal with click position
            self.clicked.emit(QPoint(img_x, img_y))
        
        # Call the parent class implementation
        super(ClickableImageLabel, self).mousePressEvent(event)

class CourtCalibrationView(QWidget):
    """
    View for the court calibration process.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the court calibration view.
        
        Args:
            parent (QWidget, optional): Parent widget
        """
        super(CourtCalibrationView, self).__init__(parent)
        
        # Initialize controller
        self.controller = CourtCalibrationController()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize UI variables
        self.left_points = []
        self.right_points = []
        self.active_side = "left"  # Default active side
        
        # Setup UI components
        self._setup_ui()
        
        # Connect signals
        self._connect_signals()
        
        # Update UI based on initial state
        self._update_ui()
        
    def _setup_ui(self):
        """Setup the UI components of the view."""
        # Main layout
        main_layout = QVBoxLayout()
        
        # Images layout (horizontal)
        images_layout = QHBoxLayout()
        
        # Left image container
        left_group = QGroupBox("Left Camera")
        left_layout = QVBoxLayout()
        self.left_image_label = ClickableImageLabel("Load Left Image")
        left_layout.addWidget(self.left_image_label)
        left_group.setLayout(left_layout)
        
        # Right image container
        right_group = QGroupBox("Right Camera")
        right_layout = QVBoxLayout()
        self.right_image_label = ClickableImageLabel("Load Right Image")
        right_layout.addWidget(self.right_image_label)
        right_group.setLayout(right_layout)
        
        # Add image groups to layout
        images_layout.addWidget(left_group)
        images_layout.addWidget(right_group)
        
        # Controls layout
        controls_layout = QHBoxLayout()
        
        # Side selection controls
        side_selection_group = QGroupBox("Active Side Selection")
        side_selection_layout = QHBoxLayout()
        
        self.left_radio = QRadioButton("Left")
        self.left_radio.setChecked(True)
        self.right_radio = QRadioButton("Right")
        
        side_selection_layout.addWidget(self.left_radio)
        side_selection_layout.addWidget(self.right_radio)
        side_selection_group.setLayout(side_selection_layout)
        
        # Image controls
        image_controls_group = QGroupBox("Image Controls")
        image_controls_layout = QHBoxLayout()
        
        self.load_left_btn = QPushButton("Load Left Image")
        self.load_right_btn = QPushButton("Load Right Image")
        self.clear_points_btn = QPushButton("Clear Points")
        
        image_controls_layout.addWidget(self.load_left_btn)
        image_controls_layout.addWidget(self.load_right_btn)
        image_controls_layout.addWidget(self.clear_points_btn)
        image_controls_group.setLayout(image_controls_layout)
        
        # Calibration controls
        calibration_controls_group = QGroupBox("Calibration Controls")
        calibration_controls_layout = QHBoxLayout()
        
        self.calibrate_btn = QPushButton("Run Calibration")
        self.tune_calibration_btn = QPushButton("Tune Calibration")
        self.save_config_btn = QPushButton("Save Configuration")
        
        calibration_controls_layout.addWidget(self.calibrate_btn)
        calibration_controls_layout.addWidget(self.tune_calibration_btn)
        calibration_controls_layout.addWidget(self.save_config_btn)
        calibration_controls_group.setLayout(calibration_controls_layout)
        
        # Status display
        self.status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)
        self.status_group.setLayout(status_layout)
        
        # Points info
        self.points_info_label = QLabel("Left: 0/14 points, Right: 0/14 points")
        
        # Add controls to layout
        controls_layout.addWidget(side_selection_group)
        controls_layout.addWidget(image_controls_group)
        controls_layout.addWidget(calibration_controls_group)
        
        # Add all layouts to main layout
        main_layout.addLayout(images_layout)
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(self.status_group)
        main_layout.addWidget(self.points_info_label)
        
        # Set main layout
        self.setLayout(main_layout)
        self.setWindowTitle("Court Calibration")
        self.resize(1000, 600)
        
    def _connect_signals(self):
        """Connect signals to slots."""
        # Image loading
        self.load_left_btn.clicked.connect(self._on_load_left_image)
        self.load_right_btn.clicked.connect(self._on_load_right_image)
        
        # Image clicking
        self.left_image_label.clicked.connect(lambda pos: self._on_image_clicked(pos, "left"))
        self.right_image_label.clicked.connect(lambda pos: self._on_image_clicked(pos, "right"))
        
        # Side selection
        self.left_radio.toggled.connect(lambda checked: self._set_active_side("left") if checked else None)
        self.right_radio.toggled.connect(lambda checked: self._set_active_side("right") if checked else None)
        
        # Other controls
        self.clear_points_btn.clicked.connect(self._on_clear_points)
        self.calibrate_btn.clicked.connect(self._on_calibrate)
        self.tune_calibration_btn.clicked.connect(self._on_tune_calibration)
        self.save_config_btn.clicked.connect(self._on_save_config)
        
        # Controller signals
        self.controller.calibration_status_changed.connect(self._on_calibration_status_changed)
        self.controller.calibration_progress.connect(self._on_calibration_progress)
        
    def _update_ui(self):
        """Update UI based on current state."""
        # Update points info
        self.points_info_label.setText(f"Left: {len(self.left_points)}/14 points, Right: {len(self.right_points)}/14 points")
        
        # Update calibrate button state
        has_enough_points = len(self.left_points) >= 4 and len(self.right_points) >= 4
        self.calibrate_btn.setEnabled(has_enough_points)
        
        # Update tune button state
        has_calibration = self.controller.has_calibration()
        self.tune_calibration_btn.setEnabled(has_calibration)
        
        # Update save button state
        self.save_config_btn.setEnabled(has_calibration)
    
    def _on_load_left_image(self):
        """Handle loading of left image."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Left Image File", "", "Images (*.png *.jpg *.jpeg)")
        
        if file_path:
            self.logger.info(f"Loading left image from: {file_path}")
            try:
                self.controller.load_left_image(file_path)
                self._update_left_image()
                self.status_label.setText(f"Left image loaded: {file_path}")
            except Exception as e:
                self.logger.error(f"Error loading left image: {str(e)}")
                QMessageBox.critical(self, "Image Load Error", f"An error occurred while loading the image: {str(e)}")
    
    def _on_load_right_image(self):
        """Handle loading of right image."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Right Image File", "", "Images (*.png *.jpg *.jpeg)")
        
        if file_path:
            self.logger.info(f"Loading right image from: {file_path}")
            try:
                self.controller.load_right_image(file_path)
                self._update_right_image()
                self.status_label.setText(f"Right image loaded: {file_path}")
            except Exception as e:
                self.logger.error(f"Error loading right image: {str(e)}")
                QMessageBox.critical(self, "Image Load Error", f"An error occurred while loading the image: {str(e)}")
    
    def _update_left_image(self):
        """Update the left image display with current points."""
        if self.controller.has_left_image():
            image = self.controller.get_left_image().copy()
            
            # Draw points on the image
            for idx, point in enumerate(self.left_points):
                cv2.circle(image, (point.x(), point.y()), 5, (0, 255, 0), -1)
                cv2.putText(image, str(idx + 1), (point.x() + 10, point.y()), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Convert to QImage and display
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            self.left_image_label.setPixmap(QPixmap.fromImage(q_image))
    
    def _update_right_image(self):
        """Update the right image display with current points."""
        if self.controller.has_right_image():
            image = self.controller.get_right_image().copy()
            
            # Draw points on the image
            for idx, point in enumerate(self.right_points):
                cv2.circle(image, (point.x(), point.y()), 5, (0, 255, 0), -1)
                cv2.putText(image, str(idx + 1), (point.x() + 10, point.y()), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Convert to QImage and display
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            self.right_image_label.setPixmap(QPixmap.fromImage(q_image))
    
    def _on_image_clicked(self, pos, side):
        """
        Handle clicking on an image to add calibration points.
        
        Args:
            pos (QPoint): Clicked position
            side (str): Side of the image ("left" or "right")
        """
        # Only add point if active side matches
        if side != self.active_side:
            return
            
        # Check max points (14 per side)
        if side == "left" and len(self.left_points) >= 14:
            self.status_label.setText("Maximum number of points reached (Left: 14)")
            return
        elif side == "right" and len(self.right_points) >= 14:
            self.status_label.setText("Maximum number of points reached (Right: 14)")
            return
        
        # Add point
        if side == "left":
            self.left_points.append(pos)
            self._update_left_image()
            self.logger.info(f"Added left point {len(self.left_points)}: ({pos.x()}, {pos.y()})")
        else:
            self.right_points.append(pos)
            self._update_right_image()
            self.logger.info(f"Added right point {len(self.right_points)}: ({pos.x()}, {pos.y()})")
        
        # Update UI
        self._update_ui()
        self.status_label.setText(f"Point added to {side.capitalize()} image: ({pos.x()}, {pos.y()})")
    
    def _set_active_side(self, side):
        """
        Set the active side for point selection.
        
        Args:
            side (str): Side to set as active ("left" or "right")
        """
        self.active_side = side
        self.logger.info(f"Active side set to: {side}")
        
        # Update radio buttons
        if side == "left":
            self.left_radio.setChecked(True)
        else:
            self.right_radio.setChecked(True)
            
        # Update status
        self.status_label.setText(f"Active side: {side}")
    
    def _on_clear_points(self):
        """Clear all selected calibration points."""
        active_side = self.active_side
        
        if active_side == "left":
            self.left_points.clear()
            self._update_left_image()
            self.logger.info("Cleared left points")
        else:
            self.right_points.clear()
            self._update_right_image()
            self.logger.info("Cleared right points")
        
        # Update UI
        self._update_ui()
        self.status_label.setText(f"{active_side.capitalize()} points cleared")
    
    def _on_calibrate(self):
        """Handle calibration button click."""
        if len(self.left_points) < 4 or len(self.right_points) < 4:
            QMessageBox.warning(self, "Insufficient Points", "At least 4 points per side are required for calibration.")
            return
        
        # Convert QPoint to (x, y) tuples
        left_points = [(p.x(), p.y()) for p in self.left_points]
        right_points = [(p.x(), p.y()) for p in self.right_points]
        
        try:
            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Start calibration
            self.controller.calibrate(left_points, right_points)
            
            # Update UI
            self._update_ui()
            self.status_label.setText("Calibration completed")
        except Exception as e:
            self.logger.error(f"Calibration error: {str(e)}")
            QMessageBox.critical(self, "Calibration Error", f"An error occurred during calibration: {str(e)}")
            self.progress_bar.setVisible(False)
    
    def _on_tune_calibration(self):
        """Handle tune calibration button click."""
        if not self.controller.has_calibration():
            QMessageBox.warning(self, "No Calibration", "You must run calibration first.")
            return
        
        try:
            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Start tuning
            self.controller.tune_calibration()
            
            # Update UI
            self._update_ui()
            self.status_label.setText("Calibration tuning completed")
        except Exception as e:
            self.logger.error(f"Calibration tuning error: {str(e)}")
            QMessageBox.critical(self, "Tuning Error", f"An error occurred during calibration tuning: {str(e)}")
            self.progress_bar.setVisible(False)
    
    def _on_save_config(self):
        """Save calibration configuration to config.json."""
        if not self.controller.has_calibration():
            QMessageBox.warning(self, "No Calibration", "No calibration data to save.")
            return
            
        try:
            # Save calibration data
            self.controller.save_calibration()
            self.status_label.setText("Calibration settings saved")
        except Exception as e:
            self.logger.error(f"Config save error: {str(e)}")
            QMessageBox.critical(self, "Save Error", f"An error occurred while saving settings: {str(e)}")
    
    def _on_calibration_status_changed(self, status):
        """
        Handle calibration status change from controller.
        
        Args:
            status (str): New status
        """
        self.status_label.setText(status)
        
    def _on_calibration_progress(self, value):
        """
        Handle calibration progress update from controller.
        
        Args:
            value (int): Progress value (0-100)
        """
        self.progress_bar.setValue(value)
        if value >= 100:
            self.progress_bar.setVisible(False) 