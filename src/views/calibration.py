import logging
import cv2
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFileDialog, QMessageBox, QGroupBox, QRadioButton, QProgressBar,
    QGraphicsScene, QGraphicsView, QGraphicsItem, QGraphicsEllipseItem, QGraphicsSimpleTextItem
)
from PyQt5.QtGui import QImage, QPixmap, QMouseEvent, QColor, QPen, QBrush
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QPointF, QRectF

from src.controllers.calibration import CourtCalibrationController


class DraggablePointItem(QGraphicsEllipseItem):
    """
    A draggable point for representing calibration points.
    """
    
    def __init__(self, x, y, radius=5, color=Qt.red):
        """
        Initialize a draggable point.
        
        Args:
            x (float): X coordinate
            y (float): Y coordinate
            radius (int): Point radius
            color (QColor): Point color
        """
        super(DraggablePointItem, self).__init__(0, 0, radius*2, radius*2)
        
        # Set position (centered at x,y)
        self.setPos(x - radius, y - radius)
        
        # Set appearance
        self.setPen(QPen(color, 2))
        self.setBrush(QBrush(color, Qt.SolidPattern))
        
        # Make item draggable
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        
        # Store radius for reference
        self.radius = radius
        
    def get_center(self):
        """Get the center point of this item."""
        return QPointF(self.pos().x() + self.radius, self.pos().y() + self.radius)
        
    def itemChange(self, change, value):
        """
        Handle item changes for position tracking.
        
        Args:
            change: The type of change
            value: The new value
            
        Returns:
            The modified value
        """
        if change == QGraphicsItem.ItemPositionChange and self.scene():
            # Ensure the point stays within the image bounds
            if self.scene().sceneRect().contains(value + QPointF(self.radius, self.radius)):
                return value
            else:
                # Constrain to the scene rect
                rect = self.scene().sceneRect()
                new_pos = QPointF(value)
                
                if value.x() < rect.left():
                    new_pos.setX(rect.left())
                elif value.x() + self.radius*2 > rect.right():
                    new_pos.setX(rect.right() - self.radius*2)
                    
                if value.y() < rect.top():
                    new_pos.setY(rect.top())
                elif value.y() + self.radius*2 > rect.bottom():
                    new_pos.setY(rect.bottom() - self.radius*2)
                    
                return new_pos
        
        return super(DraggablePointItem, self).itemChange(change, value)


class ClickableImageView(QGraphicsView):
    """
    A graphics view that captures clicks and hosts draggable points.
    """
    clicked = pyqtSignal(QPoint)
    point_moved = pyqtSignal(int, QPointF)  # Index, new position
    
    def __init__(self, parent=None):
        """
        Initialize a graphics view for holding calibration points.
        
        Args:
            parent (QWidget, optional): Parent widget
        """
        super(ClickableImageView, self).__init__(parent)
        
        # Create scene
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # Set appearance
        self.setMinimumSize(400, 300)
        self.setStyleSheet("border: 1px solid #cccccc; background-color: #f5f5f5;")
        
        # Fit in view
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setRenderHint(1)  # Antialiasing
        
        # Store points
        self.points = []
        
    def set_image(self, pixmap):
        """
        Set the background image.
        
        Args:
            pixmap (QPixmap): Image to display
        """
        self.scene.clear()
        self.points.clear()
        
        # Add pixmap to scene
        self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))
        
        # Fit view to image
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        
    def add_point(self, point, color=Qt.red):
        """
        Add a point to the scene.
        
        Args:
            point (QPoint): Point coordinates
            color (QColor): Point color
            
        Returns:
            DraggablePointItem: The created point item
        """
        # Create point item
        point_item = DraggablePointItem(point.x(), point.y(), color=color)
        
        # Add to scene and store
        self.scene.addItem(point_item)
        self.points.append(point_item)
        
        return point_item
        
    def clear_points(self):
        """Remove all points from the scene."""
        for point in self.points:
            self.scene.removeItem(point)
        self.points.clear()
        
    def get_points(self):
        """
        Get the current point positions.
        
        Returns:
            List[QPoint]: List of point positions
        """
        return [point.get_center().toPoint() for point in self.points]
        
    def mousePressEvent(self, event):
        """
        Handle mouse press events.
        
        Args:
            event (QMouseEvent): Mouse event
        """
        if event.button() == Qt.LeftButton:
            # Convert to scene coordinates
            scene_pos = self.mapToScene(event.pos())
            
            # Check if clicked on an item
            items = self.scene.items(scene_pos)
            if not any(isinstance(item, DraggablePointItem) for item in items):
                # No point clicked, emit signal for adding a new point
                self.clicked.emit(scene_pos.toPoint())
                
        # Process the event
        super(ClickableImageView, self).mousePressEvent(event)
        
    def mouseReleaseEvent(self, event):
        """
        Handle mouse release events.
        
        Args:
            event (QMouseEvent): Mouse event
        """
        # Emit signal for moved points
        for i, point in enumerate(self.points):
            if point.isSelected():
                self.point_moved.emit(i, point.get_center())
                
        super(ClickableImageView, self).mouseReleaseEvent(event)
        
    def resizeEvent(self, event):
        """
        Handle resize events to maintain proper scaling.
        
        Args:
            event: Resize event
        """
        if not self.scene.sceneRect().isEmpty():
            self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        super(ClickableImageView, self).resizeEvent(event)


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
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Court Calibration View")
        
        # Initialize controller
        self.controller = CourtCalibrationController(self)
        
        # Setup the user interface
        self._setup_ui()
        
        # Connect signals
        self._connect_signals()
        
        # Set initial active side
        self.active_side = "left"
        
        # Initialize UI state
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
        self.left_image_view = ClickableImageView()
        left_layout.addWidget(self.left_image_view)
        left_group.setLayout(left_layout)
        
        # Right image container
        right_group = QGroupBox("Right Camera")
        right_layout = QVBoxLayout()
        self.right_image_view = ClickableImageView()
        right_layout.addWidget(self.right_image_view)
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
        
        # Image clicking and point dragging
        self.left_image_view.clicked.connect(lambda pos: self._on_image_clicked(pos, "left"))
        self.right_image_view.clicked.connect(lambda pos: self._on_image_clicked(pos, "right"))
        self.left_image_view.point_moved.connect(lambda idx, pos: self._on_point_moved(idx, pos, "left"))
        self.right_image_view.point_moved.connect(lambda idx, pos: self._on_point_moved(idx, pos, "right"))
        
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
        
        # 중요: calibration_updated 시그널이 발생할 때마다 포인트 재구성
        self.controller.calibration_updated.connect(lambda: self._rebuild_points("left"))
        self.controller.calibration_updated.connect(lambda: self._rebuild_points("right"))
        self.controller.calibration_updated.connect(self._update_ui)
        
    def _update_ui(self):
        """Update UI based on current state."""
        # 모델에서 직접 포인트 정보 가져오기
        left_count = len(self.controller.get_points("left"))
        right_count = len(self.controller.get_points("right"))
        
        # Update points info
        self.points_info_label.setText(f"Left: {left_count}/14 points, Right: {right_count}/14 points")
        
        # Update calibrate button state
        has_enough_points = left_count >= 4 and right_count >= 4
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
        """Update the left image display."""
        img = self.controller.get_left_image()
        if img is not None:
            # Convert OpenCV image to QPixmap
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img)
            
            # Update image and rebuild points
            self.left_image_view.set_image(pixmap)
            self._rebuild_points("left")
    
    def _update_right_image(self):
        """Update the right image display."""
        img = self.controller.get_right_image()
        if img is not None:
            # Convert OpenCV image to QPixmap
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img)
            
            # Update image and rebuild points
            self.right_image_view.set_image(pixmap)
            self._rebuild_points("right")
    
    def _rebuild_points(self, side):
        """
        Rebuild the points display for the specified side.
        포인트 표시를 재구성하고 모든 드래그 가능한 아이템의 플래그를 올바르게 설정합니다.
        
        Args:
            side (str): The side to rebuild ('left' or 'right')
        """
        # Get the view for the specified side
        view = self.left_image_view if side == "left" else self.right_image_view
        
        # Clear existing points
        view.clear_points()
        
        # Get points for the specified side
        points = self.controller.get_points(side)
        
        # Add points with proper flags
        for idx, point in enumerate(points):
            # 새로운 포인트 아이템 생성
            point_item = view.add_point(QPoint(point[0], point[1]), Qt.red)
            
            # 중요: 반드시 플래그 설정 (드래그 가능하도록)
            point_item.setFlag(QGraphicsItem.ItemIsMovable, True)
            point_item.setFlag(QGraphicsItem.ItemIsSelectable, True)
            point_item.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
            
            # 번호 라벨 추가 및 부모-자식 관계 설정
            label = QGraphicsSimpleTextItem(str(idx+1))
            label.setPos(point[0]+10, point[1]+10)
            label.setParentItem(point_item)  # 부모-자식 관계로 연결하여 함께 이동
            
    def _on_image_clicked(self, pos, side):
        """Handle image clicks for adding points."""
        self.logger.debug(f"Image click at {pos.x()}, {pos.y()} on {side} side")
        
        # 이미 존재하는 포인트를 클릭했는지 확인
        view = self.left_image_view if side == "left" else self.right_image_view
        items = view.scene.items(pos)
        if any(isinstance(item, DraggablePointItem) for item in items):
            # 이미 존재하는 포인트를 클릭한 경우 새 포인트 추가하지 않음
            return
        
        # Only add points on the active side
        if side != self.active_side:
            self.logger.debug(f"Ignoring click on inactive {side} side")
            return
            
        # 컨트롤러를 통해 포인트 추가 (모델에 바로 반영됨)
        self.controller.add_point((pos.x(), pos.y()), side)
        
        # 포인트 재구성과 UI 업데이트는 모델 신호에 의해 자동으로 처리됨
    
    def _on_point_moved(self, index, pos, side):
        """
        Handle point movement.
        
        Args:
            index (int): Point index
            pos (QPointF): New position
            side (str): Side ("left" or "right")
        """
        self.logger.debug(f"Point {index} moved to {pos.x()}, {pos.y()} on {side} side")
        
        # 컨트롤러를 통해 모델 업데이트
        self.controller.update_point(index, (int(pos.x()), int(pos.y())), side)
    
    def _set_active_side(self, side):
        """Set the active side for point placement."""
        self.logger.debug(f"Setting active side to {side}")
        self.active_side = side
        
        # Update display for both sides
        self._rebuild_points("left")
        self._rebuild_points("right")
    
    def _on_clear_points(self):
        """Clear all calibration points."""
        self.logger.info("Clearing all calibration points")
        
        # 컨트롤러를 통해 포인트 클리어 (모델에 바로 반영됨)
        self.controller.clear_points()
        
        # 상태 메시지 업데이트
        self.status_label.setText("All points cleared")
        
        # 포인트 재구성과 UI 업데이트는 모델 신호에 의해 자동으로 처리됨
    
    def _on_calibrate(self):
        """Run the court calibration."""
        self.logger.info("Starting court calibration")
        
        # Check if we have enough points
        if len(self.controller.get_points("left")) < 4 or len(self.controller.get_points("right")) < 4:
            QMessageBox.warning(self, "Insufficient Points", 
                             "At least 4 corresponding points are required on both images")
            return
            
        # Disable UI elements
        self._set_ui_enabled(False)
        
        # Show progress bar
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.status_label.setText("Calibrating...")
        
        # Run calibration in controller
        success = self.controller.run_calibration()
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Re-enable UI
        self._set_ui_enabled(True)
        
        # Show result
        if success:
            self.status_label.setText("Calibration completed successfully")
            QMessageBox.information(self, "Calibration Success", 
                                 "Court calibration completed successfully")
        else:
            self.status_label.setText("Calibration failed")
            QMessageBox.critical(self, "Calibration Failed", 
                              "Court calibration failed. Please check the logs for details.")
            
        # Update UI state
        self._update_ui()
    
    def _on_tune_calibration(self):
        """Tune the court calibration with automatic intersection detection."""
        self.logger.info("Starting calibration tuning")
        
        # Disable UI elements
        self._set_ui_enabled(False)
        
        # Show progress bar
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.status_label.setText("Tuning calibration...")
        
        # Run tuning in controller
        success = self.controller.tune_calibration()
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Re-enable UI
        self._set_ui_enabled(True)
        
        # Show result
        if success:
            self.status_label.setText("Calibration tuning completed")
            QMessageBox.information(self, "Tuning Success", 
                                 "Calibration tuning completed successfully")
        else:
            self.status_label.setText("Calibration tuning failed")
            QMessageBox.warning(self, "Tuning Failed", 
                             "Calibration tuning did not find sufficient intersection points.")
            
        # Update UI
        self._update_ui()
    
    def _on_save_config(self):
        """Save the current calibration to a configuration file."""
        self.logger.info("Saving calibration configuration")
        
        # Check if we have a calibration
        if not self.controller.has_calibration():
            QMessageBox.warning(self, "No Calibration", 
                             "No calibration available to save")
            return
            
        # Get save file path
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Calibration Configuration", 
                                                "court_config.json", "JSON Files (*.json)")
        
        if file_path:
            try:
                # Save through controller
                self.controller.save_config(file_path)
                self.status_label.setText(f"Configuration saved to: {file_path}")
                QMessageBox.information(self, "Save Success", 
                                     f"Calibration configuration saved to: {file_path}")
            except Exception as e:
                self.logger.error(f"Error saving configuration: {str(e)}")
                QMessageBox.critical(self, "Save Error", 
                                  f"An error occurred while saving the configuration: {str(e)}")
    
    def _on_calibration_status_changed(self, status):
        """Handle calibration status changes."""
        self.status_label.setText(status)
    
    def _on_calibration_progress(self, value):
        """Handle calibration progress updates."""
        self.progress_bar.setValue(value)
    
    def _set_ui_enabled(self, enabled):
        """Enable or disable UI elements during operations."""
        # Points UI
        self.left_radio.setEnabled(enabled)
        self.right_radio.setEnabled(enabled)
        
        # Image controls
        self.load_left_btn.setEnabled(enabled)
        self.load_right_btn.setEnabled(enabled)
        self.clear_points_btn.setEnabled(enabled)
        
        # Calibration controls
        self.calibrate_btn.setEnabled(enabled and len(self.controller.get_points("left")) >= 4 and len(self.controller.get_points("right")) >= 4)
        self.tune_calibration_btn.setEnabled(enabled and self.controller.has_calibration())
        self.save_config_btn.setEnabled(enabled and self.controller.has_calibration()) 