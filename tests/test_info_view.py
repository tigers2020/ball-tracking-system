#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
InfoView 단독 테스트 스크립트.
이 스크립트는 InfoView의 시그널-슬롯 연결과 UI 갱신이 제대로 되는지 검증합니다.
"""

import sys
import os
import logging
from pathlib import Path
import time
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from PySide6.QtCore import Qt, Signal, QTimer, QObject

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# InfoView 임포트
from src.views.info_view import InfoView

# Setup logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("InfoViewTest")

class MockController(QObject):
    """
    BallTrackingController의 테스트용 모의 객체입니다.
    InfoView에 신호를 보내는 모의 처리를 합니다.
    """
    # Signal 정의
    detection_updated = Signal(int, float, object, object, object)
    roi_updated = Signal(object, object)
    prediction_updated = Signal(str, float, float, float, float)
    mask_updated = Signal(np.ndarray, np.ndarray, dict)  # left_mask, right_mask, hsv_settings
    
    def __init__(self):
        super().__init__()
        self.frame_idx = 0
        self.current_position_3d = None
    
    def emit_test_signal(self):
        """테스트용 detection_updated 시그널을 발생시킵니다."""
        self.frame_idx += 1
        detection_rate = 0.75
        left_coords = (100, 200, 10)  # x, y, r
        right_coords = (150, 220, 12)  # x, y, r
        position_3d = (1.234, 2.345, 3.456)  # x, y, z (meters)
        
        # 3D 위치 업데이트
        self.current_position_3d = position_3d
        
        # 신호 발생
        logger.info(f"Emitting detection_updated signal with frame_idx={self.frame_idx}, "
                     f"detection_rate={detection_rate}, "
                     f"left_coords={left_coords}, "
                     f"right_coords={right_coords}, "
                     f"position_3d={position_3d}")
        
        self.detection_updated.emit(
            self.frame_idx, 
            detection_rate,
            left_coords,
            right_coords,
            position_3d
        )
    
    def emit_roi_signal(self):
        """테스트용 roi_updated 시그널을 발생시킵니다."""
        left_roi = {"x": 50, "y": 50, "width": 200, "height": 200}
        right_roi = {"x": 60, "y": 60, "width": 220, "height": 220}
        
        logger.info(f"Emitting roi_updated signal with left_roi={left_roi}, right_roi={right_roi}")
        self.roi_updated.emit(left_roi, right_roi)
    
    def emit_mask_signal(self):
        """테스트용 mask_updated 시그널을 발생시킵니다."""
        # Create simple binary masks for testing
        left_mask = np.zeros((480, 640), dtype=np.uint8)
        left_mask[100:300, 100:300] = 255  # White rectangle in the middle
        
        right_mask = np.zeros((480, 640), dtype=np.uint8)
        right_mask[120:320, 150:350] = 255  # White rectangle in the middle
        
        hsv_settings = {
            "hue_min": 20, "hue_max": 40,
            "sat_min": 100, "sat_max": 255,
            "val_min": 100, "val_max": 255
        }
        
        logger.info(f"Emitting mask_updated signal with hsv_settings={hsv_settings}")
        self.mask_updated.emit(left_mask, right_mask, hsv_settings)
    
    def emit_prediction_signal(self):
        """테스트용 prediction_updated 시그널을 발생시킵니다."""
        # Left camera prediction
        self.prediction_updated.emit("left", 120.5, 220.5, 10.0, 15.0)
        # Right camera prediction
        self.prediction_updated.emit("right", 170.5, 240.5, 12.0, 17.0)
        
        logger.info("Emitted prediction_updated signals for both cameras")
    
    def get_coordinate_history(self, camera, max_points=20):
        """KalmanPathVisualizer용 get_coordinate_history 메서드"""
        if camera == 'left':
            return [(90, 190, 10), (95, 195, 10), (100, 200, 10)]
        else:
            return [(140, 210, 12), (145, 215, 12), (150, 220, 12)]
    
    def get_predictions(self):
        """KalmanPathVisualizer용 get_predictions 메서드"""
        return {
            'left': [120.5, 220.5, 10.0, 15.0],
            'right': [170.5, 240.5, 12.0, 17.0]
        }


class MockAnalyzer(QObject):
    """
    GameAnalyzer의 테스트용 모의 객체입니다.
    InfoView에 게임 상태 신호를 보내는 모의 처리를 합니다.
    """
    # Signal 정의
    tracking_updated = Signal(int, float, object, object, bool)
    in_out_detected = Signal(bool, object)
    court_position_updated = Signal(float, float, float)
    
    def __init__(self):
        super().__init__()
        self.frame_idx = 0
    
    def emit_tracking_signal(self):
        """테스트용 tracking_updated 시그널을 발생시킵니다."""
        self.frame_idx += 1
        timestamp = time.time()
        position = np.array([1.5, 2.5, 0.8])  # x, y, z (meters)
        velocity = np.array([0.1, 0.2, 0.3])  # vx, vy, vz (meters/sec)
        is_valid = True
        
        logger.info(f"Emitting tracking_updated signal with frame_idx={self.frame_idx}, "
                     f"position={position}, velocity={velocity}, is_valid={is_valid}")
        
        self.tracking_updated.emit(
            self.frame_idx,
            timestamp,
            position,
            velocity,
            is_valid
        )
    
    def emit_in_out_signal(self, is_in=True):
        """테스트용 in_out_detected 시그널을 발생시킵니다."""
        position = np.array([1.5, 2.5, 0.0])  # x, y, z (meters)
        
        logger.info(f"Emitting in_out_detected signal with is_in={is_in}, position={position}")
        self.in_out_detected.emit(is_in, position)
    
    def emit_court_position_signal(self):
        """테스트용 court_position_updated 시그널을 발생시킵니다."""
        x, y, z = 2.5, 3.5, 0.2  # 코트 좌표계에서의 위치 (meters)
        
        logger.info(f"Emitting court_position_updated signal with x={x}, y={y}, z={z}")
        self.court_position_updated.emit(x, y, z)


class TestWindow(QMainWindow):
    """
    InfoView 테스트용 윈도우 클래스입니다.
    버튼 클릭으로 각종 신호를 발생시켜 InfoView의 동작을 테스트합니다.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("InfoView 테스트")
        self.resize(1000, 600)
        
        # InfoView 생성
        self.info_view = InfoView()
        
        # 모의 컨트롤러 생성
        self.mock_controller = MockController()
        self.mock_analyzer = MockAnalyzer()
        
        # 컨트롤러 연결
        self.info_view.connect_tracking_controller(self.mock_controller)
        self.info_view.connect_game_analyzer(self.mock_analyzer)
        
        # 중앙 위젯과 레이아웃 설정
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        
        # 테스트 버튼 추가
        self.detection_button = QPushButton("감지 시그널 테스트")
        self.roi_button = QPushButton("ROI 시그널 테스트")
        self.mask_button = QPushButton("마스크 시그널 테스트")
        self.prediction_button = QPushButton("예측 시그널 테스트")
        self.tracking_button = QPushButton("추적 시그널 테스트")
        self.in_out_button = QPushButton("인/아웃 시그널 테스트 (IN)")
        self.out_button = QPushButton("인/아웃 시그널 테스트 (OUT)")
        self.court_button = QPushButton("코트 위치 시그널 테스트")
        self.direct_ui_button = QPushButton("직접 UI 업데이트 테스트")
        
        # 버튼 연결
        self.detection_button.clicked.connect(self.mock_controller.emit_test_signal)
        self.roi_button.clicked.connect(self.mock_controller.emit_roi_signal)
        self.mask_button.clicked.connect(self.mock_controller.emit_mask_signal)
        self.prediction_button.clicked.connect(self.mock_controller.emit_prediction_signal)
        self.tracking_button.clicked.connect(self.mock_analyzer.emit_tracking_signal)
        self.in_out_button.clicked.connect(lambda: self.mock_analyzer.emit_in_out_signal(True))
        self.out_button.clicked.connect(lambda: self.mock_analyzer.emit_in_out_signal(False))
        self.court_button.clicked.connect(self.mock_analyzer.emit_court_position_signal)
        self.direct_ui_button.clicked.connect(self.test_direct_ui_update)
        
        # 자동 테스트 타이머
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.auto_test)
        self.auto_test_button = QPushButton("자동 테스트 시작")
        self.auto_test_button.clicked.connect(self.toggle_auto_test)
        self.auto_testing = False
        
        # UI 배치
        layout.addWidget(self.info_view)
        layout.addWidget(self.detection_button)
        layout.addWidget(self.roi_button)
        layout.addWidget(self.mask_button)
        layout.addWidget(self.prediction_button)
        layout.addWidget(self.tracking_button)
        layout.addWidget(self.in_out_button)
        layout.addWidget(self.out_button)
        layout.addWidget(self.court_button)
        layout.addWidget(self.direct_ui_button)
        layout.addWidget(self.auto_test_button)
        
        self.setCentralWidget(central_widget)
        
        # 초기 로그 메시지
        logger.info("TestWindow initialized")
        logger.info(f"InfoView instance ID: {id(self.info_view)}")
        logger.info(f"MockController instance ID: {id(self.mock_controller)}")
        logger.info(f"MockAnalyzer instance ID: {id(self.mock_analyzer)}")
        
    def test_direct_ui_update(self):
        """InfoView의 set_... 메서드를 직접 호출하여 UI 업데이트를 테스트합니다."""
        logger.info("Directly updating InfoView UI")
        
        # 직접 메서드 호출
        self.info_view.set_detection_rate(0.85)
        self.info_view.set_left_pixel_coords(120, 220, 15)
        self.info_view.set_right_pixel_coords(170, 240, 17)
        self.info_view.set_position_coords(3.333, 4.444, 5.555)
        
        # ROI 및 Kalman 상태 업데이트
        self.info_view.set_left_roi(40, 40, 180, 180)
        self.info_view.set_right_roi(50, 50, 200, 200)
        self.info_view.set_left_state(120.0, 220.0, 5.0, 10.0)
        self.info_view.set_right_state(170.0, 240.0, 7.0, 12.0)
        
        logger.info("Direct UI update completed")
    
    def toggle_auto_test(self):
        """자동 테스트 시작/중지 토글"""
        if not self.auto_testing:
            self.timer.start(2000)  # 2초마다 테스트
            self.auto_test_button.setText("자동 테스트 중지")
            self.auto_testing = True
            logger.info("Automatic testing started")
        else:
            self.timer.stop()
            self.auto_test_button.setText("자동 테스트 시작")
            self.auto_testing = False
            logger.info("Automatic testing stopped")
    
    def auto_test(self):
        """자동 신호 테스트를 랜덤하게 수행합니다."""
        test_funcs = [
            self.mock_controller.emit_test_signal,
            self.mock_controller.emit_roi_signal,
            self.mock_controller.emit_mask_signal,
            self.mock_controller.emit_prediction_signal,
            self.mock_analyzer.emit_tracking_signal,
            lambda: self.mock_analyzer.emit_in_out_signal(True),
            lambda: self.mock_analyzer.emit_in_out_signal(False),
            self.mock_analyzer.emit_court_position_signal
        ]
        
        # 랜덤하게 하나의 테스트 함수 실행
        import random
        random.choice(test_funcs)()
        
        logger.info("Auto test executed")


def main():
    """메인 테스트 함수"""
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 