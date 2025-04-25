# 클래스 선언부 내에서 Signal 이름 변경
# 기존: 
save_requested = Signal()
load_requested = Signal()

# 변경:
save_calibration_requested = Signal(str)
load_calibration_requested = Signal(str)

# 버튼 클릭 핸들러 추가
def _on_save_clicked(self):
    path, _ = QFileDialog.getSaveFileName(self, "Save Calibration", "", "JSON Files (*.json)")
    if path:
        self.save_calibration_requested.emit(path)
        logger.info(f"Requested to save calibration to: {path}")

def _on_load_clicked(self):
    path, _ = QFileDialog.getOpenFileName(self, "Load Calibration", "", "JSON Files (*.json)")
    if path:
        self.load_calibration_requested.emit(path)
        logger.info(f"Requested to load calibration from: {path}")

# 버튼 연결 부분 (원래 _connect_signals 메서드 내)
# 기존:
self.save_btn.clicked.connect(lambda: self.save_requested.emit())
self.load_btn.clicked.connect(lambda: self.load_requested.emit())

# 변경:
self.save_btn.clicked.connect(self._on_save_clicked)
self.load_btn.clicked.connect(self._on_load_clicked) 