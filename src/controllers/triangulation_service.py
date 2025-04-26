def set_camera(self, config):
    """
    Set camera parameters from config.
    
    Args:
        config: Camera configuration dictionary
    """
    if not config:
        logging.error("No camera configuration provided")
        return
        
    try:
        # 베이스라인 값을 설정 파일에서 가져옴
        if "baseline_m" in config:
            self.baseline = config["baseline_m"]
        else:
            logging.warning("Baseline distance not specified in config, using default")
            self.baseline = 0.5  # 기본값 (미터)
            
        # 초점거리 계산 - 센서 크기로 정규화 적용
        if "focal_length_mm" in config and "sensor_width_mm" in config and "image_width_px" in config:
            # 물리적 초점거리(mm)를 픽셀 단위로 변환 (normalized)
            self.focal_length = (config["focal_length_mm"] / config["sensor_width_mm"]) * config["image_width_px"]
        elif "focal_length_px" in config:
            # 직접 픽셀 단위로 지정된 경우
            self.focal_length = config["focal_length_px"]
        else:
            logging.warning("Focal length not properly specified in config, using default")
            self.focal_length = 1000.0  # 기본값 (픽셀)
            
        # 다른 파라미터들 처리
        # ... 기존 코드 ...
    
    except KeyError as e:
        logging.error(f"Missing required camera parameter in config: {e}")
    except Exception as e:
        logging.error(f"Error setting camera parameters: {e}")
        
    logging.info(f"Triangulation parameters set: baseline={self.baseline:.3f}m, focal_length={self.focal_length:.1f}px") 