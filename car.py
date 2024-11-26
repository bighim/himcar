import cv2
import numpy as np
import time

# image_setting.py와 sliding_window.py 모듈에서 필요한 클래스와 함수 가져오기
from sliding_window import SlidingWindow
from steering_controller import SteeringController

import numpy as np
import os

# 이미지 저장 경로 설정
SAVE_DIR = "frames"

class Car:
    def __init__(self, car_number, sid, initial_speed=0.0):
        self.number = car_number
        self.sid = sid
        self.speed = initial_speed
        self.frame_count = 0
        self.frame_start_time = time.time()
        self.window = SlidingWindow()
        self.steering_controller = SteeringController()
        
        # 처리 시간 관련 변수 추가
        self.processing_times = []  # 최근 10개 프레임의 처리 시간 저장
        
    def increase_frame_count(self):
        self.frame_count += 1
        
    def update_speed(self, new_speed):
        self.speed = new_speed
        
    def get_save_dir(self):
        save_dir = os.path.join(SAVE_DIR, f"car_{self.number}")
        os.makedirs(save_dir, exist_ok=True)
        return save_dir
        
    def add_processing_time(self, processing_time):
        self.processing_times.append(processing_time)
        # 최근 10개의 프레임만 유지
        if len(self.processing_times) > 10:
            self.processing_times.pop(0)
            
    def get_processing_time(self):
        if not self.processing_times:
            return 0
        return sum(self.processing_times)