import cv2
import numpy as np
import time

# image_setting.py와 sliding_window.py 모듈에서 필요한 클래스와 함수 가져오기
from sliding_window import SlidingWindow
from steering_controller import SteeringController
from flask import Flask, request

from flask import Flask
from flask_socketio import SocketIO, emit
import numpy as np
import os
import traceback

import logging
import warnings
import pathlib
import torch
import json
import base64
from threading import Thread


# YOLOv5 저장소 경로와 사용자 정의 가중치 파일 경로
repo_path = "./yolov5"  # YOLOv5 소스코드가 있는 디렉토리
model_path = "./best_final.pt"  # 학습된 가중치 파일

pathlib.PosixPath = pathlib.WindowsPath
# YOLOv5 모델 로드 (커스텀 가중치 사용)
try:
    #YOLO 모델 구조를 로드하고, 커스텀 가중치를 적용
    model = torch.hub.load(repo_path, 'custom', path=model_path, source='local')
    model.eval()
    print("YOLOv5 커스텀 모델이 성공적으로 로드되었습니다.")
    # 모델 설정
    model.conf = 0.4  # 신뢰도 임계값
    model.max_det = 3  # 최대 검출 객체 수
except Exception as e:
    model = None
    print(f"모델 로드 중 오류 발생: {e}")
    
print(vars(model))
