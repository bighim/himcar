import cv2
import numpy as np
import time

# image_setting.py와 sliding_window.py 모듈에서 필요한 클래스와 함수 가져오기
from car import Car
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
from torch.quantization import quantize_dynamic

app = Flask(__name__)
socketio = SocketIO(app)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)  # ERROR 레벨 이상만 표시
socketio = SocketIO(app)

# 로그 레벨 설정
warnings.filterwarnings("ignore", category=FutureWarning)

pathlib.PosixPath = pathlib.WindowsPath
# YOLOv5 저장소 경로와 사용자 정의 가중치 파일 경로
repo_path = "./yolov5"  # YOLOv5 소스코드가 있는 디렉토리
model_path = "./best_final.pt"  # 학습된 가중치 파일

# 이미지 저장 경로 설정
SAVE_DIR = "frames"

# YOLOv5 모델 로드 (커스텀 가중치 사용)
try:
    #YOLO 모델 구조를 로드하고, 커스텀 가중치를 적용
    model = torch.hub.load(repo_path, 'custom', path=model_path, source='local')
    model.eval()
    print("YOLOv5 커스텀 모델이 성공적으로 로드되었습니다.")
    # 모델 설정
    model.conf = 0.5  # 신뢰도 임계값
    model.max_det = 3  # 최대 검출 객체 수
    model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
except Exception as e:
    model = None
    print(f"모델 로드 중 오류 발생: {e}")
    
# 전역 변수 수정
cars: dict[int, Car] = {}  # car_number -> Car 객체

# 각 sid별 초기 속도 설정
initial_speeds = {
    1: 0.50,
    2: 0.45,
    3: 0.40,
}

platooning_mode = False
leader_speed = 0.0

# 저장 디렉토리 설정
OUTPUT_DIR = "frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@socketio.on('connect')
def handle_connect():
    print("클라이언트가 연결되었습니다.")

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    # sid로 차량 찾아서 제거
    for car_num, car in list(cars.items()):
        if car.sid == sid:
            del cars[car_num]
            print(f"차량 연결 해제: {car_num} (sid: {sid})")
            break

@socketio.on('register_car')
def handle_register_car(data):
    sid = request.sid
    car_number = data.get('car_number')
    
    if not car_number:
        error_message = "차량 번호가 없습니다."
        print(f"차량 등록 실패: {error_message} (sid: {sid})")
        socketio.emit('registration_failed', {'message': error_message}, room=sid)
        return
        
    # 이미 등록된 차량 번호인지 확인
    if car_number in cars:
        error_message = "이미 등록된 차량 번호입니다."
        print(f"차량 등록 실패: {error_message} (sid: {sid})")
        socketio.emit('registration_failed', {'message': error_message}, room=sid)
        return
    
    # 차량 객체 생성 및 등록
    initial_speed = initial_speeds.get(car_number, 0.0)
    cars[car_number] = Car(car_number, sid, initial_speed)
    
    print(f"차량 등록 완료: {car_number} (sid: {sid})")
    socketio.emit('registration_success', {'car_number': car_number}, room=sid)

@socketio.on('frame')
def handle_frame(data):
    try:
        start_time = time.time()
        sid = request.sid
        
        # sid로 차량 찾기
        car_num = None
        for num, car in cars.items():
            if car.sid == sid:
                car_num = num
                break

        car = cars[car_num]
        car.increase_frame_count()
        
        # JSON 파싱
        frame_data = json.loads(data)
        encoded_frame = frame_data['frame']
        distance = frame_data.get('distance', None)
        
        # 프레임 디코딩
        jpg_data = base64.b64decode(encoded_frame)
        nparr = np.frombuffer(jpg_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # YOLO 처리
        height = frame.shape[0]
        cropped_frame = frame[:int(height), :]
        results = model(cropped_frame, size=240)
        yolo_frame = np.squeeze(results.render())
        
        # 차량별 저장 디렉토리에 저장
        save_dir = car.get_save_dir()
        cv2.imwrite(os.path.join(save_dir, "yolo_frame.jpg"), yolo_frame)
        
        detected_classes = [int(cls) for *box, conf, cls in results.xyxy[0]]
        if 0 in detected_classes:
            car.update_speed(0)
            
        # 차선 검출 및 조향각 계산
        steering_angle = process_line(frame, car.window, car.steering_controller, car.frame_count, save_dir)
        
        # 플래튜닝 모드에 따라 속도 갱신
        if platooning_mode:
            if car_num == 1:  # 리더 차량
                leader_speed = car.speed
                print(f"리더 차량 {car_num} 속도 업데이트: {leader_speed}")
            else:
                car.update_speed(leader_speed)
        
        # 처리 시간 계산 및 저장
        processing_time = time.time() - start_time
        car.add_processing_time(processing_time)
        
        # 10개의 프레임마다 평균 처리 시간 출력
        if car.frame_count % 10 == 0:
            avg_processing_time = car.get_processing_time()
            print(f"차량 {car_num}: {avg_processing_time:.3f}초 소요 (프레임 {car.frame_count}개 처리)")
            socketio.emit('data', {
                'speed': car.speed,
                'angle': steering_angle,
                'processing_time': avg_processing_time,
                'frame_count': car.frame_count
            }, room=sid)
        else:
            socketio.emit('data', {
                'speed': car.speed,
                'angle': steering_angle
            }, room=sid)

    except Exception as e:
        print(f"프레임 처리 중 오류 발생: {e}")
        print(traceback.format_exc())

def input_handler():
    global platooning_mode
    while True:
        print("\n명령어를 입력하세요 (1: 시작, 0: 정지, p: 플래튜닝모드)> ", end='', flush=True)
        cmd = input()
        if cmd == '1':
            print("시작 신호를 모든 차량에 전송합니다.")
            socketio.emit('start_signal')
            platooning_mode = False
            for car_num, car in cars.items():
                car.update_speed(initial_speeds.get(car_num, 0.0))
        elif cmd == '0':
            print("정지 신호를 모든 차량에 전송합니다.")
            socketio.emit('stop_signal')
            platooning_mode = False
            for car_num, car in cars.items():
                car.update_speed(initial_speeds.get(car_num, 0.0))
                print(f'{car_num}을 {car.speed}으로 초기화')
        elif cmd == 'p':
            platooning_mode = not platooning_mode
            mode_status = "활성화" if platooning_mode else "비활성화"
            print(f"플래튜닝 모드가 {mode_status}되었습니다.")
        else:
            print("잘못된 명령어입니다. 1, 0, 또는 p를 입력하세요.")

def process_line(frame, window: SlidingWindow, steering_controller: SteeringController, frame_count, save_dir):
    """
    차선을 검출하고 조향각을 계산하는 함수
    """
    # 각 차량별로 이미지 저장
    cv2.imwrite(os.path.join(save_dir, "step1_original_frame.jpg"), frame)
    
    # CLAHE 적용
    clahe_image = window.contrast_clihe(frame)
    cv2.imwrite(os.path.join(save_dir, "step2_clahe_enhanced.jpg"), clahe_image)
    
    # 이진화 처리
    original_binary = window.binary_image_with_adaptivethreshold(clahe_image)
    cv2.imwrite(os.path.join(save_dir, "step3_binary_original_binary.jpg"), original_binary)

    # 이미지 워핑
    warped_image = window.warp_image(clahe_image)
    cv2.imwrite(os.path.join(save_dir, "step4_before_bird_eye_view.jpg"), clahe_image)
    cv2.imwrite(os.path.join(save_dir, "step5_after_bird_eye_view.jpg"), warped_image)
    
    # 워핑된 이미지 이진화
    lane = window.binary_image_with_adaptivethreshold(warped_image)
    cv2.imwrite(os.path.join(save_dir, "step8_after_hls_masking.jpg"), lane)

    # 형태학적 변환
    morphological_transformation_image = window.morphological_transformation(lane)
    cv2.imwrite(os.path.join(save_dir, "step9_after_morphological_transformation.jpg"), morphological_transformation_image)

    # 슬라이딩 윈도우 적용 (frame_count 제거)
    polynomial_image, left_fit, right_fit = window.fit_polynomial(morphological_transformation_image, frame_count)
    cv2.imwrite(os.path.join(save_dir, "step10_polynomial_fitted.jpg"), polynomial_image)
    
    # 조향각 계산 및 시각화
    steering_angle = steering_controller.calculate_steering_angle(left_fit, right_fit)
    processed_frame = steering_controller.visualize_steering(polynomial_image.copy(), steering_angle)
    cv2.imwrite(os.path.join(save_dir, "step11_steering_visualization.jpg"), processed_frame)
    steering_angle = steering_angle * -1
    return steering_angle

if __name__ == '__main__':
    # 입력 처리를 위한 별도 스레드 시작
    input_thread = Thread(target=input_handler)
    input_thread.daemon = True  # 메인 프로그램 종료시 함께 종료
    input_thread.start()
    
    socketio.run(app, host='0.0.0.0',port=5000)