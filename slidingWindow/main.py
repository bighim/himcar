import cv2
import numpy as np
import scipy as sp

# image_setting.py와 sliding_window.py 모듈에서 필요한 클래스와 함수 가져오기
from sliding_window import SlidingWindow
from steering_controller import SteeringController

from flask import Flask
from flask_socketio import SocketIO, emit
import numpy as np
import os
import traceback

app = Flask(__name__)
socketio = SocketIO(app)

window = SlidingWindow()
frame_number = 0

# 전역 변수에 추가
steering_controller = SteeringController()

@socketio.on('connect')
def handle_connect():
    print("클라이언트가 연결되었습니다.")

@socketio.on('disconnect')
def handle_disconnect():
    print("클라이언트와의 연결이 종료되었습니다.")

@socketio.on('frame')
def handle_frame(data):
    global frame_number
    # 이미지 저장 경로 설정
    save_dir = "frames"
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 클라이언트로부터 받은 데이터 처리
        frame_data = data['frame']
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(os.path.join(save_dir, "step1_original_frame.jpg"), frame)
        
        # 현재 프레임 번호 가져오기
        capture_frame = frame_number
        frame_number += 1

        # CLAHE 적용한 이미지
        clahe_image = window.contrast_clihe(frame)
        cv2.imwrite(os.path.join(save_dir, "step2_clahe_enhanced.jpg"), clahe_image)
        
        # 이진화 테스트 (적응형 임계값)
        original_binary = window.binary_image_with_adaptivethreshold(clahe_image)
        cv2.imwrite(os.path.join(save_dir, "step3_binary_original_binary.jpg"), original_binary)

        # 이미지 워핑
        warped_image = window.warp_image(clahe_image)
        cv2.imwrite(os.path.join(save_dir, "step4_before_bird_eye_view.jpg"), clahe_image)
        cv2.imwrite(os.path.join(save_dir, "step5_after_bird_eye_view.jpg"), warped_image)
        
        # 이진화 처리 (적응형 임계값)
        lane = window.binary_image_with_adaptivethreshold(warped_image)
        cv2.imwrite(os.path.join(save_dir, "step8_after_hls_masking.jpg"), lane)

        # 형태학적 변환
        morphological_transformation_image = window.morphological_transformation(lane)
        cv2.imwrite(os.path.join(save_dir, "step9_after_morphological_transformation.jpg"), morphological_transformation_image)

        # 슬라이딩 윈도우 적용
        polynomial_image, left_fit, right_fit = window.fit_polynomial(morphological_transformation_image, capture_frame)
        cv2.imwrite(os.path.join(save_dir, "step10_polynomial_fitted.jpg"), polynomial_image)
        
        # polynomial_image 복사
        steering_visualization = polynomial_image.copy()
        speed = 0.7
        # 조향 정보를 이미지에 시각화
        steering_angle = steering_controller.calculate_steering_angle(left_fit, right_fit, speed)
        steering_visualization = steering_controller.visualize_steering(steering_visualization, steering_angle)
        cv2.imwrite(os.path.join(save_dir, "step11_steering_visualization.jpg"), steering_visualization)

        socketio.emit('data', {'angle': -steering_angle, 'speed': speed})
        
        # 좌우 차선 위치 계산
        if capture_frame % 10 == 0:
            lpos = window.warp_point(window.l_pos)
            rpos = window.warp_point(window.r_pos)
            # 좌표 초기화
            window.l_pos.clear()
            window.r_pos.clear()

            # 좌우 차선 위치 출력 또는 저장
            print(f"Left Positions: {lpos}")
            print(f"Right Positions: {rpos}")
            print(f"Left fit: {left_fit}")
            print(f"Right fit: {right_fit}")
        
        
    except Exception as e:
        print(f"프레임 처리 중 오류 발생: {e}")
        print(traceback.format_exc())
        

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)