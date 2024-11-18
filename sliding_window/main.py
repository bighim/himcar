import cv2
import numpy as np

# image_setting.py와 sliding_window.py 모듈에서 필요한 클래스와 함수 가져오기
from sliding_window import SlidingWindow

from flask import Flask
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import os
import traceback

app = Flask(__name__)
socketio = SocketIO(app)

window = SlidingWindow()
frame_number = 0

# 저장 디렉토리 설정
OUTPUT_DIR = "frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@socketio.on('connect')
def handle_connect():
    print("클라이언트가 연결되었습니다.")

@socketio.on('disconnect')
def handle_disconnect():
    print("클라이언트와의 연결이 종료되었습니다.")

@socketio.on('frame')
def handle_frame(data):
    global frame_number
    try:
        # 클라이언트로부터 받은 데이터 처리
        frame_data = data['frame']
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 현재 프레임 번호 가져오기
        capture_frame = frame_number
        frame_number += 1

        # CLAHE 적용한 이미지
        clahe_image = window.contrast_clihe(frame)
        
        # 이진화 테스트 (적응형 임계값)
        test = window.binary_image_with_adaptivethreshold(clahe_image)

        # 이미지 워핑
        warped_image = window.warp_image(clahe_image)

        # 이진화 처리 (적응형 임계값)
        lane = window.binary_image_with_adaptivethreshold(warped_image)

        # 형태학적 변환
        morphological_transformation_image = window.morphological_transformation(lane)

        # 슬라이딩 윈도우 적용
        sliding_window_image = window.draw_sliding_window(morphological_transformation_image, capture_frame)

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

        # 이미지 출력
        cv2.imwrite("frame.jpg", frame)
        cv2.imwrite("clahe_image.jpg", clahe_image)
        cv2.imwrite("test.jpg", test)
        cv2.imwrite("warped image.jpg", warped_image)
        cv2.imwrite("adap_thresh.jpg", lane)
        cv2.imwrite("morph_image.jpg", morphological_transformation_image)
        cv2.imwrite("sliding_window_image_adap.jpg", sliding_window_image)
        
    except Exception as e:
        print(f"프레임 처리 중 오류 발생: {e}")
        print(traceback.format_exc())
        

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)