import cv2
import numpy as np
from flask import Flask, request
from flask_socketio import SocketIO, emit
import numpy as np
import os
import traceback
import base64
import json
from threading import Thread
import logging


app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)  # ERROR 레벨 이상만 표시
socketio = SocketIO(app)

# 양방향 매핑을 위한 딕셔너리들
sid_to_car = {}  # sid -> car_number 매핑
car_to_sid = {}  # car_number -> sid 매핑

# 저장 디렉토리 설정
OUTPUT_DIR = "frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@socketio.on('connect')
def handle_connect():
    print("클라이언트가 연결되었습니다.")

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if sid in sid_to_car:
        car_number = sid_to_car[sid]
        # 양쪽 매핑에서 모두 제거
        del sid_to_car[sid]
        del car_to_sid[car_number]
        print(f"차량 연결 해제: {car_number} (sid: {sid})")
    
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
    if car_number in car_to_sid:
        error_message = "이미 등록된 차량 번호입니다."
        print(f"차량 등록 실패: {error_message} (sid: {sid})")
        socketio.emit('registration_failed', {'message': error_message}, room=sid)
        return
    
    # 양방향 매핑 저장
    sid_to_car[sid] = car_number
    car_to_sid[car_number] = sid
    
    print(f"차량 등록 완료: {car_number} (sid: {sid})")
    socketio.emit('registration_success', {'car_number': car_number}, room=sid)

@socketio.on('frame')
def handle_frame(data):
    try:
        sid = request.sid
        car_num = sid_to_car[sid]
        frame_data = json.loads(data)
        distance = frame_data.get('distance', None)
        
        # 먼저 프레임 디코딩
        encoded_frame = frame_data['frame']
        jpg_data = base64.b64decode(encoded_frame)
        nparr = np.frombuffer(jpg_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 그 다음 cropped_frame 생성
        height = frame.shape[0]
        cropped_frame = frame[int(height // 2):, :]
        
        # 라인 검출 및 조향각 계산
        processed_frame, steering_angle = process_line(frame)
        
        # 디버깅 이미지 저장
        save_debug_images(frame, processed_frame, cropped_frame, car_num)
        
        speed = 0.50 if abs(steering_angle - 90) < 10 else 0.40
        
        # 하나의 'data' 이벤트로 통합하여 전송
        socketio.emit('data', {
            'speed': speed,
            'angle': steering_angle - 90
        }, room=sid)
        
    except Exception as e:
        car_num = sid_to_car[sid]
        print(f"{car_num}번 자동차 프레임 처리 중 오류 발생: {e}")
        print(traceback.format_exc())
        
def decode_frame(encoded_frame):
    """Base64로 인코딩된 프레임 데이터를 디코딩"""
    jpg_data = base64.b64decode(encoded_frame)
    nparr = np.frombuffer(jpg_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def process_line(frame):
    """차선 검출 및 조향각 계산"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 80, 140)

    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[  # ROI 설정
        (0, height), (0, height // 2), (width, height // 2), (width, height)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    roi = cv2.bitwise_and(edges, mask)

    contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    steering_angle = 90  # 기본 조향각

    left_lines, right_lines = split_contours_by_center(contours, width)
    if left_lines or right_lines:
        steering_angle = calculate_steering_angle(left_lines, right_lines, width)

    # 디버깅 시각화를 위해 차선과 중심선 그리기
    draw_debug_lines(frame, left_lines, right_lines, steering_angle, width, height)

    return frame, steering_angle

def split_contours_by_center(contours, width):
    """윤곽선을 화면 중심 기준으로 좌우로 분리"""
    left_lines, right_lines = [], []
    for contour in contours:
        x, _, w, _ = cv2.boundingRect(contour)
        center_x = x + w // 2
        if center_x < width // 2:
            left_lines.append(contour)
        else:
            right_lines.append(contour)
    return left_lines, right_lines

def calculate_steering_angle(left_lines, right_lines, width):
    """좌우 차선 윤곽선 중심을 기반으로 조향각 계산"""
    left_center = calculate_center_of_mass(left_lines)
    right_center = calculate_center_of_mass(right_lines)
    if left_center and right_center:
        mid_x = (left_center + right_center) // 2
        center_offset = mid_x - width // 2
        angle = 90 - center_offset / (width // 2) * 20
        return max(0, min(180, angle))
    elif left_center:
        return 68  # 왼쪽 차선만 감지됨
    elif right_center:
        return 113  # 오른쪽 차선만 감지됨
    return 90

def calculate_center_of_mass(lines):
    """윤곽선들의 중심 좌표를 계산"""
    if not lines:
        return None
    moments = [cv2.moments(contour) for contour in lines]
    centers = [int(m['m10'] / m['m00']) for m in moments if m['m00'] != 0]
    return int(np.mean(centers)) if centers else None

def draw_debug_lines(frame, left_lines, right_lines, steering_angle, width, height):
    """디버깅을 위해 차선과 조향각 시각화"""
    for contour in left_lines:
        cv2.drawContours(frame, [contour], -1, (255, 0, 0), 3)  # 파란색
    for contour in right_lines:
        cv2.drawContours(frame, [contour], -1, (0, 0, 255), 3)  # 빨간색

    # 조향각 표시
    arrow_length = 100
    end_x = int(width // 2 + arrow_length * np.sin(np.radians(steering_angle - 90)))
    end_y = int(height - arrow_length * np.cos(np.radians(steering_angle - 90)))
    cv2.arrowedLine(frame, (width // 2, height), (end_x, end_y), (0, 255, 0), 5)

def check_person_detection(results, frame):
    """YOLO 검출 결과에서 사람 감지 여부 확인 및 시각적 표시"""
    if results.xyxy[0].size(0) > 0:
        for *box, conf, cls in results.xyxy[0]:
            if int(cls) == 0:  # 사람 클래스
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # 노란색 박스
                cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                return True
    return False

def save_debug_images(frame, processed_frame, cropped_frame, car_num):
    """디버깅용 이미지 저장"""
    # 차량별 디버그 폴더 생성
    car_debug_dir = os.path.join('debug', str(car_num))
    os.makedirs(car_debug_dir, exist_ok=True)
    
    # 각 이미지를 차량별 폴더에 고정된 이름으로 저장
    cv2.imwrite(os.path.join(car_debug_dir, 'original.jpg'), frame)
    cv2.imwrite(os.path.join(car_debug_dir, 'processed.jpg'), processed_frame)
    cv2.imwrite(os.path.join(car_debug_dir, 'cropped.jpg'), cropped_frame)
    
def input_handler():
    while True:
        print("\n명령어를 입력하세요 (1: 시작, 0: 정지)> ", end='', flush=True)
        cmd = input()
        if cmd == '1':
            print("시작 신호를 모든 차량에 전송합니다.")
            socketio.emit('start_signal')
        elif cmd == '0':
            print("정지 신호를 모든 차량에 전송합니다.")
            socketio.emit('stop_signal')
        else:
            print("잘못된 명령어입니다. 1 또는 0을 입력하세요.")
    
if __name__ == '__main__':
    # 입력 처리를 위한 별도 스레드 시작
    input_thread = Thread(target=input_handler)
    input_thread.daemon = True  # 메인 프로그램 종료시 함께 종료
    input_thread.start()
    
    socketio.run(app, host='0.0.0.0', port=5000)