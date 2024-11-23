import socketio
import cv2
import time
import numpy as np
import atexit
from picamera2 import Picamera2
from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import motor, servo
import base64
import json

# 모터 및 서보모터 초기화
MOTOR_M1_IN1 = 15
MOTOR_M1_IN2 = 14
i2c = busio.I2C(SCL, SDA)
pca = PCA9685(i2c, address=0x5f)  # 기본 주소는 0x5f
pca.frequency = 50

# DC 모터
motor1 = motor.DCMotor(pca.channels[MOTOR_M1_IN1], pca.channels[MOTOR_M1_IN2])
motor1.decay_mode = motor.SLOW_DECAY

# 앞바퀴 조향
servo_steering = servo.Servo(pca.channels[0], min_pulse=500, max_pulse=2400, actuation_range=160)

# 프로그램 종료 시 실행되는 함수
def cleanup():
    print("프로그램 종료 중, 모터 초기화 중...")
    motor1.throttle = 0
    servo_steering.angle = 90  # 초기화 시 기본 위치
    pca.deinit()
    print("모터 초기화 완료.")

atexit.register(cleanup)

# 소켓 클라이언트 초기화
sio = socketio.Client()

@sio.event
def connect():
    print("서버에 연결되었습니다.")

@sio.event
def disconnect():
    print("서버와의 연결이 종료되었습니다.")
    # 재연결 로직 추가
    attempt_reconnect()

@sio.on('registration_success')
def on_registration_success(data):
    car_number = data['car_number']
    print(f"자동차 등록 성공: 자동차 번호 {car_number}")

@sio.on('registration_failed')
def on_registration_failed(data):
    print("자동차 등록 실패:", data['message'])

@sio.on('error')
def on_error(data):
    print("에러 발생:", data['message'])

@sio.on('steering_angle')
def on_steering_angle(data):
    angle = data['angle']
    if 0 <= angle <= 160:  # 서버의 actuation_range와 맞춤
        servo_steering.angle = angle
        print(f"서버로부터 받은 조향각: {angle}도")
        time.sleep(0.1)

@sio.on('speed')
def on_speed(data):
    speed = data['speed']
    motor1.throttle = speed
    print(f"서버로부터 받은 속도: {speed}")

@sio.on('stop_signal')
def on_stop_signal(data):
    motor1.throttle = 0
    reason = data.get('reason', 'unknown')
    if reason == 'person_detected':
        print("사람이 감지되어 자동차를 정지합니다.")
    elif reason == 'ultrasonic':
        distance = data.get('distance', 'unknown')
        print(f"초음파 센서에 의해 정지: 거리 {distance}m")
    else:
        print("알 수 없는 이유로 자동차를 정지합니다.")

def attempt_reconnect():
    """서버와의 재연결 시도"""
    while True:
        try:
            print("서버와 재연결 시도 중...")
            sio.connect('http://192.168.0.9:5000')
            print("서버와 재연결 성공")
            break
        except Exception as e:
            print(f"재연결 실패: {e}")
            time.sleep(5)

def main():
    # Picamera2 설정
    try:
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(main={"format": 'BGR888', "size": (320, 240)}))
        picam2.start()
    except Picamera2Error as e:
        print(f"Picamera2 초기화 실패: {e}")
        return

    try:
        sio.connect('http://192.168.0.8:5000')  # 서버 주소와 포트 확인
    except Exception as e:
        print(f"서버에 연결할 수 없습니다: {e}")
        attempt_reconnect()

    while True:
        try:
            frame = picam2.capture_array()
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 30])  # JPEG 압축률 설정

            # 임의의 초음파 거리 데이터 추가 (센서를 연결한 경우 값 갱신 필요)
            distance = 0.6  # 예제 거리 값

            # Base64로 프레임 인코딩
            encoded_frame = base64.b64encode(buffer).decode('utf-8')

            # JSON 형식으로 서버로 데이터 전송
            sio.emit('frame', json.dumps({
                'frame': encoded_frame,
                'distance': distance  # 초음파 데이터 포함
            }))
            time.sleep(0.1)
        except Exception as e:
            print(f"프레임 전송 실패: {e}")
            time.sleep(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("사용자에 의해 프로그램이 종료되었습니다.")
    finally:
        cleanup()
