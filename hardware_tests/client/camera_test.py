import socketio
import cv2
import time
import numpy as np
import atexit
from picamera2 import Picamera2
from board import SCL, SDA
import busio
from gpiozero import DistanceSensor
from adafruit_pca9685 import PCA9685
from adafruit_motor import motor, servo
import base64

# 모터 및 서보모터 초기화
MOTOR_M1_IN1 = 15
MOTOR_M1_IN2 = 14
i2c = busio.I2C(SCL, SDA)
pca = PCA9685(i2c, address=0x5f)  # 기본 주소는 0x5f
pca.frequency = 50


# 서버 주소
SERVER_URL = 'http://192.168.0.11:5000'  # <SERVER_IP>를 서버 IP로 변경

camera_up_down = servo.Servo(pca.channels[4], min_pulse=500, max_pulse=2400, actuation_range=160)
camera_up_down.angle = 145

# 소켓 클라이언트 초기화
sio = socketio.Client()


@sio.event
def connect():
    print("서버에 연결되었습니다.")

@sio.event
def disconnect():
    print("서버와의 연결이 종료되었습니다.")

@sio.on('stop_signal')
def on_stop_signal(data):
    reason = data.get('reason', 'unknown')
    if reason == 'person_detected':
        print("사람이 감지되어 자동차를 정지합니다.")
    else:
        print("알 수 없는 이유로 자동차를 정지합니다.")

def main():
    try:
        sio.connect(SERVER_URL)
    except Exception as e:
        print(f"서버에 연결할 수 없습니다: {e}")
        return

    # Picamera2 초기화
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'BGR888', "size": (640, 480)}))
    picam2.start()

    try:
        while True:
            # 카메라로 프레임 캡처
            frame = picam2.capture_array()
            if frame is None or frame.size == 0:
                print("캡처된 프레임이 없습니다. 다음 프레임을 기다립니다.")
                continue

            # JPEG 인코딩
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                print("프레임 인코딩 실패")
                continue

            # Base64 인코딩 후 서버로 전송
            encoded_frame = base64.b64encode(buffer).decode('utf-8')
            sio.emit('frame', {'frame': encoded_frame})

            # 전송 속도 제어
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("사용자 중지로 종료합니다.")
    finally:
        picam2.stop()
        sio.disconnect()


if __name__ == '__main__':
    main()
