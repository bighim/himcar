from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import motor
from gpiozero import DistanceSensor
from picamera2 import Picamera2
from adafruit_motor import servo
import cv2
import time
import threading

# I2C 및 PCA9685 설정
MOTOR_M1_IN1 = 15
MOTOR_M1_IN2 = 14
i2c = busio.I2C(SCL, SDA)
pca = PCA9685(i2c, address=0x5f)  # 기본 주소는 0x5f
pca.frequency = 50

# DC 모터 설정
motor1 = motor.DCMotor(pca.channels[MOTOR_M1_IN1], pca.channels[MOTOR_M1_IN2])
motor1.decay_mode = motor.SLOW_DECAY

# 초음파 센서 설정
Tr = 23
Ec = 24
sensor = DistanceSensor(echo=Ec, trigger=Tr, max_distance=2)  # 최대 감지 거리 2m

# 서보 모터 설정
MIN_STEERING_ANGLE = 0
MAX_STEERING_ANGLE = 140
servo_steering = servo.Servo(pca.channels[0], min_pulse=500, max_pulse=2400, actuation_range=160)

# 카메라 서보 초기화
camera_left_right = servo.Servo(pca.channels[2], min_pulse=500, max_pulse=2400, actuation_range=160)
camera_up_down = servo.Servo(pca.channels[4], min_pulse=500, max_pulse=2400, actuation_range=160)

# Picamera2 설정
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'BGR888', "size": (640, 480)}))

STOP = False

# 프로그램 종료 시 실행되는 함수
def cleanup():
    print("프로그램 종료 중, 모터 초기화 중...")
    motor1.throttle = 0
    servo_steering.angle = 77
    camera_left_right.angle = 75
    camera_up_down.angle = 45
    pca.deinit()
    # picam2.close()
    print("모터 초기화 완료.")
    
# 초기화된 각도 및 스로틀 값 변수
# 서보모터 정면각도, 수정 X
motor_throttle = 0
servo_angle = 77
camera_lr_angle = 75
camera_ud_angle = 74

previous_motor_throttle = 0
previous_servo_angle = 0
previous_camera_lr_angle = 0
previous_camera_ud_angle = 0


def inputThread():
    global STOP, servo_angle, camera_lr_angle, camera_ud_angle, motor_throttle
    while not STOP:
        try:
            user_input = input("명령을 입력하세요 (종료하려면 'exit'): ")
            if user_input == "exit":
                STOP = True
                break
            tokens = user_input.strip().split()
            if tokens[0] == '0':  # 서보 모터(조향) 각도
                angle = int(tokens[1])
                if MIN_STEERING_ANGLE <= angle <= MAX_STEERING_ANGLE:
                    servo_angle = angle
                else:
                    print(f"서보 각도는 {MIN_STEERING_ANGLE}에서 {MAX_STEERING_ANGLE} 사이여야 합니다.")
            elif tokens[0] == '1':  # 카메라 좌우 각도
                angle = int(tokens[1])
                if 0 <= angle <= 180:
                    camera_lr_angle = angle
                else:
                    print("카메라 좌우 각도는 0에서 180 사이여야 합니다.")
            elif tokens[0] == '2':  # 카메라 상하 각도
                angle = int(tokens[1])
                if 0 <= angle <= 180:
                    camera_ud_angle = angle
                else:
                    print("카메라 상하 각도는 0에서 180 사이여야 합니다.")
            elif tokens[0] == 't':  # DC 모터 스로틀
                throttle = float(tokens[1])
                if -1.0 <= throttle <= 1.0:
                    motor_throttle = throttle
                else:
                    print("스로틀 값은 -1.0에서 1.0 사이여야 합니다.")
            else:
                print("알 수 없는 명령입니다.")
        except Exception as e:
            print(f"입력 처리 중 오류 발생: {e}")
            
def main():
    global previous_camera_lr_angle, previous_camera_ud_angle, previous_motor_throttle, previous_servo_angle
    # picam2.start()
    
    # 입력 쓰레드 시작
    t_input = threading.Thread(target=inputThread, daemon=True)
    t_input.start()
    
    try:
        while not STOP:
            # 변경 사항만 반영
            if previous_servo_angle != servo_angle:
                servo_steering.angle = servo_angle
                previous_servo_angle = servo_angle
                print(f"서보 각도 업데이트: {servo_angle}")
            
            if previous_camera_lr_angle != camera_lr_angle:
                camera_left_right.angle = camera_lr_angle
                previous_camera_lr_angle = camera_lr_angle
                print(f"카메라 좌우 각도 업데이트: {camera_lr_angle}")
            
            if previous_camera_ud_angle != camera_ud_angle:
                camera_up_down.angle = camera_ud_angle
                previous_camera_ud_angle = camera_ud_angle
                print(f"카메라 상하 각도 업데이트: {camera_ud_angle}")
            
            if previous_motor_throttle != motor_throttle:
                motor1.throttle = motor_throttle
                previous_motor_throttle = motor_throttle
                print(f"DC 모터 스로틀 업데이트: {motor_throttle}")

            # 프레임 캡처 및 거리 출력
            # frame = picam2.capture_array()
            # _, buffer = cv2.imencode('.jpg', frame)
            # cv2.imwrite('/home/pi/Hanhim/live/live.jpg', frame)
            distance = sensor.distance
            print(f"거리: {distance:.2f}m")
            time.sleep(0.1)
    finally:
        cleanup()

if __name__ == '__main__':
    main()
