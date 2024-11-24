import socketio
import time
import numpy as np
import atexit
from picamera2 import Picamera2
from board import SCL, SDA
import busio
from gpiozero import DistanceSensor
from adafruit_pca9685 import PCA9685
from adafruit_motor import motor, servo
import time
import cv2
import base64
import json

RUNNING_STATE = False

class CarConfig:
    # GPIO 핀 설정
    MOTOR_M1_IN1 = 15
    MOTOR_M1_IN2 = 14
    TRIGGER_PIN = 23
    ECHO_PIN = 24
    
    # 서보모터 설정
    CAMERA_HORIZONTAL_DEFAULT = None
    CAMERA_VERTICAL_DEFAULT = None
    SERVO_STEERING_DEFAULT = None
    SERVO_MIN_PULSE = 500
    SERVO_MAX_PULSE = 2400
    SERVO_RANGE = 160
    
    # PCA9685 설정
    PCA_ADDRESS = 0x5f
    PCA_FREQUENCY = 50
    
    # 카메라 설정
    CAMERA_FORMAT = 'BGR888'
    CAMERA_SIZE = (320, 240)
    
    # 서버 설정
    SERVER_URL = 'http://192.168.0.x:5000'
    
    # 차량 번호
    CAR_NUMBER = None

class CarController:
    def __init__(self):
        self.init_hardware()
        self.init_camera()
        self.init_socket()
        atexit.register(self.cleanup)
    
    def init_hardware(self):
        # I2C 및 PCA9685 초기화
        self.i2c = busio.I2C(SCL, SDA)
        self.pca = PCA9685(self.i2c, address=CarConfig.PCA_ADDRESS)
        self.pca.frequency = CarConfig.PCA_FREQUENCY
        
        # 모터 초기화
        self.motor = motor.DCMotor(
            self.pca.channels[CarConfig.MOTOR_M1_IN1],
            self.pca.channels[CarConfig.MOTOR_M1_IN2]
        )
        self.motor.decay_mode = motor.SLOW_DECAY
        
        # 서보모터 초기화
        self.init_servos()
        
        # 초음파 센서 초기화
        self.distance_sensor = DistanceSensor(
            echo=CarConfig.ECHO_PIN,
            trigger=CarConfig.TRIGGER_PIN,
            max_distance=2
        )
    
    def init_servos(self):
         # 조향 서보
        self.steering = servo.Servo(
            self.pca.channels[0],
            min_pulse=CarConfig.SERVO_MIN_PULSE,
            max_pulse=CarConfig.SERVO_MAX_PULSE,
            actuation_range=CarConfig.SERVO_RANGE
        )
        self.steering.angle = CarConfig.SERVO_STEERING_DEFAULT
        
        # 카메라 서보
        self.camera_horizontal = servo.Servo(
            self.pca.channels[2],
            min_pulse=CarConfig.SERVO_MIN_PULSE,
            max_pulse=CarConfig.SERVO_MAX_PULSE,
            actuation_range=CarConfig.SERVO_RANGE
        )
        self.camera_horizontal.angle = CarConfig.CAMERA_HORIZONTAL_DEFAULT
        
        self.camera_vertical = servo.Servo(
            self.pca.channels[4],
            min_pulse=CarConfig.SERVO_MIN_PULSE,
            max_pulse=CarConfig.SERVO_MAX_PULSE,
            actuation_range=CarConfig.SERVO_RANGE
        )
        self.camera_vertical.angle = CarConfig.CAMERA_VERTICAL_DEFAULT
    
    def init_camera(self):
        self.camera = Picamera2()
        self.camera.configure(self.camera.create_preview_configuration(
            main={"format": CarConfig.CAMERA_FORMAT, "size": CarConfig.CAMERA_SIZE}
        ))
        self.camera.start()
    
    def init_socket(self):
        self.sio = socketio.Client()
        self.setup_socket_events()
    
    def setup_socket_events(self):
        @self.sio.event
        def connect():
            print("서버에 연결되었습니다.")
            # 연결 즉시 차량 등록 요청
            self.sio.emit('register_car', {'car_number': CarConfig.CAR_NUMBER})
            
        @self.sio.event
        def disconnect():
            print("서버와의 연결이 종료되었습니다.")
            
        @self.sio.on('registration_success')
        def on_registration_success(data):
            car_number = data['car_number']
            print(f"자동차 등록 성공: 자동차 번호 {car_number}")
            
        @self.sio.on('registration_failed')
        def on_registration_failed(data):
            print("자동차 등록 실패:", data['message'])
            
        @self.sio.on('error')
        def on_error(data):
            print("에러 발생:", data['message'])
            
        @self.sio.on('data')
        def on_data(data):
            if RUNNING_STATE:
                self.motor.throttle = data['speed']
                self.steering.angle = CarConfig.CAMERA_HORIZONTAL_DEFAULT + data['angle']
            else:
                self.motor.throttle = 0
            
        @self.sio.on('start_signal')
        def on_start_signal():
            print("시작 신호 수신")
            global RUNNING_STATE
            RUNNING_STATE = True
            
        @self.sio.on('stop_signal')
        def on_stop_signal():
            print("정지 신호 수신")
            global RUNNING_STATE
            RUNNING_STATE = False
    
    def cleanup(self):
        print("프로그램 종료 중, 모터 초기화 중...")
        self.motor.throttle = 0
        self.steering.angle = CarConfig.SERVO_STEERING_DEFAULT
        self.camera_horizontal.angle = CarConfig.CAMERA_HORIZONTAL_DEFAULT
        self.camera_vertical.angle = CarConfig.CAMERA_VERTICAL_DEFAULT
        self.pca.deinit()
        print("모터 초기화 완료.")
    
    def run(self):
        try:
            self.sio.connect(CarConfig.SERVER_URL)
            
            while True:
                frame = self.camera.capture_array()
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 30])  # JPEG 압축률 설정
                encoded_frame = base64.b64encode(buffer).decode('utf-8')
                distance = self.distance_sensor.distance
                self.sio.emit('frame', json.dumps({
                    'frame': encoded_frame,
                    'distance': distance  # 초음파 데이터 포함
                }))
                time.sleep(0.1)
                
        except Exception as e:
            print(f"오류 발생: {e}")
            return

if __name__ == '__main__':
    car = CarController()
    car.run()