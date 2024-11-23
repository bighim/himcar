import numpy as np
import time

# 저장 디렉토리 설정
OUTPUT_DIR = "frames"

class SteeringController:
    def __init__(self):
        # 이미지 관련
        self.image_center = 160
        self.lane_width_pixels = 320
        
        # 실제 물리적 거리
        self.lane_width_cm = 22
        self.camera_forward_distance = 40
        
        # cm_per_pixel 계산 추가
        self.cm_per_pixel = self.lane_width_cm / self.lane_width_pixels
        
        # PID 제어 관련
        self.kp = 0.27  # 기존의 error_scaling_factor 값을 Kp로 사용
        self.ki = 0.000  # 누적 오차에 대한 보정
        self.kd = 0.000   # 급격한 변화 방지
        
        # 상태 저장 변수
        self.last_error = 0
        self.error_sum = 0
        self.last_steering_angle = 0
        self.last_time = time.time()
        
        # 적분항 제한
        self.max_error_sum = 30

    def calculate_steering_angle(self, left_fit, right_fit, speed):
        """
        차선 피팅 값을 기반으로 조향각 계산
        """
        try:
            # 하나의 y_eval 포인트만 사용
            y_eval = int(240 * 0.2)  # 이미지의 60% 지점
            
            # 중심점 계산
            if left_fit is not None and right_fit is not None:
                left_x = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
                right_x = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]
                center_x = (left_x + right_x) / 2
                
            elif left_fit is not None:
                left_x = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
                center_x = left_x + self.lane_width_pixels/2
                
            elif right_fit is not None:
                right_x = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]
                center_x = right_x - self.lane_width_pixels/2
            else:
                return self.last_steering_angle
            
            # 픽셀 단위 오차
            error_pixels = center_x - self.image_center
            
            # 시간 간격 계산
            current_time = time.time()
            dt = current_time - self.last_time
            
            # PID 제어
            error = error_pixels  # 현재 오차
            
            # P항: 비례 제어
            p_term = self.kp * error
            
            # I항: 적분 제어 (누적 오차)
            self.error_sum = np.clip(self.error_sum + error * dt, 
                                   -self.max_error_sum, self.max_error_sum)
            i_term = self.ki * self.error_sum
            
            # D항: 미분 제어 (변화율)
            d_term = 0
            if dt > 0:
                d_term = self.kd * (error - self.last_error) / dt
            
            # 조향각 계산
            steering_angle = p_term + i_term + d_term
            
            # 최대 조향각 제한
            max_angle = 65
            steering_angle = np.clip(steering_angle, -max_angle, max_angle)
            
            # 상태 업데이트
            self.last_error = error
            self.last_time = current_time
            self.last_steering_angle = steering_angle
            
            return steering_angle
            
        except Exception as e:
            print(f"조향각 계산 중 오류 발생: {e}")
            return self.last_steering_angle

    def visualize_steering(self, image, steering_angle):
        """
        조향 정보를 이미지에 시각화
        """
        import cv2
        
        # 중앙선 그리기
        cv2.line(image, (self.image_center, 240), (self.image_center, 0), (0, 255, 0), 1)
        
        # 조향 방향 화살표 그리기
        arrow_length = 50
        arrow_x = self.image_center + int(arrow_length * np.sin(np.radians(steering_angle)))
        arrow_y = 200  # 이미지 상단에서 40픽셀 아래
        cv2.arrowedLine(image, (self.image_center, 240), (arrow_x, arrow_y), 
                       (0, 0, 255), 2)
        
        # 조향각 텍스트 표시
        cv2.putText(image, f"Steering: {steering_angle:.1f}deg", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        
        return image
