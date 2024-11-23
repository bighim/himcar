import numpy as np

class SteeringController:
    def __init__(self):
        # 이미지 관련 (픽셀 단위)
        self.image_center = 160          # 워핑된 이미지의 중앙점 (320/2)
        self.lane_width_pixels = 320     # 워핑된 이미지에서의 차선 간 픽셀 거리
        
        # 실제 물리적 거리 (cm 단위)
        self.lane_width_cm = 22          # 실제 차선 간 거리
        self.camera_forward_distance = 30 # 차량 중심부터 카메라 인식 지점까지 거리
        
        # 픽셀당 실제 거리 비율 계산 (cm/pixel)
        self.cm_per_pixel = self.lane_width_cm / self.lane_width_pixels
        
        # 조향 제어 관련
        self.error_scaling_factor = 3.6  # 기하학적 계산 결과를 미세 조정
        self.last_steering_angle = 0

    def calculate_steering_angle(self, left_fit, right_fit):
        try:
            y_eval = int(240 * 0.6)
            
            if left_fit is not None and right_fit is not None:
                left_x = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
                right_x = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]
                center_x = (left_x + right_x) / 2
            elif right_fit is not None:
                right_x = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]
                center_x = right_x - self.lane_width_pixels/2
            elif left_fit is not None:
                left_x = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
                center_x = left_x + self.lane_width_pixels/2
            else:
                return self.last_steering_angle
                
            # 픽셀 단위 오차
            error_pixels = center_x - self.image_center
            
            # 실제 거리(cm) 단위로 변환
            error_cm = error_pixels * self.cm_per_pixel
            
            # 기하학적 계산으로 기본 조향각 계산
            base_angle = np.degrees(np.arctan2(error_cm, self.camera_forward_distance))
            
            # error_scaling_factor를 적용하여 조향각 미세 조정
            steering_angle = base_angle * self.error_scaling_factor
            
            # 조향각 제한
            max_angle = 60
            steering_angle = np.clip(steering_angle, -max_angle, max_angle)
            
            # 급격한 조향 변화 방지
            max_angle_change = 10
            angle_diff = steering_angle - self.last_steering_angle
            if abs(angle_diff) > max_angle_change:
                steering_angle = self.last_steering_angle + np.sign(angle_diff) * max_angle_change
                
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
        cv2.putText(image, f"Steering: {steering_angle:.1f}deg", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return image 