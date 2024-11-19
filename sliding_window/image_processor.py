# image_processor.py
"""
이미지 전처리를 위한 클래스
CLAHE, 이진화, 모폴로지 연산 등을 수행
"""

import cv2
import numpy as np

class ImageProcessor:
    @staticmethod
    def contrast_clahe(image):
        """
        CLAHE(Contrast Limited Adaptive Histogram Equalization)를 적용하여 
        이미지의 대비를 향상시킴
        
        Args:
            image: 입력 이미지
        Returns:
            대비가 향상된 이미지
        """
        clahe = cv2.createCLAHE(clipLimit=0.2, tileGridSize=(8, 8))
        ycrcb_mat = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb_planes = list(cv2.split(ycrcb_mat))
        ycrcb_planes[0] = clahe.apply(ycrcb_planes[0])  # 밝기 채널에만 CLAHE 적용
        ycrcb_mat = cv2.merge(ycrcb_planes)
        return cv2.cvtColor(ycrcb_mat, cv2.COLOR_YCrCb2BGR)

    @staticmethod
    def binary_threshold(image):
        """
        HLS 색공간과 Otsu 이진화를 결합하여 차선을 검출
        
        Args:
            image: 입력 이미지
        Returns:
            이진화된 이미지
        """
        # 노이즈 제거를 위한 가우시안 블러
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        
        # HLS 색공간에서 흰색 차선 검출
        img_hls = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
        lower_white = np.array([0, 200, 0])
        upper_white = np.array([255, 255, 255])
        hls_mask = cv2.inRange(img_hls, lower_white, upper_white)
        
        # Otsu 이진화로 보완
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return cv2.bitwise_and(hls_mask, otsu_mask)

    @staticmethod  
    def morphological_transform(image):
        """
        모폴로지 닫힘 연산으로 노이즈 제거 및 끊어진 차선 연결
        
        Args:
            image: 이진화된 이미지
        Returns:
            모폴로지 연산이 적용된 이미지
        """
        kernel = np.ones((6, 6), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)