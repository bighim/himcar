# sliding_window.py
"""
차선 검출 파이프라인을 구현하는 메인 클래스
이미지 처리, 차선 검출, 좌표 변환 등을 통합 관리
"""

import os
import cv2
from config import *
from image_processor import ImageProcessor
from warp_transformer import WarpTransformer
from lane_detector import LaneDetector

class SlidingWindow:
    """
    차선 검출 파이프라인을 관리하는 메인 클래스
    각 단계별 처리 결과를 저장하고 시각화
    """
    
    def __init__(self):
        """클래스 초기화 및 구성요소 생성"""
        self.l_pos = []  # 좌측 차선 위치 저장
        self.r_pos = []  # 우측 차선 위치 저장
        
        # 각 처리 단계별 객체 생성
        self.image_processor = ImageProcessor()
        self.warp_transformer = WarpTransformer()
        self.lane_detector = LaneDetector()
        
        # 결과 저장 디렉토리 생성
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def process_frame(self, frame, frame_num):
        """
        입력 프레임에 대한 차선 검출 수행
        
        Args:
            frame: 입력 영상 프레임
            frame_num: 현재 프레임 번호
        Returns:
            transformed_l: 변환된 좌측 차선 좌표
            transformed_r: 변환된 우측 차선 좌표
        """
        # 이미지 전처리
        clahe = self.image_processor.contrast_clahe(frame)
        warped = self.warp_transformer.warp_image(clahe)
        binary = self.image_processor.binary_threshold(warped)
        morphed = self.image_processor.morphological_transform(binary)
        
        # 차선 검출
        l_pos, r_pos, result = self.lane_detector.detect_lanes(morphed, frame_num)
        
        # 중간 결과 저장
        self._save_debug_images(frame, clahe, warped, binary, morphed, result)
        
        # 검출된 차선 위치 변환 및 반환
        if frame_num % 10 == 0:
            self.l_pos.extend(l_pos)
            self.r_pos.extend(r_pos)
            transformed_l = self.warp_transformer.warp_point(self.l_pos)
            transformed_r = self.warp_transformer.warp_point(self.r_pos)
            self.l_pos.clear()
            self.r_pos.clear()
            return transformed_l, transformed_r
            
        return [], []

    def _save_debug_images(self, *images):
        """
        처리 단계별 결과 이미지 저장
        
        Args:
            *images: 저장할 이미지들 (원본, CLAHE, 워핑, 이진화, 모폴로지, 결과)
        """
        names = ["frame", "clahe", "warped", "binary", "morphed", "result"]
        for img, name in zip(images, names):
            cv2.imwrite(f"{OUTPUT_DIR}/{name}.jpg", img)