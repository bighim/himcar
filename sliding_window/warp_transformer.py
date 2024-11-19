# warp_transformer.py
"""
차선 검출을 위한 버드아이뷰(Bird's eye view) 변환을 수행하는 클래스
"""

import cv2
import numpy as np
from config import *

class WarpTransformer:
    """
    이미지 워핑을 위한 클래스
    원근 변환 행렬을 계산하고 이미지/좌표 변환을 수행
    """
    
    def __init__(self):
        """
        변환 행렬 초기화 및 계산
        """
        self.src_to_dist_mtx = None  # 원본->버드아이뷰 변환 행렬
        self.dist_to_src_mtx = None  # 버드아이뷰->원본 변환 행렬
        self.warp_src_mtx = None     # 워핑 소스 좌표
        self._calculate_transform()

    def _calculate_transform(self):
        """
        원근 변환을 위한 변환 행렬 계산
        """
        # 워핑을 위한 소스 및 목적지 좌표 설정
        src_pts = [
            (240 - 180, HALF_HEIGHT + 60),  # 좌상단
            (0, IMAGE_HEIGHT - 40),         # 좌하단
            (400 + 180, HALF_HEIGHT + 60),  # 우상단
            (IMAGE_WIDTH, IMAGE_HEIGHT - 40) # 우하단
        ]
        dst_pts = [
            (0, 0),                # 좌상단
            (0, HALF_HEIGHT),      # 좌하단
            (HALF_WIDTH, 0),       # 우상단
            (HALF_WIDTH, HALF_HEIGHT) # 우하단
        ]
        
        # 변환 행렬 계산
        self.warp_src_mtx = np.float32(src_pts)
        warp_dst_mtx = np.float32(dst_pts)
        self.src_to_dist_mtx = cv2.getPerspectiveTransform(self.warp_src_mtx, warp_dst_mtx)
        self.dist_to_src_mtx = cv2.getPerspectiveTransform(warp_dst_mtx, self.warp_src_mtx)

    def warp_image(self, image):
        """
        이미지를 버드아이뷰로 변환
        
        Args:
            image: 입력 이미지
        Returns:
            워핑된 이미지
        """
        warped = cv2.warpPerspective(image, self.src_to_dist_mtx, 
                                   (HALF_WIDTH, HALF_HEIGHT))
        # 디버깅을 위한 워핑 포인트 표시
        for point in self.warp_src_mtx:
            cv2.circle(image, tuple(map(int, point)), 5, (255, 0, 0), -1)
        return warped

    def warp_point(self, points):
        """
        검출된 차선 좌표를 원본 이미지 좌표로 역변환
        
        Args:
            points: 버드아이뷰 상의 차선 좌표 리스트
        Returns:
            원본 이미지 상의 차선 좌표 리스트
        """
        if not points:
            return []
        points = np.float32(points).reshape(-1, 1, 2)
        warped_points = cv2.perspectiveTransform(points, self.dist_to_src_mtx)
        return warped_points.reshape(-1, 2).tolist()