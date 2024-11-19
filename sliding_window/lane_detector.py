# lane_detector.py
"""
슬라이딩 윈도우 기법을 사용한 차선 검출기

이 모듈은 이진화된 이미지에서 차선을 검출하는 기능을 제공합니다.
히스토그램 분석과 슬라이딩 윈도우 방식으로 좌/우 차선을 검출하고,
연속된 프레임에서 차선을 추적합니다.
"""

from config import *
import cv2
import numpy as np

class LaneDetector:
    """
    차선 검출을 담당하는 클래스
    
    슬라이딩 윈도우 방식으로 차선을 검출하고 추적합니다.
    각 윈도우에서 히스토그램 분석으로 차선의 위치를 파악하고,
    이전 프레임의 검출 결과를 활용하여 연속성을 유지합니다.

    Attributes:
        before_l_detected (bool): 이전 프레임 좌측 차선 검출 여부
        before_r_detected (bool): 이전 프레임 우측 차선 검출 여부
        min_points (int): 윈도우당 최소 필요 픽셀 수
    """

    def __init__(self):
        """차선 검출기 초기화"""
        self.before_l_detected = True  
        self.before_r_detected = True
        self.min_points = None

    def detect_lanes(self, binary_img, frame_num):
        """
        입력 이미지에서 차선을 검출하고 시각화

        Args:
            binary_img (ndarray): 이진화된 입력 이미지
            frame_num (int): 현재 프레임 번호

        Returns:
            tuple: (l_pos, r_pos, out_img)
                - l_pos (list): 검출된 좌측 차선 좌표 [(x1,y1), ...]
                - r_pos (list): 검출된 우측 차선 좌표 [(x1,y1), ...]
                - out_img (ndarray): 시각화된 결과 이미지
        """
        # 윈도우 설정
        window_height = binary_img.shape[0] // NUM_WINDOWS
        self.min_points = window_height // 4
        
        # 결과 저장용 변수 초기화
        out_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        lx, rx = [], []  # 각 윈도우에서 검출된 차선 중심점
        l_pos, r_pos = [], []  # 최종 차선 위치
        
        # 상단에서 하단으로 윈도우 이동하며 차선 검출
        for window_idx in range(NUM_WINDOWS - 1, -1, -1):
            # 현재 윈도우 영역 설정
            roi_y_low = int(window_idx * binary_img.shape[0] / NUM_WINDOWS)
            roi_y_high = int((window_idx + 1) * binary_img.shape[0] / NUM_WINDOWS)
            window = binary_img[roi_y_low:roi_y_high, :]
            
            # 윈도우 영역의 히스토그램 계산
            histogram = cv2.reduce(window, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S).flatten()
            mid_point = binary_img.shape[1] // 2

            # 차선 위치 검출
            if window_idx == (NUM_WINDOWS - 1) or (not self.before_l_detected and not self.before_r_detected):
                # 첫 윈도우 또는 이전 검출 실패시 전체 영역 탐색
                leftx_current, rightx_current = self._detect_initial_lanes(histogram, mid_point)
            else:
                # 이전 검출 결과 기반으로 지역 탐색
                leftx_current, rightx_current = self._detect_continuous_lanes(
                    histogram, lx, rx, binary_img.shape[1])

            # 검출 결과 시각화 및 저장
            self._update_and_visualize(
                leftx_current, rightx_current,
                roi_y_low, roi_y_high,
                out_img, frame_num, window_idx,
                l_pos, r_pos, lx, rx
            )

        return l_pos, r_pos, out_img

    def _detect_initial_lanes(self, histogram, mid_point):
        """
        전체 영역에서 초기 차선 위치 검출

        Args:
            histogram (ndarray): 현재 윈도우의 히스토그램
            mid_point (int): 이미지 중앙점

        Returns:
            tuple: (leftx_current, rightx_current) 좌/우 차선 x좌표
        """
        # 좌측 영역 검출
        left_histogram = histogram[:mid_point]
        left_max = np.argmax(left_histogram)
        leftx_current = left_max if np.max(left_histogram) > MIN_PIXELS * 255 else None

        # 우측 영역 검출
        right_histogram = histogram[mid_point:]
        right_max = np.argmax(right_histogram) + mid_point
        rightx_current = right_max if np.max(right_histogram) > MIN_PIXELS * 255 else None

        return leftx_current, rightx_current

    def _detect_continuous_lanes(self, histogram, lx, rx, img_width):
        """
        이전 검출 결과를 기반으로 지역 검색

        Args:
            histogram (ndarray): 현재 윈도우의 히스토그램
            lx (list): 이전 좌측 차선 x좌표 목록
            rx (list): 이전 우측 차선 x좌표 목록
            img_width (int): 이미지 너비

        Returns:
            tuple: (leftx_current, rightx_current) 좌/우 차선 x좌표
        """
        leftx_current = rightx_current = None

        # 좌측 차선 검출
        if self.before_l_detected and lx:
            leftx_last = lx[-1]
            left_start = max(0, leftx_last - WINDOW_MARGIN)
            left_end = min(img_width, leftx_last + WINDOW_MARGIN)
            leftx_current = self._calculate_window_center(histogram, left_start, left_end)

        # 우측 차선 검출
        if self.before_r_detected and rx:
            rightx_last = rx[-1]
            right_start = max(0, rightx_last - WINDOW_MARGIN)
            right_end = min(img_width, rightx_last + WINDOW_MARGIN)
            rightx_current = self._calculate_window_center(histogram, right_start, right_end)

        return leftx_current, rightx_current

    def _update_and_visualize(self, leftx_current, rightx_current, 
                            roi_y_low, roi_y_high, out_img, frame_num, 
                            window_idx, l_pos, r_pos, lx, rx):
        """
        검출 결과 업데이트 및 시각화

        Args:
            leftx_current (int): 현재 좌측 차선 x좌표
            rightx_current (int): 현재 우측 차선 x좌표
            roi_y_low (int): ROI 하단 y좌표
            roi_y_high (int): ROI 상단 y좌표
            out_img (ndarray): 출력 이미지
            frame_num (int): 현재 프레임 번호
            window_idx (int): 현재 윈도우 인덱스
            l_pos (list): 좌측 차선 위치 목록
            r_pos (list): 우측 차선 위치 목록
            lx (list): 좌측 차선 x좌표 목록
            rx (list): 우측 차선 x좌표 목록
        """
        # 좌측 차선 업데이트
        if leftx_current is not None:
            win_xll = leftx_current - WINDOW_WIDTH
            win_xlh = leftx_current + WINDOW_WIDTH
            cv2.rectangle(out_img, (win_xll, roi_y_low), 
                        (win_xlh, roi_y_high), (0, 0, 255), 2)
            
            if frame_num % 10 == 0 and window_idx == NUM_WINDOWS - 6:
                l_pos.append(((win_xll + win_xlh) // 2, roi_y_high))
            lx.append(leftx_current)
            self.before_l_detected = True
        else:
            self.before_l_detected = False

        # 우측 차선 업데이트
        if rightx_current is not None:
            win_xrl = rightx_current - WINDOW_WIDTH
            win_xrh = rightx_current + WINDOW_WIDTH
            cv2.rectangle(out_img, (win_xrl, roi_y_low),
                        (win_xrh, roi_y_high), (0, 255, 0), 2)
            
            if frame_num % 10 == 0 and window_idx == NUM_WINDOWS - 6:
                r_pos.append(((win_xrl + win_xrh) // 2, roi_y_high))
            rx.append(rightx_current)
            self.before_r_detected = True
        else:
            self.before_r_detected = False

    def _calculate_window_center(self, histogram, start_idx, end_idx):
        """
        윈도우 내 차선 중심점 계산

        Args:
            histogram (ndarray): 히스토그램
            start_idx (int): 시작 인덱스
            end_idx (int): 종료 인덱스

        Returns:
            int or None: 차선 중심점 x좌표 또는 None(검출 실패시)
        """
        window_area = histogram[start_idx:end_idx]
        nonzero_indices = np.nonzero(window_area)[0]
        if len(nonzero_indices) > MIN_PIXELS:
            return int(np.mean(nonzero_indices)) + start_idx
        return None