# config.py
"""
슬라이딩 윈도우 차선 검출에 필요한 설정값 정의
"""

# 이미지 크기 관련 설정
IMAGE_WIDTH = 640  # 입력 이미지 너비
IMAGE_HEIGHT = 480  # 입력 이미지 높이
HALF_WIDTH = 320  # 워핑된 이미지 너비
HALF_HEIGHT = 240  # 워핑된 이미지 높이

# 슬라이딩 윈도우 파라미터
NUM_WINDOWS = 18  # 윈도우 개수
WINDOW_WIDTH = 20  # 윈도우 너비
WINDOW_MARGIN = 80  # 윈도우 탐색 마진
MIN_PIXELS = 10  # 최소 픽셀 수
LANE_WIDTH = 200  # 예상 차선 너비

# 디버그 이미지 저장 경로
OUTPUT_DIR = "frames"