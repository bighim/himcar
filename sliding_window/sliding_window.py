import cv2
import numpy as np

# 이미지 크기 정의 (가로, 세로)
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
HALF_WIDTH = 320
HALF_HEIGHT = 240

image_size = (IMAGE_WIDTH, IMAGE_HEIGHT)

class LaneDetectionParams:
    """
    Sliding Window 파라미터 객체
    """
    def __init__(self, num_SlidingWindow, width_sliding_window, window_margin, min_points, min_pixels, lane_width, mid_point):
        self.num_SlidingWindow = num_SlidingWindow
        self.width_sliding_window = width_sliding_window
        self.window_margin = window_margin
        self.min_points = min_points
        self.min_pixels = min_pixels
        self.lane_width = lane_width
        self.mid_point = mid_point
        # 각 반복에서 업데이트되는 속성
        self.histogram = None
        self.roi_y_low = None
        self.roi_y_high = None
        self.direction = None
        self.frame = None
        self.i = None

class SlidingWindow:
    def __init__(self):
        # 차선 좌표 초기화
        self.l_pos = []
        self.r_pos = []

        # 원근 변환 행렬 초기화
        self.warp_src_mtx = None
        self.warp_dist_mtx = None
        self.src_to_dist_mtx = None
        self.dist_to_src_mtx = None

        # 기본 파라미터 객체 초기화
        self.params = self.initialize_params()

        # 원근 변환 행렬 계산
        self.calculate_perspective_transform()
        
    def initialize_params(self):
        """
        기본 Sliding Window 파라미터 객체를 초기화합니다.
        """
        num_SlidingWindow = 18
        width_sliding_window = 20
        window_margin = 80
        min_points = (HALF_HEIGHT // num_SlidingWindow) // 4
        min_pixels = 10
        lane_width = 200
        mid_point = IMAGE_WIDTH // 2

        return LaneDetectionParams(
            num_SlidingWindow=num_SlidingWindow,
            width_sliding_window=width_sliding_window,
            window_margin=window_margin,
            min_points=min_points,
            min_pixels=min_pixels,
            lane_width=lane_width,
            mid_point=mid_point
        )
        
    def contrast_clihe(self, image):
        # 명암비 향상을 위한 CLAHE 적용
        clahe = cv2.createCLAHE(clipLimit=0.2, tileGridSize=(8, 8))
        ycrcb_mat = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        # 튜플을 리스트로 변환하여 요소 변경 가능하도록 함
        ycrcb_planes = list(cv2.split(ycrcb_mat))
        ycrcb_planes[0] = clahe.apply(ycrcb_planes[0])

        ycrcb_mat = cv2.merge(ycrcb_planes)
        clahe_image = cv2.cvtColor(ycrcb_mat, cv2.COLOR_YCrCb2BGR)

        return clahe_image
    
    def warp_image(self, image):
        # 이미지 워핑
        warped_image = cv2.warpPerspective(image, self.src_to_dist_mtx, (HALF_WIDTH, HALF_HEIGHT), flags=cv2.INTER_LINEAR)

        # 워핑 기준점 확인 (원 그리기)
        for point in self.warp_src_mtx:
            cv2.circle(image, tuple(map(int, point)), 5, (255, 0, 0), -1)

        return warped_image

    def binary_image_with_adaptivethreshold(self, image):
        # Apply HLS Thresholding
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        img_hls = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
        lower_white = np.array([0, 200, 0])  # Adjusted HLS threshold
        upper_white = np.array([255, 255, 255])
        hls_mask = cv2.inRange(img_hls, lower_white, upper_white)

        # Apply Otsu's Thresholding
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Combine both masks using AND operation
        combined_mask = cv2.bitwise_and(hls_mask, otsu_mask)

        return combined_mask
    
    def morphological_transformation(self, image):
        # 형태학적 변환 (모폴로지 닫힘 연산)을 통한 노이즈 제거
        kernel = np.ones((6, 6), np.uint8)
        morphological_transformation_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        return morphological_transformation_image

    def draw_sliding_window(self, image, frame):
        """
        Sliding Window를 사용하여 차선을 검출하고 이미지를 반환합니다.
        """
        lx, rx = [], []
        before_l_detected, before_r_detected = True, True
        out_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Sliding Window를 한 번 수행
        for i in range(self.params.num_SlidingWindow - 1, -1, -1):
            # 현재 창의 영역 계산
            roi_y_low, roi_y_high = self.get_window_bounds(i, image.shape[0], self.params.num_SlidingWindow)

            # 히스토그램 계산
            window = image[roi_y_low:roi_y_high, :]
            histogram = cv2.reduce(window, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S).flatten()
            
            if histogram.size == 0:
                print(f"Warning: Empty histogram at window index {i}. Skipping this window.")
                continue

            # 파라미터 업데이트
            self.params.histogram = histogram
            self.params.roi_y_low = roi_y_low
            self.params.roi_y_high = roi_y_high
            self.params.i = i
            self.params.frame = frame

            # 초기 차선 위치 계산
            if i == (self.params.num_SlidingWindow - 1):
                leftx_current, rightx_current = self.initialize_lane_positions(histogram, self.params.mid_point, self.params.min_points)
            else:
                leftx_current, rightx_current = self.update_lane_positions(
                    histogram, lx, rx, self.params.window_margin, self.params.lane_width,
                    self.params.min_points, before_l_detected, before_r_detected
                )

            # 오른쪽 차선 처리
            self.params.direction = "right"
            rightx_current, before_r_detected = self.process_lane(
                out_img, rightx_current, rx, self.r_pos, before_r_detected, self.params
            )

            # 왼쪽 차선 처리
            self.params.direction = "left"
            leftx_current, before_l_detected = self.process_lane(
                out_img, leftx_current, lx, self.l_pos, before_l_detected, self.params
            )

        return out_img

    def get_window_bounds(self, i, img_height, num_SlidingWindow):
        """
        주어진 슬라이딩 윈도우 인덱스에서 Y 좌표의 범위를 계산.
        """
        roi_y_low = int(i * img_height / num_SlidingWindow)
        roi_y_high = int((i + 1) * img_height / num_SlidingWindow)
        return roi_y_low, roi_y_high


    def initialize_lane_positions(self, histogram, mid_point, min_points):
        """
        초기 창에서 히스토그램의 최대값을 기반으로 왼쪽/오른쪽 차선의 초기 위치를 설정.
        """
        left_histogram = histogram[0:mid_point]
        right_histogram = histogram[mid_point:]

         # 빈 배열 처리
        if left_histogram.size == 0:
            leftx_current = None
        else:
            left_max_val = np.max(left_histogram)
            leftx_current = np.argmax(left_histogram) if left_max_val > 255 * min_points else None

        if right_histogram.size == 0:
            rightx_current = None
        else:
            right_max_val = np.max(right_histogram)
            rightx_current = np.argmax(right_histogram) + mid_point if right_max_val > 255 * min_points else None

        return leftx_current, rightx_current


    def update_lane_positions(self, histogram, lx, rx, window_margin, lane_width, min_points, before_l_detected, before_r_detected):
        """
        이전 창의 결과를 기반으로 차선의 위치를 갱신.
        """
        leftx_current, rightx_current = None, None

        if before_l_detected and before_r_detected:
            leftx_current = self.find_lane(histogram, lx[-1], window_margin, min_points)
            rightx_current = self.find_lane(histogram, rx[-1], window_margin, min_points)
        elif not before_l_detected and before_r_detected:
            rightx_current = self.find_lane(histogram, rx[-1], window_margin, min_points)
            if rightx_current is not None:
                leftx_current = self.find_opposite_lane(histogram, rightx_current, lane_width, "left", min_points)
        elif before_l_detected and not before_r_detected:
            leftx_current = self.find_lane(histogram, lx[-1], window_margin, min_points)
            if leftx_current is not None:
                rightx_current = self.find_opposite_lane(histogram, leftx_current, lane_width, "right", min_points)

        return leftx_current, rightx_current


    def find_lane(self, histogram, last_pos, margin, min_points):
        """
        특정 히스토그램 영역에서 최대값을 찾아 차선의 중심 위치를 반환.
        """
        start = max(0, last_pos - margin)
        end = min(len(histogram) - 1, last_pos + margin)
        sub_histogram = histogram[start:end]

        max_val = np.max(sub_histogram)
        max_loc = np.argmax(sub_histogram) + start if max_val > 255 * min_points else None

        return max_loc


    def find_opposite_lane(self, histogram, ref_pos, lane_width, direction, min_points):
        """
        반대쪽 차선을 찾음.
        """
        if direction == "left" and ref_pos - lane_width > 0:
            search_histogram = histogram[0:ref_pos - lane_width]
        elif direction == "right" and ref_pos + lane_width < len(histogram):
            search_histogram = histogram[ref_pos + lane_width:]
        else:
            return None

        max_val = np.max(search_histogram)
        max_loc = np.argmax(search_histogram) if max_val > 255 * min_points else None

        if direction == "right" and max_loc is not None:
            max_loc += ref_pos + lane_width

        return max_loc


    def process_lane(self, out_img, x_current, x_list, pos_list, detected, params):
        """
        차선을 검출하고 시각화.
        """
        x_current = self.calculate_lane_position(params.histogram, x_current, params.width_sliding_window, params.min_pixels)
        if x_current is not None:
            self.draw_lane(out_img, x_current, params.roi_y_low, params.roi_y_high, params.width_sliding_window,
                        (0, 255, 0) if params.direction == "right" else (0, 0, 255))
            detected = True
            self.save_lane_position(params.frame, params.i, params.num_SlidingWindow, x_current, params.width_sliding_window,
                                params.roi_y_high, pos_list)
            x_list.append(x_current)
        else:
            detected = False

        return x_current, detected
    
    def calculate_lane_position(self, histogram, x_current, width_sliding_window, min_pixels):
        """
        히스토그램에서 차선 픽셀 평균 계산.
        """
        if x_current is None:
            return None
        start = max(0, x_current - width_sliding_window)
        end = min(len(histogram) - 1, x_current + width_sliding_window)
        nonzeros = np.nonzero(histogram[start:end])[0]
        nonzeros = [x + start for x in nonzeros]
        if len(nonzeros) > min_pixels:
            return int(np.mean(nonzeros))
        return None


    def draw_lane(self, out_img, x_current, roi_y_low, roi_y_high, width_sliding_window, color):
        """
        차선 창을 그립니다.
        """
        if x_current is not None:
            cv2.rectangle(out_img, (x_current - width_sliding_window, roi_y_low),
                        (x_current + width_sliding_window, roi_y_high), color, 2)


    def save_lane_position(self, frame, i, num_SlidingWindow, x_current, width_sliding_window, roi_y_high, pos_list):
        """
        특정 프레임에서 차선 좌표 저장.
        """
        if frame % 10 == 0 and i == num_SlidingWindow - 6:
            pos_list.append(((x_current - width_sliding_window + x_current + width_sliding_window) // 2, roi_y_high))
            
    def calculate_perspective_transform(self):
        """
        원근 변환 행렬을 계산합니다.
        """
        warp_image_width = HALF_WIDTH
        warp_image_height = HALF_HEIGHT

        warp_x_margin = 180
        warp_y_margin = 3

        src_pts1 = (240 - warp_x_margin, HALF_HEIGHT + 60)
        src_pts2 = (0, IMAGE_HEIGHT - 40)
        src_pts3 = (400 + warp_x_margin, HALF_HEIGHT + 60)
        src_pts4 = (IMAGE_WIDTH, IMAGE_HEIGHT - 40)

        dist_pts1 = (0, 0)
        dist_pts2 = (0, warp_image_height)
        dist_pts3 = (warp_image_width, 0)
        dist_pts4 = (warp_image_width, warp_image_height)

        self.warp_src_mtx = np.float32([src_pts1, src_pts2, src_pts3, src_pts4])
        self.warp_dist_mtx = np.float32([dist_pts1, dist_pts2, dist_pts3, dist_pts4])

        self.src_to_dist_mtx = cv2.getPerspectiveTransform(self.warp_src_mtx, self.warp_dist_mtx)
        self.dist_to_src_mtx = cv2.getPerspectiveTransform(self.warp_dist_mtx, self.warp_src_mtx)
            
    def warp_point(self, points):
        # 포인트를 원근 변환하여 실제 좌표로 변환
        if not points:
            return []

        points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        warped_points = cv2.perspectiveTransform(points, self.dist_to_src_mtx)
        warped_points = warped_points.reshape(-1, 2).tolist()

        return warped_points