import cv2
import numpy as np

# 이미지 크기 정의 (가로, 세로)
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
HALF_WIDTH = 320
HALF_HEIGHT = 240

image_size = (IMAGE_WIDTH, IMAGE_HEIGHT)

# SlidingWindow 클래스 정의
class SlidingWindow:
    def __init__(self):
        # l_pos와 r_pos를 인스턴스 변수로 초기화
        self.l_pos = []
        self.r_pos = []
        # 원근 변환에 필요한 행렬을 미리 계산하여 저장
        self.warp_src_mtx = None
        self.warp_dist_mtx = None
        self.src_to_dist_mtx = None
        self.dist_to_src_mtx = None
        self.calculate_perspective_transform()

    def calculate_perspective_transform(self):
        # 워핑 관련 변수 설정
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

    def calibrate_image(self, src, map1, map2, roi):
        # 이미지 보정
        mapping_image = cv2.remap(src, map1, map2, cv2.INTER_LINEAR)
        x, y, w, h = roi  # roi는 (x, y, w, h) 형태의 튜플
        mapping_image = mapping_image[y:y+h, x:x+w]
        calibrated_image = cv2.resize(mapping_image, image_size)
        return calibrated_image

    def warp_image(self, image):
        # 이미지 워핑
        warped_image = cv2.warpPerspective(image, self.src_to_dist_mtx, (HALF_WIDTH, HALF_HEIGHT), flags=cv2.INTER_LINEAR)

        # 워핑 기준점 확인 (원 그리기)
        for point in self.warp_src_mtx:
            cv2.circle(image, tuple(map(int, point)), 5, (255, 0, 0), -1)

        return warped_image

    def warp_point(self, points):
        # 포인트를 원근 변환하여 실제 좌표로 변환
        if not points:
            return []

        points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        warped_points = cv2.perspectiveTransform(points, self.dist_to_src_mtx)
        warped_points = warped_points.reshape(-1, 2).tolist()

        return warped_points

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

    def morphological_transformation(self, image):
        # 형태학적 변환 (모폴로지 닫힘 연산)을 통한 노이즈 제거
        kernel = np.ones((6, 6), np.uint8)
        morphological_transformation_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        return morphological_transformation_image

    def draw_sliding_window(self, image, frame):
        # 전역 변수 대신 인스턴스 변수 사용
        num_SlidingWindow = 18
        width_sliding_window = 20

        window_height = HALF_HEIGHT // num_SlidingWindow
        window_margin = 80

        min_points = window_height // 4
        min_pixels = 10

        lane_width = 200

        lx = []
        rx = []

        before_l_detected = True
        before_r_detected = True

        out_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        for i in range(num_SlidingWindow - 1, -1, -1):
            leftx_current = None
            rightx_current = None

            # 윈도우 생성
            roi_y_low = int(i * image.shape[0] / num_SlidingWindow)
            roi_y_high = int((i + 1) * image.shape[0] / num_SlidingWindow)
            window = image[roi_y_low:roi_y_high, :]

            # 히스토그램 생성
            histogram = cv2.reduce(window, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
            histogram = histogram.flatten()

            # 이미지의 중간점
            mid_point = image.shape[1] // 2

            # 히스토그램 범위 초기화
            right_histogram_start = None
            left_histogram_start = None

            if i == (num_SlidingWindow - 1):
                left_histogram = histogram[0:mid_point]
                right_histogram = histogram[mid_point:]

                left_max_val = np.max(left_histogram)
                left_max_loc = np.argmax(left_histogram)
                if left_max_val > 255 * min_points:
                    leftx_current = left_max_loc
                else:
                    leftx_current = None

                right_max_val = np.max(right_histogram)
                right_max_loc = np.argmax(right_histogram) + mid_point
                if right_max_val > 255 * min_points:
                    rightx_current = right_max_loc
                else:
                    rightx_current = None

                right_histogram_start = mid_point
                left_histogram_start = 0

            elif before_l_detected and before_r_detected:
                leftx_last = lx[-1]
                rightx_last = rx[-1]

                left_start = max(0, leftx_last - window_margin)
                left_end = min(histogram.shape[0] - 1, leftx_last + window_margin)
                left_histogram = histogram[left_start:left_end]

                right_start = max(0, rightx_last - window_margin)
                right_end = min(histogram.shape[0] - 1, rightx_last + window_margin)
                right_histogram = histogram[right_start:right_end]

                left_max_val = np.max(left_histogram)
                left_max_loc = np.argmax(left_histogram) + left_start
                if left_max_val > 255 * min_points:
                    leftx_current = left_max_loc
                else:
                    leftx_current = None

                right_max_val = np.max(right_histogram)
                right_max_loc = np.argmax(right_histogram) + right_start
                if right_max_val > 255 * min_points:
                    rightx_current = right_max_loc
                else:
                    rightx_current = None

                right_histogram_start = right_start
                left_histogram_start = left_start

            elif not before_l_detected and before_r_detected:
                rightx_last = rx[-1]
                right_start = max(0, rightx_last - window_margin)
                right_end = min(histogram.shape[0] - 1, rightx_last + window_margin)
                right_histogram = histogram[right_start:right_end]

                right_max_val = np.max(right_histogram)
                right_max_loc = np.argmax(right_histogram) + right_start
                if right_max_val > 255 * min_points:
                    rightx_current = right_max_loc
                else:
                    rightx_current = None

                if rightx_last - lane_width > 0:
                    left_histogram = histogram[0:rightx_last - lane_width]
                    if len(left_histogram) > 0:
                        left_max_val = np.max(left_histogram)
                        left_max_loc = np.argmax(left_histogram)
                        if left_max_val > 255 * min_points:
                            leftx_current = left_max_loc
                        else:
                            leftx_current = None
                    else:
                        leftx_current = None
                else:
                    leftx_current = None

                right_histogram_start = right_start
                left_histogram_start = 0

            elif before_l_detected and not before_r_detected:
                leftx_last = lx[-1]
                left_start = max(0, leftx_last - window_margin)
                left_end = min(histogram.shape[0] - 1, leftx_last + window_margin)
                left_histogram = histogram[left_start:left_end]

                left_max_val = np.max(left_histogram)
                left_max_loc = np.argmax(left_histogram) + left_start
                if left_max_val > 255 * min_points:
                    leftx_current = left_max_loc
                else:
                    leftx_current = None

                if leftx_last + lane_width < histogram.shape[0]:
                    right_histogram = histogram[leftx_last + lane_width:]
                    if len(right_histogram) > 0:
                        right_max_val = np.max(right_histogram)
                        right_max_loc = np.argmax(right_histogram) + leftx_last + lane_width
                        if right_max_val > 255 * min_points:
                            rightx_current = right_max_loc
                        else:
                            rightx_current = None
                    else:
                        rightx_current = None
                else:
                    rightx_current = None

                right_histogram_start = leftx_last + lane_width
                left_histogram_start = left_start

            else:
                left_histogram = histogram[0:mid_point]
                right_histogram = histogram[mid_point:]

                left_max_val = np.max(left_histogram)
                left_max_loc = np.argmax(left_histogram)
                if left_max_val > 255 * min_points:
                    leftx_current = left_max_loc
                else:
                    leftx_current = None

                right_max_val = np.max(right_histogram)
                right_max_loc = np.argmax(right_histogram) + mid_point
                if right_max_val > 255 * min_points:
                    rightx_current = right_max_loc
                else:
                    rightx_current = None

                right_histogram_start = mid_point
                left_histogram_start = 0

            win_yl = int((i + 1) * window_height)
            win_yh = int(i * window_height)

             # 오른쪽 차선 처리
            if rightx_current is not None:
                right_nz = np.nonzero(histogram[right_histogram_start:right_histogram_start + len(right_histogram)])[0]
                right_nonzeros = [x + right_histogram_start for x in right_nz]

                if len(right_nonzeros) > min_pixels:
                    rightx_current = int(np.mean(right_nonzeros))

                    win_xrl = rightx_current - width_sliding_window
                    win_xrh = rightx_current + width_sliding_window

                    cv2.rectangle(out_img, (win_xrl, roi_y_low), (win_xrh, roi_y_high), (0, 255, 0), 2)
                    before_r_detected = True

                    if frame % 10 == 0 and i == num_SlidingWindow - 6:
                        self.r_pos.append(((win_xrl + win_xrh) // 2, roi_y_high))
                else:
                    before_r_detected = False

                rx.append(rightx_current)
            else:
                before_r_detected = False

            # 왼쪽 차선 처리
            if leftx_current is not None:
                left_nz = np.nonzero(histogram[left_histogram_start:left_histogram_start + len(left_histogram)])[0]
                left_nonzeros = [x + left_histogram_start for x in left_nz]

                if len(left_nonzeros) > min_pixels:
                    leftx_current = int(np.mean(left_nonzeros))

                    win_xll = leftx_current - width_sliding_window
                    win_xlh = leftx_current + width_sliding_window

                    cv2.rectangle(out_img, (win_xll, roi_y_low), (win_xlh, roi_y_high), (0, 0, 255), 2)
                    before_l_detected = True

                    if frame % 10 == 0 and i == num_SlidingWindow - 6:
                        self.l_pos.append(((win_xll + win_xlh) // 2, roi_y_high))
                else:
                    before_l_detected = False

                lx.append(leftx_current)
            else:
                before_l_detected = False

        return out_img