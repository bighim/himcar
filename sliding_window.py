import cv2
import numpy as np
import os

# 이미지 크기 정의 (가로, 세로)
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
HALF_WIDTH = 320
HALF_HEIGHT = 240

image_size = (IMAGE_WIDTH, IMAGE_HEIGHT)

# 이미지 저장 경로 설정
save_dir = "frames"
os.makedirs(save_dir, exist_ok=True)

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
        
        # 슬라이딩 윈도우 관련 변수들을 인스턴스 변수로 선언
        self.num_SlidingWindow = 18
        self.width_sliding_window = 20
        self.lx = []
        self.rx = []

    def calculate_perspective_transform(self):
        # 워핑 관련 변수 설정
        warp_image_width = HALF_WIDTH
        warp_image_height = HALF_HEIGHT

        warp_x_margin = 200
        warp_y_margin = 3

        src_pts1 = (240 - warp_x_margin, HALF_HEIGHT + 60)
        src_pts2 = (0, IMAGE_HEIGHT + 20)
        src_pts3 = (400 + warp_x_margin, HALF_HEIGHT + 60)
        src_pts4 = (IMAGE_WIDTH, IMAGE_HEIGHT + 20)

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

        return hls_mask

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

    def draw_sliding_window(self, image, frame_count):
        # num_SlidingWindow를 지역 변수 대신 self.num_SlidingWindow 사용
        window_height = HALF_HEIGHT // self.num_SlidingWindow
        window_margin = 80
        
        # lx, rx 초기화
        self.lx = []
        self.rx = []
        
        min_points = window_height // 4
        min_pixels = 10

        lane_width = 200

        before_l_detected = True
        before_r_detected = True

        out_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        for i in range(self.num_SlidingWindow - 1, -1, -1):
            leftx_current = None
            rightx_current = None

            # 윈도우 생성
            roi_y_low = int(i * image.shape[0] / self.num_SlidingWindow)
            roi_y_high = int((i + 1) * image.shape[0] / self.num_SlidingWindow)
            window = image[roi_y_low:roi_y_high, :]

            # 히스토그램 생성
            histogram = cv2.reduce(window, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
            histogram = histogram.flatten()

            # 이미지의 중간점
            mid_point = image.shape[1] // 2

            # 히스토그램 범위 초기화
            right_histogram_start = None
            left_histogram_start = None

            if i == (self.num_SlidingWindow - 1):
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
                leftx_last = self.lx[-1]
                rightx_last = self.rx[-1]

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
                rightx_last = self.rx[-1]
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
                leftx_last = self.lx[-1]
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

                    win_xrl = rightx_current - self.width_sliding_window
                    win_xrh = rightx_current + self.width_sliding_window

                    cv2.rectangle(out_img, (win_xrl, roi_y_low), (win_xrh, roi_y_high), (0, 255, 0), 2)
                    before_r_detected = True

                    if frame_count is not None and frame_count % 10 == 0 and i == self.num_SlidingWindow - 6:
                        self.r_pos.append(((win_xrl + win_xrh) // 2, roi_y_high))
                else:
                    before_r_detected = False

                self.rx.append(rightx_current)
            else:
                before_r_detected = False

            # 왼쪽 차선 처리
            if leftx_current is not None:
                left_nz = np.nonzero(histogram[left_histogram_start:left_histogram_start + len(left_histogram)])[0]
                left_nonzeros = [x + left_histogram_start for x in left_nz]

                if len(left_nonzeros) > min_pixels:
                    leftx_current = int(np.mean(left_nonzeros))

                    win_xll = leftx_current - self.width_sliding_window
                    win_xlh = leftx_current + self.width_sliding_window

                    cv2.rectangle(out_img, (win_xll, roi_y_low), (win_xlh, roi_y_high), (0, 0, 255), 2)
                    before_l_detected = True

                    if frame_count is not None and frame_count % 10 == 0 and i == self.num_SlidingWindow - 6:
                        self.l_pos.append(((win_xll + win_xlh) // 2, roi_y_high))
                else:
                    before_l_detected = False

                self.lx.append(leftx_current)
            else:
                before_l_detected = False

        return out_img

    def fit_polynomial(self, image, frame_count):
        out_img = self.draw_sliding_window(image, frame_count)
        cv2.rectangle(out_img, (5, 5), (315, 50), (0, 0, 0), -1)
        
        try:
            left_fit = None
            if len(self.lx) > 0:
                # y 좌표를 반대로 생성 (아래에서 위로)
                y_points = np.linspace(image.shape[0]-1, 0, num=len(self.lx))
                x_points = np.array(self.lx)
                
                left_fit = np.polyfit(y_points, x_points, 2)
                
                # 곡선 그리기용 y 좌표도 반대로
                plot_y = np.linspace(image.shape[0]-1, 0, num=image.shape[0])
                left_fitx = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
                
                pts_left = np.array([[int(x), int(y)] for x, y in zip(left_fitx, plot_y)])
                cv2.polylines(out_img, [pts_left], isClosed=False, color=(255, 0, 0), thickness=2)
            
            # 오른쪽 차선 피팅
            right_fit = None
            if len(self.rx) > 0:
                # y 좌표를 반대로 생성 (아래에서 위로)
                y_points = np.linspace(image.shape[0]-1, 0, num=len(self.rx))
                x_points = np.array(self.rx)
                
                right_fit = np.polyfit(y_points, x_points, 2)
                
                plot_y = np.linspace(image.shape[0]-1, 0, num=image.shape[0])
                right_fitx = right_fit[0]*plot_y**2 + right_fit[1]*plot_y + right_fit[2]
                
                pts_right = np.array([[int(x), int(y)] for x, y in zip(right_fitx, plot_y)])
                cv2.polylines(out_img, [pts_right], isClosed=False, color=(0, 0, 255), thickness=2)
            
            # 텍스트 표시 (None 포함)
            left_text = f"Left: {'None' if left_fit is None else f'{left_fit[0]:.6f}x^2 + {left_fit[1]:.6f}x + {left_fit[2]:.6f}'}"
            right_text = f"Right: {'None' if right_fit is None else f'{right_fit[0]:.6f}x^2 + {right_fit[1]:.6f}x + {right_fit[2]:.6f}'}"
            
            # 텍스트 그리기
            cv2.putText(out_img, left_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.4, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(out_img, left_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.4, (255, 0, 0), 1, cv2.LINE_AA)
            
            cv2.putText(out_img, right_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.4, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(out_img, right_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.4, (0, 0, 255), 1, cv2.LINE_AA)
            
            return out_img, left_fit, right_fit
                
        except Exception as e:
            print(f"Error in fit_polynomial: {e}")
            return out_img, None, None