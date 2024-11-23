from flask import Flask
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import json

app = Flask(__name__)
socketio = SocketIO(app)

# 연결된 클라이언트 관리
connected_clients = set()

@socketio.on('connect')
def handle_connect():
    """클라이언트 연결 처리"""

@socketio.on('disconnect')
def handle_disconnect():
    """클라이언트 연결 해제 처리"""

@socketio.on('frame')
def handle_frame(data):
    """프레임 데이터 처리"""
    try:
        # data가 이미 딕셔너리 형태로 전달되므로 json.loads() 불필요
        encoded_frame = data['frame']
        distance = data.get('distance', None)

        # Base64 디코딩
        jpg_data = base64.b64decode(encoded_frame)
        nparr = np.frombuffer(jpg_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is not None:
            # 이미지 중앙에 빨간 점 그리기
            height, width = frame.shape[:2]
            center_x = width // 2
            center_y = height // 2
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # 빨간색 점

            # 거리 정보가 있다면 표시
            if distance is not None:
                cv2.putText(frame, f"Distance: {distance:.2f}m", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)

            # 디버그용 이미지 저장
            cv2.imwrite('debug_frame.jpg', frame)
            
            print(f"프레임 처리 완료 - 거리: {distance}m")

    except Exception as e:
        print(f"프레임 처리 중 오류 발생: {e}")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)