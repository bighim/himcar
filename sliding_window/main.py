# main.py
from flask import Flask
from flask_socketio import SocketIO
import cv2
import numpy as np
import os
import traceback
from sliding_window import SlidingWindow

app = Flask(__name__)
socketio = SocketIO(app)
window = SlidingWindow()
frame_number = 0

@socketio.on('frame')
def handle_frame(data):
    global frame_number
    try:
        frame_data = data['frame']
        frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        
        lpos, rpos = window.process_frame(frame, frame_number)
        if lpos and rpos:
            print(f"Left Positions: {lpos}")
            print(f"Right Positions: {rpos}")
            
        frame_number += 1
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        print(traceback.format_exc())

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)