import onnxruntime
from yolov5.export import run as export_model
import os 
import pathlib

pathlib.PosixPath = pathlib.WindowsPath

model_path = "./best_final.pt"  # 학습된 가중치 파일

# YOLOv5 모델 로드 및 ONNX 변환
try:
    # ONNX 모델 파일 경로
    onnx_path = "./best_final.onnx"
    
    # PyTorch 모델을 ONNX로 변환 (처음 한 번만 실행)
    if not os.path.exists(onnx_path):
        print("PyTorch 모델을 ONNX로 변환 중...")
        export_model(
            weights=model_path,
            include=['onnx'],
            imgsz=[320, 320],
            device='cpu',
            simplify=True
        )
        print("ONNX 변환 완료")
    
    # ONNX 런타임 세션 생성
    providers = ['CPUExecutionProvider']
    session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    
    print("ONNX 모델이 성공적으로 로드되었습니다.")

except Exception as e:
    session = None
    print(f"모델 로드 중 오류 발생: {e}")