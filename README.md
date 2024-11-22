# 차량 2대 이상 주행 테스트

서버는 app.py, 차량은 multi_car.py
수정해서 사용.

```python
class CarConfig:

    # 서보모터 설정
    CAMERA_HORIZONTAL_DEFAULT = None
    CAMERA_VERTICAL_DEFAULT = None
    SERVO_STEERING_DEFAULT = None

    # 서버 설정
    SERVER_URL = 'http://192.168.0.x:5000'

    # 차량 번호
    CAR_NUMBER = None
```
