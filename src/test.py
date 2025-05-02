from ultralytics import YOLO
from PIL import Image

# 모델 로드 (YOLOv5 또는 YOLOv8 사용 가능)
model = YOLO('yolov5s.pt')  # 또는 'yolov8n.pt'

# 이미지 객체 검출
results = model('city.jpg')  # 경로 또는 numpy 배열 입력 가능
results[0].show()  # PIL 이미지 윈도우 출력

for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        print(f"Class: {model.names[cls_id]}, Conf: {conf:.2f}, Box: {xyxy}")