# train.py
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model to start from (yolov8n is small and fast)
model = YOLO('yolov8n.pt')

# Train the model using your 'data.yaml' file
if __name__ == '__main__':
    results = model.train(data='data.yaml', epochs=100, imgsz=640)