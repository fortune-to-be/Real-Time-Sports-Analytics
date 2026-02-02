import torch
from ultralytics import YOLO

# Load YOLOv5 model
model = YOLO('yolov5s.pt')  # Pretrained weights

# Dataset path (from Kaggle)
dataset_path = 'data/player_detection_dataset/'

# Train
model.train(
    data={
        'train': dataset_path + 'train/images',
        'val': dataset_path + 'val/images',
        'nc': 1,
        'names': ['player']
    },
    epochs=50,
    imgsz=640,
    batch=16
)

# Save trained model
model.export(format='torchscript', optimize=True)
model.save('models/player_detector.pt')
