#pip install ultralytics
import torch
import os
from ultralytics import YOLO

#!cd /home/panyue/AL/data
ROOT_DIR = "/home/yue/AL/data"
#!ls -lt $ROOT_DIR

model = YOLO('yolov8n.yaml')  # build a new model from YAML
#TOWARDSDS TUTORIAL CODES
#yolo task=detect mode=train model=yolov8n.pt data=config_file.yaml epochs=3 imgsz=800

results = model.train(data = os.path.join(ROOT_DIR, "config_file.yaml"), epochs = 100)
