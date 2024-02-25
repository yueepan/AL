#！pip install ultralytics
import torch
import os
from ultralytics import YOLO
from torch.utils.data import Dataset, DataLoader
from time import time
import eco2ai

ROOT_DIR = "/home/yue/AL/VisDrone"
#!ls -lt $ROOT_DIR

# Start tracking
tracker = eco2ai.Tracker(
    project_name="Whole dataset Carbon", 
    experiment_description="training <Whole dataset, imgsz: 640> model",
    file_name="emission.csv"
)
tracker.start()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device}")

# Create YOLO model, adapted from Ultralytics YOLOv8 Docs
model = YOLO('yolov8n.pt').to(device)
# model = YOLO('yolov8n.pt')  

# # Check for available GPUs and define the device
# if torch.cuda.is_available():
#     device = 'cuda'
#     print(f"CUDA is available. Training on GPU.")
# else:
#     device = 'cpu'
#     print("CUDA is not available. Training on CPU.")

# Check if the model is running on GPU or CPU
if next(model.parameters()).is_cuda:
    print("Model is running on GPU.")
else:
    print("Model is running on CPU.")


# Different images sizes 
# results = model.train(data = os.path.join(ROOT_DIR, "VisDrone.yaml"), epochs = 100, patience = 10, imgsz= 2560, batch= 4, workers=2, save=True)
# results = model.train(data = os.path.join(ROOT_DIR, "VisDrone.yaml"), epochs = 100, patience = 10, imgsz= 1280, batch= 4, workers=2, save=True)
results = model.train(data = os.path.join(ROOT_DIR, "VisDrone.yaml"), epochs = 50, patience = 10, imgsz= 640, batch= 4, workers=2, save=True)
# steam= true for not exceeding NMS time

batch_size = 4
dataloader = DataLoader(dataset = os.path.join(ROOT_DIR, "VisDrone.yaml"), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category
print(metrics.box.map)    # map50-95
print(metrics.box.map50)  # map50

start = time.localtime()
#Training on the whole training and val datasets, infer on test dataset
model.predict('VisDrone/test/images', save=True, imgsz=6400)
end = time.localtime()

duration = end - start
print(f"Duration: {duration}")
f = open("duration.txt", "w")

for d in [duration]:
    f.write(f"{d}\n")

torch.save(model.state_dict(), os.path.join(ROOT_DIR, "weights", "yolo_v8.pth"))

tracker.stop()
