import numpy as np
import time
from math import log2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io, color
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import random
import shutil
import cv2  
from ultralytics import YOLO
from PIL import Image
import torch
import yaml 
from sklearn.metrics.pairwise import euclidean_distances
import eco2ai
from skimage import io

ROOT_DIR = "./VisDrone"
OUTPUT_DIR = "./VisDrone/hybrid_seeds"

# Directories of each dataset
train_images_dir = os.path.join(ROOT_DIR, 'train', 'images')
val_images_dir = os.path.join(ROOT_DIR, 'val', 'images')
train_labels_dir = os.path.join(ROOT_DIR, 'train', 'labels')
val_labels_dir = os.path.join(ROOT_DIR, 'val', 'labels')
test_images_dir = os.path.join(ROOT_DIR, 'test', 'images')
test_labels_dir = os.path.join(ROOT_DIR, 'test', 'labels')

train_images_file = os.listdir(train_images_dir)
train_labels_file = os.listdir(train_labels_dir)
val_images_file = os.listdir(val_images_dir)
val_labels_file = os.listdir(val_labels_dir)
test_images_file = os.listdir(test_images_dir)
test_labels_file = os.listdir(test_labels_dir)


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print("Using device:", device)

percentage = 10  # Percentage for additional images


def count_images_in_folder(folder_path):
    image_files = [file for file in os.listdir(folder_path) if file.endswith('.jpg')]
    return len(image_files)

class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [file for file in os.listdir(image_dir) if file.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = os.path.join(self.image_dir, self.image_files[idx])
        image = io.imread(image_name)
        if self.transform: 
            image = self.transform(image)
        return image   

# Define the transformation you want to apply to your images                                
transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((1280, 1280)),
                                transforms.ToTensor()])

# Create a custom dataset for your data
dataset = CustomDataset(os.path.join(ROOT_DIR, 'train', 'images'), transform=transform)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Keep track of the carbon emissions for the model training
tracker = eco2ai.Tracker(
    project_name="Hybrid Carbon", 
    experiment_description="hybrid seed4 model",
    file_name="emission.csv"
    )

tracker.start()

# Unlabeled images
previous_seeds_images = []
previous_seeds_labels = []
seed_folders = ['seed3']  

for seed_name in seed_folders:
    seed_images = os.path.join(OUTPUT_DIR, seed_name, 'train', 'images')
    seed_labels = os.path.join(OUTPUT_DIR, seed_name, 'train', 'labels')
    previous_seeds_images.extend([os.path.join(seed_images, file) for file in os.listdir(seed_images)])
    previous_seeds_labels.extend([os.path.join(seed_labels, file) for file in os.listdir(seed_labels)])

# Unlabeled images and labels
unlabeled_images_train = [os.path.join(train_images_dir, file) for file in train_images_file if file not in previous_seeds_images]
unlabeled_labels_train = [os.path.join(train_labels_dir, file) for file in train_labels_file if file not in previous_seeds_labels]

unlabeled_images_val = [os.path.join(train_images_dir, file) for file in val_images_file if file not in previous_seeds_images]
unlabeled_labels_val = [os.path.join(train_labels_dir, file) for file in val_labels_file if file not in previous_seeds_labels]

seed3_path = os.path.join(OUTPUT_DIR, "seed3")
seed3_train_images_folder = os.path.join(seed3_path, 'train', 'images')
seed3_val_images_folder = os.path.join(seed3_path, 'val', 'images')
seed3_train_labels_folder = os.path.join(seed3_path, 'train', 'labels')
seed3_val_labels_folder = os.path.join(seed3_path, 'val', 'labels')


seed3_train_images = os.listdir(seed3_train_images_folder)
seed3_val_images = os.listdir(seed3_val_images_folder)

print(f"Seed3: Train images - {len(seed3_train_images)}, Val images - {len(seed3_val_images)}")

cumulative_train_images = set(seed3_train_images)
cumulative_val_images = set(seed3_val_images)

# List files in each directory
train_images_file = os.listdir(train_images_dir)
val_images_file = os.listdir(val_images_dir)



# Load YOLO model
model_path = os.path.join(OUTPUT_DIR, "seed3", 'best_seed3.pt')
model = YOLO(model_path)

# Calculate entropy scores for each image
image_entropy_scores = []
batch_size = 4
for i in range(0, len(train_images_file), batch_size):
    batch_unlabeled_images = [os.path.join(train_images_dir, train_images_file[j]) 
                              for j in range(i, min(i + batch_size, len(train_images_file)))]
    batch_results = model(batch_unlabeled_images, imgsz=1280, save=True, stream=True)

    for img_file, result in zip(batch_unlabeled_images, batch_results):
        confidences = [box.conf.cpu().item() for box in result.boxes]
        entropies = [-np.sum(conf * np.log2(conf + 1e-10)) for conf in confidences] if confidences else [np.nan]
        average_entropy = np.nanmean(entropies)
        image_entropy_scores.append((img_file, average_entropy))

    print(f'Processed {min(i + batch_size, len(train_images_file))} images')

# Sort the image file paths based on average entropy scores in descending order
sorted_entropy_scores = sorted(image_entropy_scores, key=lambda x: x[1], reverse=True)
# additional_images_train = [img for img, _ in sorted_entropy_scores[:int(len(train_images_file) * percentage / 100)]]
discrepancy = 650
additional_images_train = [img for img, _ in sorted_entropy_scores[:int(len(train_images_file) * percentage / 100 + discrepancy )]]

# Directly select additional validation images from unlabeled_images_val
# additional_images_val = random.sample(unlabeled_images_val, int(len(val_images_file) * percentage / 100))
additional_images_val = random.sample([os.path.join(val_images_dir, file) for file in val_images_file if file not in cumulative_val_images], int(len(val_images_file) * percentage / 100))
print("Additional val images paths:", additional_images_val)

print(f"Selected additional images: Train - {len(additional_images_train)}, Val - {len(additional_images_val)}")


# Create seed4
seed_name = 'seed4'
seed_folder = os.path.join(OUTPUT_DIR, seed_name)

if not os.path.exists(seed_folder):
    os.makedirs(seed_folder, exist_ok=True)

    # Create directories for train and val
    new_train_images_folder = os.path.join(seed_folder, 'train', 'images')
    new_train_labels_folder = os.path.join(seed_folder, 'train', 'labels')
    new_val_images_folder = os.path.join(seed_folder, 'val', 'images')
    new_val_labels_folder = os.path.join(seed_folder, 'val', 'labels')

    os.makedirs(new_train_images_folder, exist_ok=True)
    os.makedirs(new_train_labels_folder, exist_ok=True)
    os.makedirs(new_val_images_folder, exist_ok=True)
    os.makedirs(new_val_labels_folder, exist_ok=True)

    # Copy images and labels from the seed3
    for file in seed3_train_images:
        shutil.copy(os.path.join(seed3_train_images_folder, file), new_train_images_folder)
        shutil.copy(os.path.join(seed3_train_labels_folder, file.replace('.jpg', '.txt')), new_train_labels_folder)

    for file in seed3_val_images:
        shutil.copy(os.path.join(seed3_val_images_folder, file), new_val_images_folder)
        shutil.copy(os.path.join(seed3_val_labels_folder, file.replace('.jpg', '.txt')), new_val_labels_folder)
    
    print(f"After copying from the seed3 seed: Train images - {count_images_in_folder(new_train_images_folder)}, Val images - {count_images_in_folder(new_val_images_folder)}")
    print(f"Number of additional train images selected: {len(additional_images_train)}")
    print(f"Number of additional val images selected: {len(additional_images_val)}")

    # Copy newly selected additional images and their labels
    for img_file in additional_images_train:
        if os.path.exists(img_file):
            shutil.copy(img_file, new_train_images_folder)
            label_file = img_file.replace(train_images_dir, train_labels_dir).replace(".jpg", ".txt")
            shutil.copy(label_file, new_train_labels_folder)
        else:
            print(f"Warning: Image file not found: {img_file}")

    for img_file in additional_images_val:
        if os.path.exists(img_file):
            shutil.copy(img_file, new_val_images_folder)
            label_file = img_file.replace(val_images_dir, val_labels_dir).replace(".jpg", ".txt")
            shutil.copy(label_file, new_val_labels_folder)
        else:
            print(f"Warning: Image file not found: {img_file}")

    # Count again after copying additional images
    final_train_count_seed4 = count_images_in_folder(new_train_images_folder)
    final_val_count_seed4 = count_images_in_folder(new_val_images_folder)
    print(f"Final Seed 4: Train images - {final_train_count_seed4}, Val images - {final_val_count_seed4}")


# Create YAML file for seed4
yaml_dict = {
            "path": ROOT_DIR,
            "train": os.path.join("hybrid_seeds/", seed_name, "train/images"),
            "val": os.path.join("hybrid_seeds/", seed_name, "val/images"),
            "names": {
                0: "pedestrian",
                1: "people",
                2: "bicycle",
                3: "car",
                4: "van",
                5: "truck",
                6: "tricycle",
                7: "awning-tricycle",
                8: "bus",
                9: "motor"
            },
        }
yaml_file_path = os.path.join(seed_folder, f"{seed_name}.yaml")
with open(yaml_file_path, "w") as file:
    yaml.dump(yaml_dict, file)

# Train the YOLO model on seed4
print(f"Training YOLO model on {seed_folder}...")
results = model.train(data=yaml_file_path, epochs=50, patience=10, imgsz=1280, batch=4)

# Validate the model
metrics = model.val()
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category

# Inference on test data
start_time = time.time()
results = model(test_images_dir, imgsz=1280, batch=4, stream=True)
end_time = time.time()
duration = end_time - start_time
print(f"Inference Duration for {seed_name}: {duration} seconds")

# Stop the tracker and clear the CUDA cache
tracker.stop()
torch.cuda.empty_cache()