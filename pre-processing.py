import os
import random
import shutil
import cv2  
import numpy as np
from math import log2
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm 
import torch

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

import os
from pathlib import Path
from tqdm import tqdm 
# from ultralytics.yolo.utils.downloads import download


ROOT_DIR = Path("./VisDrone")

def visdrone2yolo(ROOT_DIR):
    from PIL import Image
    from tqdm import tqdm

    def convert_box(size, box):
        # Convert VisDrone box to YOLO xywh box
        dw = 1. / size[0]
        dh = 1. / size[1]
        return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

    (ROOT_DIR / 'labels').mkdir(parents=True, exist_ok=True)  # make labels directory
    pbar = tqdm((ROOT_DIR / 'annotations').glob('*.txt'), desc=f'Converting {ROOT_DIR}')
    for f in pbar:
        img_path = ROOT_DIR / 'images' / (f.stem + '.jpg')
        if img_path.exists():
            img_size = Image.open(img_path).size
            lines = []
    # for f in pbar:
    #     img_size = Image.open((ROOT_DIR / 'images' / f.name).with_suffix('.jpg')).size
    #     lines = []
        with open(f, 'r') as file:  # read annotations.txt
            for row in [x.split(',') for x in file.read().strip().splitlines()]:
                if row[4] == '0':  # VisDrone 'ignored regions' class 0
                    continue
                cls = int(row[5]) - 1
                box = convert_box(img_size, tuple(map(int, row[:4])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
                with open(str(f).replace(f'{os.sep}annotations{os.sep}', f'{os.sep}labels{os.sep}'), 'w') as fl:
                # with open(str(f).replace(f'{os.sep}annotations{os.sep}', f'{os.sep}labels{os.sep}'), 'a') as fl:
                    fl.writelines(lines)  # write label.txt
            else:
                print(f"Image not found: {img_path}")

# Convert
for d in 'train', 'val', 'test':
      visdrone2yolo(ROOT_DIR / d)  # convert VisDrone annotations to YOLO labels

train_images_dir = os.path.join(ROOT_DIR, 'train', 'images')
val_images_dir = os.path.join(ROOT_DIR, 'val', 'images')
train_labels_dir = os.path.join(ROOT_DIR, 'train', 'labels')
val_labels_dir = os.path.join(ROOT_DIR, 'val', 'labels')
test_images_dir = os.path.join(ROOT_DIR, 'test', 'images')
test_labels_dir = os.path.join(ROOT_DIR, 'test', 'labels')

# Get the list of image and label files
train_images_file = os.listdir(train_images_dir)
train_labels_file = os.listdir(train_labels_dir)

val_images_file = os.listdir(val_images_dir)
val_labels_file = os.listdir(val_labels_dir)

test_images_file = os.listdir(test_images_dir)
test_labels_file = os.listdir(test_labels_dir)

print(len(train_images_file))
print(len(train_labels_file))
print(len(val_images_file))
print(len(val_labels_file))
print(len(test_images_file))
print(len(test_labels_file))


# # Remove the images without their associated labels
phase = "train"
# this script removed the images without label in train, test and val

def remove_images_without_labels():
    for phase in ['train', 'test', 'val']:
        image_dir = os.path.join("VisDrone", phase, "images")
        labels_dir = os.path.join("VisDrone", phase, "labels")
        i = 0

        for image in os.listdir(image_dir):
            # Create the correct annotation filename
            label_name = os.path.splitext(image)[0] + ".txt"
            label_path = os.path.join(labels_dir, label_name)

            if not os.path.exists(label_path):
                image_path = os.path.join(image_dir, image)
                os.remove(image_path)
                i += 1
                print(f"Removed {image_path} as corresponding label {label_path} is missing.")

        print(f"Removed {i} images without labels in {phase} phase.")

remove_images_without_labels()


# Update the lists of image and label files after removal
train_images_file = os.listdir(train_images_dir)
train_labels_file = os.listdir(train_labels_dir)
val_images_file = os.listdir(val_images_dir)
val_labels_file = os.listdir(val_labels_dir)
test_images_file = os.listdir(test_images_dir)
test_labels_file = os.listdir(test_labels_dir)

# Now print the updated lengths of the image and label files
print(len(train_images_file))
print(len(train_labels_file))
print(len(val_images_file))
print(len(val_labels_file))
print(len(test_images_file))
print(len(test_labels_file))

# Move around 1000 images and labels from training dataset to val dataset, making the ratio 4:1
# Set the number of images to move
num_images_to_move = 1000

# Randomly select 1000 images from the training dataset
selected_images = random.sample(train_images_file, num_images_to_move)

# Print the selected images before the move
print("Selected Images to Move:")
print(selected_images)

# Move the selected images and labels to the validation dataset folder
for image_file in selected_images:
    # Move the image
    src_image_path = os.path.join(train_images_dir, image_file)
    dst_image_path = os.path.join(val_images_dir, image_file)

    # Move the corresponding label (assuming the same filename but with a different extension)
    label_file = os.path.splitext(image_file)[0] + '.txt'
    src_label_path = os.path.join(train_labels_dir, label_file)
    dst_label_path = os.path.join(val_labels_dir, label_file)

    # Check if the label file exists before moving
    if os.path.exists(src_label_path):
        shutil.move(src_image_path, dst_image_path)
        shutil.move(src_label_path, dst_label_path)
    else:
        print(f"Label file {label_file} not found in the source directory. Skipping image: {image_file}")

print(f"Successfully moved {num_images_to_move} images and labels from training to validation dataset.")

# Update the lists of image and label files after the move
train_images_file = os.listdir(train_images_dir)
train_labels_file = os.listdir(train_labels_dir)
val_images_file = os.listdir(val_images_dir)
val_labels_file = os.listdir(val_labels_dir)

# Print the updated lengths of the image and label files
print("After Move:")
print(len(train_images_file))
print(len(train_labels_file))
print(len(val_images_file))
print(len(val_labels_file))
print(len(test_images_file))
print(len(test_labels_file))


# Shuffle the training and validation label files randomly
random_state = 42
random.shuffle(train_labels_file)
random.shuffle(val_labels_file)

# all_images_dir = os.path.join(ROOT_DIR, 'all_images')
# os.makedirs(all_images_dir, exist_ok=True)
# all_images = [os.path.join(train_images_dir, file) for file in train_images_file] + \
#               [os.path.join(val_images_dir, file) for file in val_images_file]

# # Ensure that 'all_images' is a list of file paths, not just directory paths
# all_images = [file for file in all_images if os.path.isfile(file)]

print("Checkpoint 1")
print("Length of train_images_file:", len(train_images_file))
print("Length of val_images_file:", len(val_images_file))


class CustomDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = io.imread(image_path)
        if self.transform:
            image = self.transform(image)
        return image

#Define the transformation to apply to the images
transform = transforms.Compose([transforms.ToPILImage(),  # Convert to PIL image
                                transforms.Resize((2560, 2560)),  # Resize the image
                                transforms.ToTensor()])  # Convert to a PyTorch tensor

# Prepare the data loaders
train_dataset = CustomDataset([os.path.join(ROOT_DIR, 'train', 'images', f) for f in os.listdir(os.path.join(ROOT_DIR, 'train', 'images'))], transform=transform)
val_dataset = CustomDataset([os.path.join(ROOT_DIR, 'val', 'images', f) for f in os.listdir(os.path.join(ROOT_DIR, 'val', 'images'))], transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

# Process your data and save the results
processed_images = []
for batch in tqdm(train_loader, desc='Processing Images'):
    processed_batch = transform(batch)
    processed_images.extend(processed_batch)

# Save the processed data
results_file = 'batch_processing_results.npz'

print("Checkpoint 2")