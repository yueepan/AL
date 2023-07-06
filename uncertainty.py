import os
import random
import shutil
import time
import numpy as np
from math import log2
from ultralytics import YOLO

ROOT_DIR = "./VisDrone"
OUTPUT_DIR = "./VisDrone/seeds_uncertainty"

training_images_dir = os.path.join(ROOT_DIR, 'train', 'images')
validation_images_dir = os.path.join(ROOT_DIR, 'val', 'images')
training_labels_dir = os.path.join(ROOT_DIR, 'train', 'labels')
validation_labels_dir = os.path.join(ROOT_DIR, 'val', 'labels')

# Get the list of image files
train_image_files = os.listdir(training_images_dir)
val_image_files = os.listdir(validation_images_dir)

# Get the list of label files
train_label_files = os.listdir(training_labels_dir)
val_label_files = os.listdir(validation_labels_dir)

# Create sets of image and label filenames (without extensions) for training and validation
train_images_set = {os.path.splitext(file)[0] for file in train_image_files}
train_labels_set = {os.path.splitext(file)[0] for file in train_label_files}
val_images_set = {os.path.splitext(file)[0] for file in val_image_files}
val_labels_set = {os.path.splitext(file)[0] for file in val_label_files}

# Find images without labels for training and validation
nolabel_train_images = train_images_set - train_labels_set
nolabel_val_images = val_images_set - val_labels_set

print(len(nolabel_train_images))
print(len(nolabel_val_images))

# Print the images without labels for training
print("Number of the training Images without Labels:")
for image_file in nolabel_train_images:
    print(len(image_file))

# Print the images without labels for validation
print("Number of the validation Images without Labels:")
for image_file in nolabel_val_images:
    print(len(image_file))

# List of percentages for each seed folder
percentages = [1, 2, 2, 5, 10, 15, 15]

# Create the initial seed folder (1%)
initial_seed_folder = os.path.join(OUTPUT_DIR, "initial seed")
os.makedirs(initial_seed_folder, exist_ok=True)

# Calculate the number of labels for the initial seed
num_initial_labels = int(percentages[0] / 100 * (len(os.listdir(training_labels_dir)) + len(os.listdir(validation_labels_dir))))
num_initial_train_labels = int(percentages[0] / 100 * len(os.listdir(training_labels_dir)))
num_initial_val_labels = int(percentages[0] / 100 * len(os.listdir(validation_labels_dir)))

# Shuffle the training and validation label files randomly
random_state = 42
training_label_files = os.listdir(training_labels_dir)
random.shuffle(training_label_files)
validation_label_files = os.listdir(validation_labels_dir)
random.shuffle(validation_label_files)

# Select the labels from the training and val datasets
initial_train_labels = training_label_files[:num_initial_train_labels]
initial_val_labels = validation_label_files[:num_initial_val_labels]

# Create the train and val directories within the initial seed folder
initial_train_dir = os.path.join(initial_seed_folder, 'train')
initial_val_dir = os.path.join(initial_seed_folder, 'val')
os.makedirs(initial_train_dir, exist_ok=True)
os.makedirs(initial_val_dir, exist_ok=True)

# Create the images and labels directories within the initial train folder
initial_train_images_dir = os.path.join(initial_train_dir, 'images')
initial_train_labels_dir = os.path.join(initial_train_dir, 'labels')
os.makedirs(initial_train_images_dir, exist_ok=True)
os.makedirs(initial_train_labels_dir, exist_ok=True)

# Create the images and labels directories within the initial val folder
initial_val_images_dir = os.path.join(initial_val_dir, 'images')
initial_val_labels_dir = os.path.join(initial_val_dir, 'labels')
os.makedirs(initial_val_images_dir, exist_ok=True)
os.makedirs(initial_val_labels_dir, exist_ok=True)

initial_train_images = os.listdir(initial_train_images_dir)
initial_val_images = os.listdir(initial_val_images_dir)

# Copy the selected initial seed labels and their corresponding images to the output directories
for label_file in initial_train_labels:
    src_label_path = os.path.join(training_labels_dir, label_file)
    dst_label_path = os.path.join(initial_train_labels_dir, label_file)
    shutil.copyfile(src_label_path, dst_label_path)

    # Get the corresponding image file
    image_file = os.path.splitext(label_file)[0] + '.jpg'
    src_image_path = os.path.join(training_images_dir, image_file)
    dst_image_path = os.path.join(initial_train_images_dir, image_file)
    shutil.copyfile(src_image_path, dst_image_path)

for label_file in initial_val_labels:
    src_label_path = os.path.join(validation_labels_dir, label_file)
    dst_label_path = os.path.join(initial_val_labels_dir, label_file)
    shutil.copyfile(src_label_path, dst_label_path)

    # Get the corresponding image file
    image_file = os.path.splitext(label_file)[0] + '.jpg'
    src_image_path = os.path.join(validation_images_dir, image_file)
    dst_image_path = os.path.join(initial_val_images_dir, image_file)
    shutil.copyfile(src_image_path, dst_image_path)

print("Created initial seed 'initial_seed' with {}% of labels".format(percentages[0]))

initial_seed_dir = os.path.join(OUTPUT_DIR, "initial seed")

# Model building
model = YOLO('yolov8n.pt')  
results = model.train(data = os.path.join(ROOT_DIR, "initial seed.yaml"), epochs =100, patience = 20, imgsz= 2560, batch= 4)

# Create a list of all training and validation labels
all_labels = training_label_files + validation_label_files
unlabeled_labels = list(set(all_labels) - set(initial_train_labels) - set(initial_val_labels))

#create a list of all images without the initial images
train_images = [os.path.join(training_images_dir, file) for file in os.listdir(training_images_dir)]
val_images = [os.path.join(validation_images_dir, file) for file in os.listdir(validation_images_dir)]
all_images = train_images + val_images

# remove ones from initial seed
# unlabeled_images = all_images - initial_train_images - initial_val_images - nolabel_train_images - nolabel_val_images
unlabeled_images = list(set(all_images) - set(initial_train_images) - set(initial_val_images) - nolabel_train_images - nolabel_val_images)
 
print("Unlabeled Images:")
for image_path in unlabeled_images:
    print(image_path)

# Create the unlabeled images and labels directories
unlabeled_data_dir = os.path.join(ROOT_DIR, 'unlabeled_data')
os.makedirs(unlabeled_data_dir, exist_ok=True)

unlabeled_images_dir = os.path.join(unlabeled_data_dir, 'unlabeled_images')
os.makedirs(unlabeled_images_dir, exist_ok=True)

unlabeled_labels_dir = os.path.join(unlabeled_data_dir, 'unlabeled_labels')
os.makedirs(unlabeled_labels_dir, exist_ok=True)


# Perform predictions on the unlabeled images (99%)
model = YOLO('best.pt') 
predictions = model.predict(unlabeled_images, save=True, imgsz=2560, conf=0.5)

# Calculate the average confidence for each image
average_list = [np.average(pred.boxes.conf.cpu()) for pred in predictions]

# Sort the indices based on least confidence
sorted_indices_least_confidence = np.argsort(average_list)

# Create the seed folder for seed 2 (2% of the unlabeled data)
seed_folder = os.path.join(OUTPUT_DIR, "seed2")
os.makedirs(seed_folder, exist_ok=True)

# Create seed2 by calculate the number of labels to select based on the percentage 2
num_labels_to_select = int(2 / 100 * len(unlabeled_labels))
# Select the least confident data points
selected_indices = sorted_indices_least_confidence[:num_labels_to_select]

# Copy the selected labels and their corresponding images to the seed folder
for idx in selected_indices:
    label_file = unlabeled_labels[idx]

    src_label_path = os.path.join(unlabeled_labels_dir, label_file)
    dst_label_path = os.path.join(seed_folder, 'labels', label_file)
    shutil.copyfile(src_label_path, dst_label_path)

    image_file = os.path.splitext(label_file)[0] + '.jpg'
    src_image_path = os.path.join(unlabeled_images_dir, image_file)
    dst_image_path = os.path.join(seed_folder, 'images', image_file)
    shutil.copyfile(src_image_path, dst_image_path)

# Copy the images and labels from the initial seed to seed2
initial_train_images_dir = os.path.join(initial_seed_dir, 'train', 'images')
initial_train_labels_dir = os.path.join(initial_seed_dir, 'train', 'labels')
initial_val_images_dir = os.path.join(initial_seed_dir, 'val', 'images')
initial_val_labels_dir = os.path.join(initial_seed_dir, 'val', 'labels')

os.makedirs(os.path.join(seed_folder, 'images'), exist_ok=True)
os.makedirs(os.path.join(seed_folder, 'labels'), exist_ok=True)

# Copy the images and labels from the initial train set to seed2
for label_file in os.listdir(initial_train_labels_dir):
    src_label_path = os.path.join(initial_train_labels_dir, label_file)
    dst_label_path = os.path.join(seed_folder, 'labels', label_file)
    shutil.copyfile(src_label_path, dst_label_path)

    image_file = os.path.splitext(label_file)[0] + '.jpg'
    src_image_path = os.path.join(initial_train_images_dir, image_file)
    dst_image_path = os.path.join(seed_folder, 'images', image_file)
    shutil.copyfile(src_image_path, dst_image_path)

# Copy the images and labels from the initial val set to seed2
for label_file in os.listdir(initial_val_labels_dir):
    src_label_path = os.path.join(initial_val_labels_dir, label_file)
    dst_label_path = os.path.join(seed_folder, 'labels', label_file)
    shutil.copyfile(src_label_path, dst_label_path)

    image_file = os.path.splitext(label_file)[0] + '.jpg'
    src_image_path = os.path.join(initial_val_images_dir, image_file)
    dst_image_path = os.path.join(seed_folder, 'images', image_file)
    shutil.copyfile(src_image_path, dst_image_path)

print(f"Copied the initial seed and 2% of labels to the 'seed2'")



## LOOPS
# Iterate over the remaining percentages and create the seeds
for seed in range(2, len(percentages) + 1):
    # Create the seed folder
    seed_folder = os.path.join(OUTPUT_DIR, f"seed{seed}_uncertainty")
    os.makedirs(seed_folder, exist_ok=True)

    # Calculate the number of labels to select based on the percentage
    num_labels_to_select = int(percentages[seed - 1] / 100 * len(unlabeled_images))
    # Select the least confident data points
    selected_indices = sorted_indices_least_confidence[:num_labels_to_select]

    # Copy the selected labels and their corresponding images to the seed folder
    for idx in selected_indices:
        label_file = unlabeled_labels[idx]

        src_label_path = os.path.join(unlabeled_labels_dir, label_file)
        dst_label_path = os.path.join(seed_folder, 'labels', label_file)
        shutil.copyfile(src_label_path, dst_label_path)

        image_file = os.path.splitext(label_file)[0] + '.jpg'
        src_image_path = os.path.join(unlabeled_images_dir, image_file)
        dst_image_path = os.path.join(seed_folder, 'images', image_file)
        shutil.copyfile(src_image_path, dst_image_path)

    # Copy the images and labels from the initial seed to seed2
    initial_train_images_dir = os.path.join(initial_seed_dir, 'train', 'images')
    initial_train_labels_dir = os.path.join(initial_seed_dir, 'train', 'labels')
    initial_val_images_dir = os.path.join(initial_seed_dir, 'val', 'images')
    initial_val_labels_dir = os.path.join(initial_seed_dir, 'val', 'labels')

    os.makedirs(os.path.join(seed_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(seed_folder, 'labels'), exist_ok=True)

    # Copy the images and labels from the initial train set to seeds
    for label_file in os.listdir(initial_train_labels_dir):
        src_label_path = os.path.join(initial_train_labels_dir, label_file)
        dst_label_path = os.path.join(seed_folder, 'labels', label_file)
        shutil.copyfile(src_label_path, dst_label_path)

        image_file = os.path.splitext(label_file)[0] + '.jpg'
        src_image_path = os.path.join(initial_train_images_dir, image_file)
        dst_image_path = os.path.join(seed_folder, 'images', image_file)
        shutil.copyfile(src_image_path, dst_image_path)

    # Copy the images and labels from the initial val set to seeds
    for label_file in os.listdir(initial_val_labels_dir):
        src_label_path = os.path.join(initial_val_labels_dir, label_file)
        dst_label_path = os.path.join(seed_folder, 'labels', label_file)
        shutil.copyfile(src_label_path, dst_label_path)

        image_file = os.path.splitext(label_file)[0] + '.jpg'
        src_image_path = os.path.join(initial_val_images_dir, image_file)
        dst_image_path = os.path.join(seed_folder, 'images', image_file)
        shutil.copyfile(src_image_path, dst_image_path)

        print(f"Created seed '{seed_folder}' with {percentages[seed - 1]}% of labels")


#model train on each seed

# Perform inference on the test dataset
start_time = time.time()
model.predict(os.path.join(ROOT_DIR, "test", "images"), save=True, imgsz=2560, conf=0.5)
end_time = time.time()
duration = end_time - start_time
print("Inference Duration:", duration, "seconds")