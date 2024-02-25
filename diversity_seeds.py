import os
import shutil
import numpy as np
from torchvision.io import read_image
from torchvision.transforms import Resize
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances
import random
import eco2ai
import yaml
import time
from math import log2
from ultralytics import YOLO
import torch

ROOT_DIR = "./VisDrone"
OUTPUT_DIR = os.path.join(ROOT_DIR, 'diversity_seeds')

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



# Keep track of the carbon emissions for the model training
tracker = eco2ai.Tracker(
    project_name="Diversity Seeds Carbon", 
    experiment_description="training <diversity seeds> model",
    file_name="emission.csv"
    )

tracker.start()

diversity_scores_file = os.path.join(OUTPUT_DIR, 'diversity_scores.npz')

# Check if the 'diversity_scores.npz' file already exists
if not os.path.exists(diversity_scores_file):
    def load_and_process_images(image_dir, image_files, resize_size=(1280, 1280), batch_size=10):
        resize = Resize(resize_size)
        processed_images = []

        for i, file in enumerate(image_files):
            image_path = os.path.join(image_dir, file)
            image = read_image(image_path)
            image = resize(image)
            processed_images.append(image.numpy().flatten())

            if (i + 1) % batch_size == 0 or i == len(image_files) - 1:
                yield np.array(processed_images)
                processed_images = []

    def apply_minibatch_kmeans(image_dir, image_files, n_clusters=10, batch_size=10, resize_size=(1280, 1280)):
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size)
        for batch in load_and_process_images(image_dir, image_files, resize_size, batch_size):
            if batch.ndim < 2:
                batch = batch.reshape(-1, 1)
            kmeans.partial_fit(batch)

        # Calculate distances for each image batch and then maximum distances
        max_distances = []
        for batch in load_and_process_images(image_dir, image_files, resize_size, batch_size):
            distances = euclidean_distances(batch, kmeans.cluster_centers_)
            max_distances.extend(np.max(distances, axis=1))

        return np.array(max_distances)

    # Example usage
    diversity_scores = apply_minibatch_kmeans(train_images_dir, train_images_file, n_clusters=10, batch_size=10, resize_size=(1280, 1280))

    # Combine filenames with their diversity scores and sort in descending order
    image_scores = list(zip(train_images_file, diversity_scores))
    image_scores.sort(key=lambda x: x[1], reverse=True)  # Sort in descending order

    # Extract sorted filenames and indices
    sorted_filenames = [x[0] for x in image_scores]

    # Ensure OUTPUT_DIR exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Now save the .npz file
    np.savez(diversity_scores_file, sorted_filenames=sorted_filenames, diversity_scores=diversity_scores)
    print("Images sorted based on diversity scores.")
else:
    # Load diversity scores and sorted filenames from the file
    loaded_results = np.load(diversity_scores_file)
    diversity_scores = loaded_results['diversity_scores']
    sorted_filenames = loaded_results['sorted_filenames']
    print("Loaded existing diversity scores and sorted filenames.")

loaded_results = np.load(diversity_scores_file)
diversity_scores = loaded_results['diversity_scores']
sorted_filenames = loaded_results['sorted_filenames']
print(f'sorted_indices: {len(sorted_filenames)}')

# Function to count images in a folder
def count_images_in_folder(folder_path):
    image_files = [file for file in os.listdir(folder_path) if file.endswith('.jpg')]
    return len(image_files)

cumulative_train_images = set()
cumulative_val_images = set()

# Percentage of images to add for each seed (10% of the original dataset size)
percentage_per_seed = 0.1
seed_counter = 1

while seed_counter <= 4:  # Create up to seed4
    # Determine the seed name
    seed_name = f'seed{seed_counter}'
    seed_folder = os.path.join(OUTPUT_DIR, seed_name)

    if os.path.exists(seed_folder):
        print(f"{seed_folder} already exists, moving to the next seed.")
        # Update cumulative sets with existing images
        train_folder = os.path.join(seed_folder, 'train', 'images')
        val_folder = os.path.join(seed_folder, 'val', 'images')
        cumulative_train_images.update({filename for filename in os.listdir(train_folder) if filename.endswith('.jpg')})
        cumulative_val_images.update({filename for filename in os.listdir(val_folder) if filename.endswith('.jpg')})
    else:
        # Calculate the number of new images to add
        total_train_images_needed = int(len(train_images_file) * (percentage_per_seed * seed_counter))
        total_val_images_needed = int(len(val_images_file) * (percentage_per_seed * seed_counter))

        num_new_train_images = total_train_images_needed - len(cumulative_train_images)
        num_new_val_images = total_val_images_needed - len(cumulative_val_images)

        # Select new images from the remaining datasets
        remaining_train_filenames = [filename for filename in train_images_file if filename not in cumulative_train_images]
        seed_images_train = random.sample(remaining_train_filenames, min(num_new_train_images, len(remaining_train_filenames)))

        remaining_val_filenames = [filename for filename in val_images_file if filename not in cumulative_val_images]
        seed_images_val = random.sample(remaining_val_filenames, min(num_new_val_images, len(remaining_val_filenames)))

        # Add the new images to the cumulative sets
        cumulative_train_images.update(seed_images_train)
        cumulative_val_images.update(seed_images_val)

    # Determine the seed name
    seed_name = f'seed{seed_counter}'
    seed_counter += 1

    # Create a seed folder
    seed_folder = os.path.join(OUTPUT_DIR, seed_name)

    if not os.path.exists(seed_folder):
        os.makedirs(seed_folder, exist_ok=True)

        # Create "train" and "val" folders
        train_folder = os.path.join(seed_folder, 'train')
        val_folder = os.path.join(seed_folder, 'val')
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)

        # Create "images" and "labels" folders inside "train" and "val"
        for folder in ['images', 'labels']:
            os.makedirs(os.path.join(train_folder, folder), exist_ok=True)
            os.makedirs(os.path.join(val_folder, folder), exist_ok=True)

        # Copy selected images to the "train" folder
        for filename in cumulative_train_images:
            image_path = os.path.join(train_images_dir, filename)
            label_path = os.path.join(train_labels_dir, filename.replace(".jpg", ".txt"))

            if os.path.exists(image_path) and os.path.exists(label_path):
                shutil.copy(image_path, os.path.join(train_folder, 'images'))
                shutil.copy(label_path, os.path.join(train_folder, 'labels'))

        # Copy selected images to the "val" folder
        for filename in cumulative_val_images:
            image_path = os.path.join(val_images_dir, filename)
            label_path = os.path.join(val_labels_dir, filename.replace(".jpg", ".txt"))

            if os.path.exists(image_path) and os.path.exists(label_path):
                shutil.copy(image_path, os.path.join(val_folder, 'images'))
                shutil.copy(label_path, os.path.join(val_folder, 'labels'))

        # Count and print the number of images in the train and val folders of the current seed
        num_train_images_current_seed = count_images_in_folder(os.path.join(train_folder, 'images'))
        num_val_images_current_seed = count_images_in_folder(os.path.join(val_folder, 'images'))
    

        # Create YAML file for the seed
        yaml_dict = {
            "path": ROOT_DIR,
            "train": os.path.join("diversity_seeds/", seed_name, "train/images"),
            "val": os.path.join("diversity_seeds/", seed_name, "val/images"),
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

        yaml_file_path = os.path.join(OUTPUT_DIR, f"{os.path.basename(seed_folder)}.yaml")

        with open(yaml_file_path, "w") as yaml_file:
            yaml.dump(yaml_dict, yaml_file)
            print(f"YAML file {yaml_file_path} created")
            
        print(f"Seed {seed_name}:")
        print(f"Train images: {num_train_images_current_seed}")
        print(f"Val images: {num_val_images_current_seed}")

        # Update cumulative images for the next iteration
        cumulative_train_images.update(set(seed_images_train))
        cumulative_val_images.update(set(seed_images_val))

    # Training on the current seed
    print(f"Training YOLO model on {seed_folder}...")
    yaml_file_path = os.path.join(OUTPUT_DIR, f"{os.path.basename(seed_folder)}.yaml")

    model = YOLO('yolov8n.pt')
    results = model.train(data=yaml_file_path, epochs=50, patience=10, imgsz=1280, batch=4)

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
    print(f"Inference Duration for {seed_folder}: {duration} seconds")

tracker.stop()

torch.cuda.empty_cache()