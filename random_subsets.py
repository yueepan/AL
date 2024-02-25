import os
import shutil
import random
from ultralytics import YOLO
import eco2ai
import torch

ROOT_DIR = "./VisDrone"
OUTPUT_DIR = os.path.join(ROOT_DIR, 'random_subsets')

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



# Keep track of the carbon emissions for the model training
tracker = eco2ai.Tracker(
    project_name="Random Sampling Carbon",
    experiment_description="training <Random Subsets> model",
    file_name="emission.csv"
)

tracker.start()

# Adjusted percentages for each subset
cumulative_percentages = [10, 20, 30, 40]


def create_subset_and_yaml(subset_name, train_images_list, val_images_list, train_labels_dir, val_labels_dir, output_dir, cumulative_train_images, cumulative_val_images, additional_percentage):
    # Calculate the number of new images to add based on the total dataset size
    num_new_train_images = int(additional_percentage / 100 * len(train_images_list))
    num_new_val_images = int(additional_percentage / 100 * len(val_images_list))

    # Ensure that new images are not already in the cumulative lists
    remaining_train_images = list(set(train_images_list) - set(cumulative_train_images))
    remaining_val_images = list(set(val_images_list) - set(cumulative_val_images))

    new_train_images = random.sample(remaining_train_images, min(num_new_train_images, len(remaining_train_images)))
    new_val_images = random.sample(remaining_val_images, min(num_new_val_images, len(remaining_val_images)))

    subset_train_images = cumulative_train_images + new_train_images
    subset_val_images = cumulative_val_images + new_val_images

    # Create the subset directories within the output folder
    subset_dir = os.path.join(output_dir, f"subset{subset_name}")
    os.makedirs(subset_dir, exist_ok=True)

    # Create train and val directories within the subset folder
    subset_train_dir = os.path.join(subset_dir, 'train')
    subset_val_dir = os.path.join(subset_dir, 'val')
    os.makedirs(subset_train_dir, exist_ok=True)
    os.makedirs(subset_val_dir, exist_ok=True)

    # Create images and labels directories within the train and val folders
    subset_train_images_dir = os.path.join(subset_train_dir, 'images')
    subset_train_labels_dir = os.path.join(subset_train_dir, 'labels')
    os.makedirs(subset_train_images_dir, exist_ok=True)
    os.makedirs(subset_train_labels_dir, exist_ok=True)
    subset_val_images_dir = os.path.join(subset_val_dir, 'images')
    subset_val_labels_dir = os.path.join(subset_val_dir, 'labels')
    os.makedirs(subset_val_images_dir, exist_ok=True)
    os.makedirs(subset_val_labels_dir, exist_ok=True)

    # Copy all cumulative images and their corresponding labels to the output directories
    for image_file in subset_train_images:
        src_image_path = os.path.join(train_images_dir, image_file)
        dst_image_path = os.path.join(subset_train_images_dir, image_file)
        shutil.copyfile(src_image_path, dst_image_path)

        label_file = os.path.splitext(image_file)[0] + '.txt'
        src_label_path = os.path.join(train_labels_dir, label_file)
        dst_label_path = os.path.join(subset_train_labels_dir, label_file)
        shutil.copyfile(src_label_path, dst_label_path)

    for image_file in subset_val_images:
        src_image_path = os.path.join(val_images_dir, image_file)
        dst_image_path = os.path.join(subset_val_images_dir, image_file)
        shutil.copyfile(src_image_path, dst_image_path)

        label_file = os.path.splitext(image_file)[0] + '.txt'
        src_label_path = os.path.join(val_labels_dir, label_file)
        dst_label_path = os.path.join(subset_val_labels_dir, label_file)
        shutil.copyfile(src_label_path, dst_label_path)

    # Create a YAML file for the subset
    yaml_content = f"""
    path: '{ROOT_DIR}/random_subsets'  # dataset root dir
    train: 'subset{subset_name}/train/images'  # train images (relative to 'path')
    val: 'subset{subset_name}/val/images'  # val images (relative to 'path')
    test: '{ROOT_DIR}/test/images'  # test images (optional)
    names:
        0: pedestrian
        1: people
        2: bicycle
        3: car
        4: van
        5: truck
        6: tricycle
        7: awning-tricycle
        8: bus
        9: motor
    """
    yaml_path = os.path.join(subset_dir, f'subset_{subset_name}.yaml')
    with open(yaml_path, 'w') as yaml_file:
        yaml_file.write(yaml_content)

    return yaml_path, subset_train_images, subset_val_images

# Initialize the cumulative images from previous subsets
cumulative_train_images = []
cumulative_val_images = []


for subset_name, cumulative_percentage in enumerate(cumulative_percentages, start=1):
    subset_dir = os.path.join(OUTPUT_DIR, f"subset{subset_name}")

    if not os.path.exists(subset_dir):
        # Calculate the additional percentage of images needed for the new subset
        if subset_name == 1:
            additional_percentage = cumulative_percentage
        else:
            additional_percentage = cumulative_percentage - cumulative_percentages[subset_name - 2]

        train_yaml, subset_train_images, subset_val_images = create_subset_and_yaml(
            subset_name, train_images_file, val_images_file, train_labels_dir, val_labels_dir, OUTPUT_DIR, cumulative_train_images, cumulative_val_images, additional_percentage)

        # Update the cumulative lists
        cumulative_train_images = subset_train_images
        cumulative_val_images = subset_val_images

        # Create YOLO model for each subset
        model = YOLO('yolov8n.pt')

        # Train the model on the created subset
        results = model.train(data=train_yaml, epochs=50, patience=10, imgsz=1280, batch=4)
    

        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered

        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # a list contains map50-95 of each category

    else:
        print(f"Subset {subset_name} already exists. Skipping creation.")

    # If subset 4 is reached, break out of the loop
    if subset_name == 4:
        break

tracker.stop()

torch.cuda.empty_cache()