import os
import random
import shutil
import time
import numpy as np
from math import log2
import yaml
from ultralytics import YOLO

ROOT_DIR = "./VisDrone"
OUTPUT_DIR = "./VisDrone/seeds_uncertainty"

phase = "train"
# this script removed the images without label in train, test and val

def remove_images_without_labels():
    for phase in ['train', 'test', 'val']:
        i=0
        for image in os.listdir(os.path.join("VisDrone", phase, "images")):
            # check if corresponding annotation file exists
            annotation_name = image.replace("jpg", "txt")
            if not os.path.isfile(os.path.join("VisDrone", phase, "labels", annotation_name)):
                i=i+1
                os.remove(os.path.join("VisDrone", phase, "images", image))
                print(phase, i)



training_images_dir = os.path.join(ROOT_DIR, 'train', 'images')
validation_images_dir = os.path.join(ROOT_DIR, 'val', 'images')
training_labels_dir = os.path.join(ROOT_DIR, 'train', 'labels')
validation_labels_dir = os.path.join(ROOT_DIR, 'val', 'labels')

# Get the list of image and label files
train_image_files = os.listdir(training_images_dir)
val_image_files = os.listdir(validation_images_dir)
train_label_files = os.listdir(training_labels_dir)
val_label_files = os.listdir(validation_labels_dir)

len(train_image_files) # 5207
len(train_label_files) # 5207
len(val_image_files) # 1549
len(val_label_files) # 1549


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


# Model building
model = YOLO('yolov8n.pt')  
results = model.train(data = os.path.join(ROOT_DIR, "initial seed.yaml"), epochs =10, patience = 20, imgsz= 2560, batch= 4)

# Create a list of all training and validation labels
all_labels = training_label_files + validation_label_files

# remove ones from initial seed and images with no labels
# unlabeled_images = list(train_images_set + val_images_set -set(initial_train_images) - set(initial_val_images))
unlabeled_images = (
    [os.path.join(training_images_dir, file) for file in train_image_files if file not in initial_train_images] +
    [os.path.join(validation_images_dir, file) for file in val_image_files if file not in initial_val_images])
# remove the images that are in initial train or initial val

# unlabeled_labels
unlabeled_labels = (
    [os.path.join(training_labels_dir, file) for file in train_label_files if file not in initial_train_labels] +
    [os.path.join(validation_labels_dir, file) for file in val_label_files if file not in initial_val_labels])


len(unlabeled_images) # 6756 without removing the ones used for training
len(unlabeled_images) # 6689 --> conclusion; 52+15 used for training/val for initial_eeed


# # Perform predictions on the unlabeled images (99%)
# # model = YOLO('best.pt') 
model = YOLO("/home/yue/AL/data/best.pt")

# Run batched inference on a list of images
results = model(unlabeled_images, save=True, imgsz=2560, stream=True)  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs

# # reduce confidence to get some outputs
# # confidence is low because we trained for only 1 epoch
# # predictions = model.predict(unlabeled_images[:10], save=True, imgsz=256) #dont forget to remove the [:10]
# predictions = model(unlabeled_images, save=True, imgsz=2560)


# # Calculate the average confidence for each image
# average_list = [np.average(pred.boxes.conf.cpu()) for pred in predictions]
# average_list[1] = 0.2 # can be removed later
# average_list[2] = 0.4
# average_list[3] = 0.5

# predictions[0].boxes
# len(average_list)

# predict by entropy scores of each image
# entropy_scores = -np.sum(np.multiply(predictions, np.log2(np.maximum(predictions, 1e-10))))
entropy_scores = -np.sum(np.multiply(results, np.log2(np.maximum(results, 1e-10))))

# Sort the indices based on entropy scores in descending order
sorted_indices_entropy= np.argsort(entropy_scores)[::-1]

# Create the seed folder for seed 2 (2% of the unlabeled data)
num_labels_to_select = int(percentages / 100 * len(unlabeled_images))
print(num_labels_to_select)
# num_labels_to_select = 2 # overwrite num_labels_to_select because we predicted only 10 images
# Select the least confident data points
selected_indices = sorted_indices_entropy[:num_labels_to_select]

new_seed_name = "seed2"
old_seed_name = "initial seed"
idx=1

type(selected_indices)

def create_new_seed(new_seed_name: str, old_seed_name: str, selected_indices: np.ndarray):
    # copy files and structure from old seed to new seed
    shutil.copytree(os.path.join(OUTPUT_DIR, old_seed_name), os.path.join(OUTPUT_DIR, new_seed_name))
    print(f"Structure from {old_seed_name} copied to {new_seed_name}")
    # copy files that were annotated to new seed
    for idx in selected_indices:
        # copy label
        label_file = unlabeled_images[idx].replace("jpg", "txt").replace("images", "labels")
        #if random.random() < 0.7:
        if True:
            shutil.copy(label_file, os.path.join(OUTPUT_DIR, new_seed_name, "train", "labels", label_file.split("/")[-1]))
            # copy image
            shutil.copy(unlabeled_images[idx], os.path.join(OUTPUT_DIR, new_seed_name, "train", "images", unlabeled_images[idx].split("/")[-1]))
        else:
            shutil.copy(label_file, os.path.join(OUTPUT_DIR, new_seed_name, "val", "labels", label_file.split("/")[-1]))
            # copy image
            shutil.copy(unlabeled_images[idx], os.path.join(OUTPUT_DIR, new_seed_name, "val", "images", unlabeled_images[idx].split("/")[-1]))
    print(f"{len(selected_indices)} images and labels copied from unlabeled_images to {new_seed_name}")
    # create yaml file
    yaml_dict = {
        "path": "/home/yue/AL/data/VisDrone",
        "train": os.path.join("/home/yue/AL/data/VisDrone/seeds_uncertainty/", new_seed_name, "train/images"),
        "val": os.path.join("/home/yue/AL/data/VisDrone/seeds_uncertainty/", new_seed_name, "val/images"),
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
        }
    }
    yaml_file=open(os.path.join(OUTPUT_DIR, new_seed_name, f"{new_seed_name}.yaml"),"w")
    yaml.dump(yaml_dict,yaml_file)
    print(f"YAML file {new_seed_name}.yaml created")

# apply function
create_new_seed("seed2", old_seed_name="initial seed", selected_indices=selected_indices)

## CREATE EACH COMBINED 
combined_seed_base_folder = os.path.join(OUTPUT_DIR, "combined_seeds")
os.makedirs(combined_seed_base_folder, exist_ok=True)
num_unlabeled_data = len(unlabeled_images)
# Calculate the number of data points for each seed based on the percentage
num_data_per_seed = [int(p / 100 * num_unlabeled_data) for p in percentages]

for i, num_data in enumerate(num_data_per_seed):
    seed_folder = os.path.join(combined_seed_base_folder, f"seed{i+1}")
    os.makedirs(seed_folder, exist_ok=True)
    seed_folder.append(seed_folder)

    # Copy the data from previous seeds
    for j in range(i):
        previous_seed_folder = seed_folder[j]

        src_images_dir = os.path.join(previous_seed_folder, "images")
        dst_images_dir = os.path.join(seed_folder, "images")
        shutil.copytree(src_images_dir, dst_images_dir)

        src_labels_dir = os.path.join(previous_seed_folder, "labels")
        dst_labels_dir = os.path.join(seed_folder, "labels")
        shutil.copytree(src_labels_dir, dst_labels_dir)

    # Copy new data for the current seed
    end_index = start_index + num_data
    for image_path in unlabeled_images[start_index:end_index]:

        label_file = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        src_label_path = os.path.join(unlabeled_labels, label_file)
        dst_label_path = os.path.join(seed_folder, "labels", label_file)
        shutil.copyfile(src_label_path, dst_label_path)

        dst_image_path = os.path.join(seed_folder, "images", os.path.basename(image_path))
        shutil.copyfile(image_path, dst_image_path)

    start_index = end_index
    print(f"Created seed '{os.path.basename(seed_folder)}' with {percentages[i]}% of labels")


#model train on each seed
model = YOLO('/home/yue/AL/data/best.pt')   
results = model.train(data = os.path.join(ROOT_DIR, "seed2.yaml"), epochs =100, patience = 20, imgsz= 2560, batch= 4)
results = model.train(data = os.path.join(ROOT_DIR, "seed3.yaml"), epochs =100, patience = 20, imgsz= 2560, batch= 4)
results = model.train(data = os.path.join(ROOT_DIR, "seed4.yaml"), epochs =100, patience = 20, imgsz= 2560, batch= 4)
results = model.train(data = os.path.join(ROOT_DIR, "seed5.yaml"), epochs =100, patience = 20, imgsz= 2560, batch= 4)
results = model.train(data = os.path.join(ROOT_DIR, "seed6.yaml"), epochs =100, patience = 20, imgsz= 2560, batch= 4)
results = model.train(data = os.path.join(ROOT_DIR, "seed7.yaml"), epochs =100, patience = 20, imgsz= 2560, batch= 4)

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs


# Perform inference on the test dataset
start_time = time.time()
model.predict(os.path.join(ROOT_DIR, "test", "images"), save=True, imgsz=2560, conf=0.5)
end_time = time.time()
duration = end_time - start_time
print("Inference Duration:", duration, "seconds")
