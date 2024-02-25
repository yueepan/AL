import os
from pathlib import Path
from tqdm import tqdm 
# from ultralytics.yolo.utils.downloads import download

def visdrone2yolo(dir):
    from PIL import Image
    from tqdm import tqdm

    def convert_box(size, box):
        # Convert VisDrone box to YOLO xywh box
        dw = 1. / size[0]
        dh = 1. / size[1]
        return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

    (dir / 'labels').mkdir(parents=True, exist_ok=True)  # make labels directory
    pbar = tqdm((dir / 'annotations').glob('*.txt'), desc=f'Converting {dir}')
    for f in pbar:
        img_path = dir / 'images' / (f.stem + '.jpg')
        if img_path.exists():
            img_size = Image.open(img_path).size
            lines = []
    # for f in pbar:
    #     img_size = Image.open((dir / 'images' / f.name).with_suffix('.jpg')).size
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

dir = Path('/home/yue/AL/VisDrone')  # dataset root dir
# urls = ['https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip',
#           'https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-val.zip',
#           'https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-test-dev.zip',
#           'https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-test-challenge.zip']
# download(urls, dir=dir, curl=True, threads=4)

# Convert
for d in 'train', 'val', 'test':
      visdrone2yolo(dir / d)  # convert VisDrone annotations to YOLO labels

train_images_dir = os.path.join(dir, 'train', 'images')
val_images_dir = os.path.join(dir, 'val', 'images')
train_labels_dir = os.path.join(dir, 'train', 'labels')
val_labels_dir = os.path.join(dir, 'val', 'labels')
test_images_dir = os.path.join(dir, 'test', 'images')
test_labels_dir = os.path.join(dir, 'test', 'labels')

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

# import os
# from pathlib import Path
# from tqdm import tqdm 
# from PIL import Image

# dir = Path('/home/yue/AL/VisDrone')  # dataset root dir

# train_images_dir = os.path.join(dir, 'train', 'images')
# val_images_dir = os.path.join(dir, 'val', 'images')
# train_annotations_dir = os.path.join(dir, 'train', 'annotations')
# val_annotations_dir = os.path.join(dir, 'val', 'annotations')
# test_images_dir = os.path.join(dir, 'test', 'images')
# test_annotations_dir = os.path.join(dir, 'test', 'annotations')

# # Get the list of image and label files
# train_images_file = os.listdir(train_images_dir)
# train_annotations_file = os.listdir(train_annotations_dir)

# val_images_file = os.listdir(val_images_dir)
# val_annotations_file = os.listdir(val_annotations_dir)

# test_images_file = os.listdir(test_images_dir)
# test_annotations_file = os.listdir(test_annotations_dir)

# print(len(train_images_file))
# print(len(train_annotations_file))
# print(len(val_images_file))
# print(len(val_annotations_file))
# print(len(test_images_file))
# print(len(test_annotations_file))


# def visdrone2yolo(dir):
#     def convert_box(size, box):
#         dw = 1. / size[0]
#         dh = 1. / size[1]
#         return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

#     (dir / 'labels').mkdir(parents=True, exist_ok=True)  # make labels directory
#     pbar = tqdm((dir / 'annotations').glob('*.txt'), desc=f'Converting {dir}')
#     for f in pbar:
#         img_path = dir / 'images' / (f.stem + '.jpg')
#         if img_path.exists():
#             img_size = Image.open(img_path).size
#             lines = []
#             with open(f, 'r') as file:  # read annotations.txt
#                 for row in [x.split(',') for x in file.read().strip().splitlines()]:
#                     if row[4] == '0':  # VisDrone 'ignored regions' class 0
#                         continue
#                     cls = int(row[5]) - 1
#                     box = convert_box(img_size, tuple(map(int, row[:4])))
#                     lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
#             with open(str(f).replace(f'{os.sep}annotations{os.sep}', f'{os.sep}labels{os.sep}'), 'a') as fl:
#                 fl.writelines(lines)  # write label.txt
#         # else:
#         #     print(f"Image not found: {img_path}")

# dir = Path('/home/yue/AL/VisDrone')  # dataset root dir
# for d in 'train', 'val', 'test':
#     visdrone2yolo(dir / d)  # convert VisDrone annotations to YOLO labels

# train_images_dir = os.path.join(dir, 'train', 'images')
# val_images_dir = os.path.join(dir, 'val', 'images')
# train_labels_dir = os.path.join(dir, 'train', 'labels')
# val_labels_dir = os.path.join(dir, 'val', 'labels')
# test_images_dir = os.path.join(dir, 'test', 'images')
# test_labels_dir = os.path.join(dir, 'test', 'labels')

# # Get the list of image and label files
# train_images_file = os.listdir(train_images_dir)
# train_labels_file = os.listdir(train_labels_dir)

# val_images_file = os.listdir(val_images_dir)
# val_labels_file = os.listdir(val_labels_dir)

# test_images_file = os.listdir(test_images_dir)
# test_labels_file = os.listdir(test_labels_dir)

# print(len(train_images_file))
# print(len(train_labels_file))
# print(len(val_images_file))
# print(len(val_labels_file))
# print(len(test_images_file))
# print(len(test_labels_file))
