import os
import shutil
import random

train_dir = 'datasets/ISL_Dataset/train/'
val_dir = 'datasets/ISL_Dataset/val/'
test_dir = 'datasets/ISL_Dataset/test/'

if not os.path.exists(val_dir):
    os.makedirs(val_dir)

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

train_split = 0.8
val_split = 0.1
test_split = 0.1

for class_name in os.listdir(train_dir):
    class_train_dir = os.path.join(train_dir, class_name)
    class_val_dir = os.path.join(val_dir, class_name)
    class_test_dir = os.path.join(test_dir, class_name)

    if not os.path.exists(class_val_dir):
        os.makedirs(class_val_dir)

    if not os.path.exists(class_test_dir):
        os.makedirs(class_test_dir)

    images = os.listdir(class_train_dir)

    random.shuffle(images)

    num_train_images = int(len(images) * train_split)
    num_val_images = int(len(images) * val_split)
    num_test_images = int(len(images) * test_split)

    train_files = images[:num_train_images]
    val_files = images[num_train_images:num_train_images + num_val_images]
    test_files = images[num_train_images + num_val_images:]

    for img_name in val_files:
        src_path = os.path.join(class_train_dir, img_name)
        dest_path = os.path.join(class_val_dir, img_name)
        shutil.move(src_path, dest_path)

    for img_name in test_files:
        src_path = os.path.join(class_train_dir, img_name)
        dest_path = os.path.join(class_test_dir, img_name)
        shutil.move(src_path, dest_path)
