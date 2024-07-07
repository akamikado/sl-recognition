import os
import shutil
import random

train_dir = ''
val_dir = ''

if not os.path.exists(val_dir):
    os.makedirs(val_dir)

val_split = 0.2

for class_name in os.listdir(train_dir):
    class_train_dir = os.path.join(train_dir, class_name)
    class_val_dir = os.path.join(val_dir, class_name)

    if not os.path.exists(class_val_dir):
        os.makedirs(class_val_dir)

    images = os.listdir(class_train_dir)

    random.shuffle(images)

    num_val_images = int(len(images) * val_split)

    for i in range(num_val_images):
        img_name = images[i]
        src_path = os.path.join(class_train_dir, img_name)
        dest_path = os.path.join(class_val_dir, img_name)
        shutil.move(src_path, dest_path)

print("Training images split into training and validation sets.")
