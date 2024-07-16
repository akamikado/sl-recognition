import os
import shutil
import random

dataset = 'asl_alphabet_train'

src_dir = f'datasets/{dataset}'


def split_dataset(split):
    final_dir = f'datasets/{dataset}-{int(split*100)}'
    if os.path.exists(f'{final_dir}') and os.path.exists(f'{final_dir}.yaml'):
        return

    shutil.copy(f'{src_dir}.yaml', f'{final_dir}.yaml')

    if not os.path.exists(final_dir):
        os.makedirs(final_dir)

    for mode in os.listdir(src_dir):
        src_mode = os.path.join(src_dir, mode)
        dest_mode = os.path.join(final_dir, mode)
        if not os.path.exists(dest_mode):
            os.makedirs(dest_mode)
        for class_name in os.listdir(src_mode):
            src_class_dir = os.path.join(src_mode, class_name)
            dest_class_dir = os.path.join(dest_mode, class_name)
            if not os.path.exists(dest_class_dir):
                os.makedirs(dest_class_dir)
            images = os.listdir(src_class_dir)
            random.shuffle(images)

            num_images = int(len(images)*split)
            for i in range(num_images):
                img = images[i]
                src_img = os.path.join(src_class_dir, img)
                dest_img = os.path.join(dest_class_dir, img)
                shutil.copy(src_img, dest_img)


def delete_split_dataset(split):
    final_dir = f'datasets/{dataset}-{int(split*100)}'
    os.rmdir(final_dir)


split_dataset(float(input("Split:")))
