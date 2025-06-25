"""
Created on Tue Jun 25 15:19:02 2025

@author: paulmayer
This script performs data augmentation on images in a specified directory.
It applies various transformations such as rotation, flipping, brightness/contrast adjustment,
and cropping to generate augmented images. The number of augmentations per image can be specified.
"""
import os
import cv2
import numpy as np
import albumentations as A

def data_augmentator(target_dir, n_aug=10):
   
    image_files = [f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"No images found in {target_dir}.")
        return

    transform = A.Compose([
        A.Rotate(limit=40, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=0, p=0.5),
        A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), p=0.3)
    ])

    for img_file in image_files:
        img_path = os.path.join(target_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not read {img_path}, skipping.")
            continue
        for i in range(1, n_aug + 1):
            augmented = transform(image=image)['image']
            out_name = f"{os.path.splitext(img_file)[0]}_aug{i}.jpg"
            cv2.imwrite(os.path.join(target_dir, out_name), augmented)
    print('✅ Augmentation completed.')

if __name__ == "__main__":
    dir_path = input("Please enter the path to the directory containing images for augmentation: ").strip()
    if not os.path.isdir(dir_path):
        print(f"❌ The path '{dir_path}' is not a valid directory.")
    else:
        data_augmentator(dir_path)