"""
This script collects all images from the 'colored' directory
and copies them to the 'images' directory. In enables the user to quickly view all images in the dataset 
and select those that are most useful for training.
"""

import os
import shutil

COLORED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_datasets', 'colored')
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_datasets', 'images')

os.makedirs(IMAGES_DIR, exist_ok=True)

IMAGE_EXTENSIONS = ('.jpg')

def collect_images(src_dir, dst_dir):
    # Collect all image files for progress bar
    all_images = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith(IMAGE_EXTENSIONS):
                all_images.append((root, file))
    total_images = len(all_images)
    for idx, (root, file) in enumerate(all_images, 1):
        src_file = os.path.join(root, file)
        dst_file = os.path.join(dst_dir, file)
        try:
            shutil.copy2(src_file, dst_file)
            print(f"Copied: {src_file} -> {dst_file}")
        except Exception as e:
            print(f"Error copying {src_file}: {e}")
        # Progress bar
        percent = int((idx / total_images) * 100)
        bar = ('#' * (percent // 2)).ljust(50)
        print(f"Progress: |{bar}| {percent}% ({idx}/{total_images})", end='\r' if idx < total_images else '\n')

def main():
    collect_images(COLORED_DIR, IMAGES_DIR)

if __name__ == '__main__':
    main()
