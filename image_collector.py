"""
This script collects all StreetView images from the 'colored' directory
and copies them to the 'images' directory. It enables the user to quickly view all StreetView images 
in the dataset and select those that are most useful for training.
"""

import os
import shutil

COLORED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_datasets', 'colored')
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_datasets', 'images')

os.makedirs(IMAGES_DIR, exist_ok=True)

# Target specific filename ending
TARGET_SUFFIX = "StreetView.png"

def collect_images(src_dir, dst_dir):
    # Collect all StreetView image files for progress bar
    all_images = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(TARGET_SUFFIX):
                all_images.append((root, file))
    
    total_images = len(all_images)
    
    if total_images == 0:
        print(f"No files with suffix '{TARGET_SUFFIX}' found in {src_dir}")
        return
        
    print(f"Found {total_images} StreetView images to copy")
    
    for idx, (root, file) in enumerate(all_images, 1):
        src_file = os.path.join(root, file)
        dst_file = os.path.join(dst_dir, file)
        try:
            shutil.copy2(src_file, dst_file)
            print(f"Copied: {file}")
        except Exception as e:
            print(f"Error copying {file}: {e}")
        # Progress bar
        percent = int((idx / total_images) * 100)
        bar = ('#' * (percent // 2)).ljust(50)
        print(f"Progress: |{bar}| {percent}% ({idx}/{total_images})", end='\r' if idx < total_images else '\n')
    
    print(f"\nSuccessfully copied {total_images} StreetView images to {dst_dir}")

def main():
    collect_images(COLORED_DIR, IMAGES_DIR)

if __name__ == '__main__':
    main()