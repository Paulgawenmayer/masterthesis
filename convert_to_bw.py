"""
This script converts all images in the 'colored' directory to black and white (BW)
and saves them in the 'BW' directory. It must therefore be executed after survey_data_processor.
It preserves the directory structure and copies non-image
files as they are.
"""

import os
import shutil
from PIL import Image

COLORED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_datasets', 'colored')
BW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_datasets', 'BW')

os.makedirs(BW_DIR, exist_ok=True)

def convert_images_to_bw(src_dir, dst_dir):
    # Collect all files to process for progress bar
    all_files = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            all_files.append((root, file))
    total_files = len(all_files)
    for idx, (root, file) in enumerate(all_files, 1):
        rel_path = os.path.relpath(root, src_dir)
        target_dir = os.path.join(dst_dir, rel_path)
        os.makedirs(target_dir, exist_ok=True)
        src_file = os.path.join(root, file)
        dst_file = os.path.join(target_dir, file)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                img = Image.open(src_file).convert('L')
                img.save(dst_file)
                print(f"Converted to BW: {dst_file}")
            except Exception as e:
                print(f"Error converting {src_file}: {e}")
        else:
            shutil.copy2(src_file, dst_file)
            print(f"Copied (non-image): {dst_file}")
        # Progress bar
        percent = int((idx / total_files) * 100)
        bar = ('#' * (percent // 2)).ljust(50)
        print(f"Progress: |{bar}| {percent}% ({idx}/{total_files})", end='\r' if idx < total_files else '\n')

def main():
    convert_images_to_bw(COLORED_DIR, BW_DIR)

if __name__ == '__main__':
    main()
