"""
Created on Tue Jun 10 12:19:02 2025

@author: paulmayer

This script scans a specified directory for image files and removes duplicates based on their content.
It uses a hash function to identify duplicates, ensuring that only unique images remain in the directory.
"""

import os
import hashlib

def delete_duplicate_images(output_dir):
    """
    Scans the given directory for image files and removes duplicates based on file content.
    Prints the number of deleted duplicate images.
    """
    image_hashes = {}
    deleted_images = 0

    def compute_hash(path):
        hasher = hashlib.md5()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(4096), b""):
                hasher.update(block)
        return hasher.hexdigest()

    for filename in os.listdir(output_dir):
        filepath = os.path.join(output_dir, filename)
        if os.path.isfile(filepath) and filename.lower().endswith((
            '.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            hash_value = compute_hash(filepath)
            if hash_value in image_hashes:
                os.remove(filepath)
                print(f"Duplicate deleted: {filepath}")
                deleted_images += 1
            else:
                image_hashes[hash_value] = filepath

    print(f"\n✅ Process completed. {deleted_images} duplicate images were deleted.")

if __name__ == "__main__":
    dir_path = input("Please enter the path to the directory to scan for duplicate images: ").strip()
    if not os.path.isdir(dir_path):
        print(f"❌ The path '{dir_path}' is not a valid directory.")
    else:
        delete_duplicate_images(dir_path)