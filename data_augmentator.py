"""
Data augmentation script that creates modified versions of image files.
It can process a directory structure recursively, applying augmentations to
all image files in all subdirectories.
"""

import os
import cv2
import albumentations as A
import numpy as np
import argparse
from tqdm import tqdm

def data_augmentator(target_dir, n_aug=10, recursive=True):
    """
    Augments image files in the target directory and optionally its subdirectories.
    
    Parameters:
    - target_dir: Directory containing images to augment
    - n_aug: Number of augmented images to create per original image
    - recursive: If True, process subdirectories recursively
    
    Returns:
    - Dictionary with statistics about the augmentation process
    """
    # Define augmentation pipeline
    transform = A.Compose([
        #A.Rotate(limit=40, p=0.7), # prevent rotation
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.0, rotate_limit=0, p=0.5), # prevent scaling and rotation
        #A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), p=0.3) # prevent cropping
    ])
    
    # Statistics dictionary
    stats = {
        "directories_processed": 0,
        "images_found": 0,
        "augmentations_created": 0,
        "errors": 0
    }
    
    # Process the current directory
    _process_directory(target_dir, transform, n_aug, stats)
    
    # If recursive, process all subdirectories
    if recursive:
        for root, dirs, _ in os.walk(target_dir):
            for dir_name in dirs:
                subdir_path = os.path.join(root, dir_name)
                _process_directory(subdir_path, transform, n_aug, stats)
    
    return stats

def _process_directory(directory, transform, n_aug, stats):
    """Helper function to process a single directory"""
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return
    
    # Update stats
    stats["directories_processed"] += 1
    
    # Get all image files in the directory
    image_files = []
    for file in os.listdir(directory):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(directory, file))
    
    stats["images_found"] += len(image_files)
    
    if not image_files:
        print(f"No image files found in {directory}")
        return
    
    print(f"Processing {len(image_files)} images in {directory}")
    
    # Process each image file
    for img_path in tqdm(image_files, desc="Augmenting images"):
        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read image: {img_path}")
                stats["errors"] += 1
                continue
                
            # Convert to RGB (albumentations expects RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get file name and extension
            file_name, file_ext = os.path.splitext(img_path)
            
            # Create n_aug augmented images
            for i in range(n_aug):
                try:
                    # Apply augmentation
                    augmented = transform(image=img)
                    augmented_img = augmented["image"]
                    
                    # Convert back to BGR for saving
                    augmented_img = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR)
                    
                    # Save augmented image
                    output_path = f"{file_name}_aug{i+1}{file_ext}"
                    cv2.imwrite(output_path, augmented_img)
                    stats["augmentations_created"] += 1
                    
                except Exception as e:
                    print(f"Error augmenting {img_path} (augmentation {i+1}): {e}")
                    stats["errors"] += 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            stats["errors"] += 1
    
def main():
    """Main function when script is run directly"""
    parser = argparse.ArgumentParser(description="Data augmentation for images")
    parser.add_argument("--dir", type=str, help="Directory containing images to augment")
    parser.add_argument("--n_aug", type=int, default=10, help="Number of augmented images to create per original")
    parser.add_argument("--no-recursive", action="store_true", help="Don't process subdirectories")
    args = parser.parse_args()
    
    # If no directory is specified, ask for input
    target_dir = args.dir
    if not target_dir:
        target_dir = input("Enter the directory path containing images to augment: ")
    
    # Get number of augmentations if not provided
    n_aug = args.n_aug
    if not args.dir:  # If we're in interactive mode, also ask for n_aug
        n_aug_input = input(f"Enter the number of augmentations per image (default: {n_aug}): ")
        if n_aug_input.strip():
            n_aug = int(n_aug_input)
    
    # Process directory
    recursive = not args.no_recursive
    stats = data_augmentator(target_dir, n_aug, recursive)
    
    # Print summary
    print("\n=== Augmentation Summary ===")
    print(f"Directories processed: {stats['directories_processed']}")
    print(f"Original images found: {stats['images_found']}")
    print(f"Augmented images created: {stats['augmentations_created']}")
    print(f"Errors encountered: {stats['errors']}")
    
if __name__ == "__main__":
    main()