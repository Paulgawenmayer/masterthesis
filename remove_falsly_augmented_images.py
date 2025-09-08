"""
This script removes all augmented images from a directory and its subdirectories.
Augmented images are identified by having 'augXX' in their filename, where XX is any number.  
The usage of this script is not crucial for the workflow and should be seen as an additional tool, 
in case the user has done some manual augmentation and wants to clean up the directory.
"""

import os
import re
import argparse
from tqdm import tqdm

def remove_augmented_images(root_dir):
    """
    Recursively removes all augmented images from the specified directory
    and its subdirectories.
    
    Args:
        root_dir (str): Path to the root directory to process
    
    Returns:
        tuple: (number of files removed, total size freed in bytes)
    """
    if not os.path.exists(root_dir):
        print(f"Error: Directory '{root_dir}' does not exist.")
        return 0, 0
    
    # Regular expression to match "augXX" where XX is any number
    aug_pattern = re.compile(r'aug\d+', re.IGNORECASE)
    
    files_removed = 0
    total_size_freed = 0
    
    # Find all image files recursively
    all_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                all_files.append(os.path.join(root, file))
    
    # Filter and remove augmented images
    if len(all_files) == 0:
        print(f"No image files found in '{root_dir}' or its subdirectories.")
        return 0, 0
    
    print(f"Scanning {len(all_files)} image files for augmented versions...")
    
    # Use tqdm for a nice progress bar
    for file_path in tqdm(all_files, desc="Removing augmented images"):
        base_name = os.path.basename(file_path)
        
        # Check if the filename contains "augXX"
        if aug_pattern.search(base_name):
            try:
                # Get file size before removal
                file_size = os.path.getsize(file_path)
                
                # Remove the file
                os.remove(file_path)
                
                files_removed += 1
                total_size_freed += file_size
            except Exception as e:
                print(f"Error removing {file_path}: {str(e)}")
    
    return files_removed, total_size_freed

def format_size(size_bytes):
    """Format bytes into a human-readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ("B", "KB", "MB", "GB", "TB")
    i = 0
    while size_bytes >= 1024 and i < len(size_names)-1:
        size_bytes /= 1024
        i += 1
        
    return f"{size_bytes:.2f} {size_names[i]}"

def main():
    parser = argparse.ArgumentParser(description='Remove augmented images from a directory and its subdirectories.')
    parser.add_argument('directory', nargs='?', help='Directory to process')
    args = parser.parse_args()
    
    # Get directory from command line or prompt
    directory = args.directory
    if not directory:
        directory = input("Enter directory path to process: ")
    
    # Confirm operation
    print(f"This will remove all augmented images (containing 'augXX' in filename) from:")
    print(f"  {os.path.abspath(directory)}")
    print("and all its subdirectories.")
    
    confirm = input("Do you want to proceed? (y/n): ")
    if confirm.lower() != 'y':
        print("Operation cancelled.")
        return
    
    # Remove augmented images
    files_removed, size_freed = remove_augmented_images(directory)
    
    # Show results
    print("\nOperation completed:")
    print(f"  {files_removed} augmented image files removed")
    print(f"  {format_size(size_freed)} disk space freed")

if __name__ == "__main__":
    main()