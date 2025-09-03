"""
This script copies files from the "Downloads" folder to matching folders in the "training_datasets/colored" directory.

For each folder in "Downloads", the script checks if there's a matching folder with the same name
in "training_datasets/colored". If a match is found, all files from the source folder are copied
to the destination folder.
"""

import os
import shutil
import sys

def distribute_downloaded_data(source_base_dir, target_base_dir):
    """
    Copies files from source directories to matching target directories.
    
    Parameters:
    - source_base_dir: Path to the source directory containing subdirectories with downloaded files
    - target_base_dir: Path to the target directory where files should be copied to
    
    Returns:
    - Dictionary with statistics about the operation
    """
    # Check if directories exist
    if not os.path.exists(source_base_dir):
        print(f"Error: Source directory does not exist: {source_base_dir}")
        return None
    
    if not os.path.exists(target_base_dir):
        print(f"Error: Target directory does not exist: {target_base_dir}")
        return None
    
    # Statistics
    stats = {
        "total_source_folders": 0,
        "matched_folders": 0,
        "unmatched_folders": 0,
        "total_files": 0,
        "copied_files": 0,
        "skipped_files": 0,
        "error_files": 0
    }
    
    # Get list of all folders in source directory
    source_folders = [f for f in os.listdir(source_base_dir) 
                    if os.path.isdir(os.path.join(source_base_dir, f))]
    stats["total_source_folders"] = len(source_folders)
    
    # Get list of all folders in target directory
    target_folders = [f for f in os.listdir(target_base_dir) 
                     if os.path.isdir(os.path.join(target_base_dir, f))]
    
    # Process each source folder
    for folder_name in source_folders:
        source_folder_path = os.path.join(source_base_dir, folder_name)
        
        # Check if there's a matching folder in the target directory
        if folder_name in target_folders:
            stats["matched_folders"] += 1
            target_folder_path = os.path.join(target_base_dir, folder_name)
            
            # Get all files in the source folder
            files = [f for f in os.listdir(source_folder_path) 
                    if os.path.isfile(os.path.join(source_folder_path, f))]
            stats["total_files"] += len(files)
            
            print(f"Processing: {folder_name} ({len(files)} files)")
            
            # Copy each file
            for file_name in files:
                source_file_path = os.path.join(source_folder_path, file_name)
                target_file_path = os.path.join(target_folder_path, file_name)
                
                try:
                    # Check if target file already exists
                    if os.path.exists(target_file_path):
                        # If target is newer, skip copying
                        if os.path.getmtime(target_file_path) >= os.path.getmtime(source_file_path):
                            print(f"  Skipping (target is newer): {file_name}")
                            stats["skipped_files"] += 1
                            continue
                    
                    # Copy the file, preserving metadata
                    shutil.copy2(source_file_path, target_file_path)
                    print(f"  Copied: {file_name}")
                    stats["copied_files"] += 1
                    
                except Exception as e:
                    print(f"  Error copying {file_name}: {e}")
                    stats["error_files"] += 1
        else:
            stats["unmatched_folders"] += 1
            print(f"No matching folder found for: {folder_name}")
    
    return stats

def main():
    """Main function to execute the script"""
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define source and target directories
    source_dir = os.path.join(script_dir, "Downloads")
    target_dir = os.path.join(script_dir, "training_datasets", "colored")
    
    print(f"Source directory: {source_dir}")
    print(f"Target directory: {target_dir}")
    
    # Ask for confirmation
    confirmation = input("Do you want to proceed with copying files? (y/n): ").lower()
    if confirmation != 'y':
        print("Operation cancelled.")
        return
    
    # Distribute the data
    stats = distribute_downloaded_data(source_dir, target_dir)
    
    if stats:
        # Print summary
        print("\n=== Summary ===")
        print(f"Total source folders: {stats['total_source_folders']}")
        print(f"Matched folders: {stats['matched_folders']}")
        print(f"Unmatched folders: {stats['unmatched_folders']}")
        print(f"Total files processed: {stats['total_files']}")
        print(f"Files copied: {stats['copied_files']}")
        print(f"Files skipped: {stats['skipped_files']}")
        print(f"Files with errors: {stats['error_files']}")

if __name__ == "__main__":
    main()