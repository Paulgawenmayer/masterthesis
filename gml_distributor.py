"""
This script distributes GML files from field_survey/Object_GMLs to corresponding folders 
in training_datasets/colored based on matching names.

For each .gml file in the source directory, it checks if there's a folder with the same name
(without the .gml extension) in the target directory and copies the GML file into that folder.
"""

import os
import shutil

def distribute_gml_files():
    # Define base directory and paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(base_dir, "field_survey", "Object_GMLs")
    target_base_dir = os.path.join(base_dir, "training_datasets", "colored")
    
    # Check if directories exist
    if not os.path.exists(source_dir):
        print(f"Source directory does not exist: {source_dir}")
        return
    
    if not os.path.exists(target_base_dir):
        print(f"Target directory does not exist: {target_base_dir}")
        return
    
    # Count statistics
    total_gml_files = 0
    copied_files = 0
    skipped_files = 0
    
    # Find all GML files in source directory and subdirectories
    for root, _, files in os.walk(source_dir):
        for filename in files:
            if filename.lower().endswith('.gml'):
                total_gml_files += 1
                
                # Get the name without extension (to match folder names)
                name_without_ext = os.path.splitext(filename)[0]
                source_file_path = os.path.join(root, filename)
                
                # Look for matching folder in target directory
                target_folder = os.path.join(target_base_dir, name_without_ext)
                
                if os.path.exists(target_folder) and os.path.isdir(target_folder):
                    # Folder exists, copy the file
                    target_file_path = os.path.join(target_folder, filename)
                    try:
                        shutil.copy2(source_file_path, target_file_path)
                        print(f"Copied: {filename} -> {os.path.join(name_without_ext, filename)}")
                        copied_files += 1
                    except Exception as e:
                        print(f"Error copying {filename}: {str(e)}")
                        skipped_files += 1
                else:
                    print(f"No matching folder for: {filename}")
                    skipped_files += 1
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Total GML files found: {total_gml_files}")
    print(f"Files copied: {copied_files}")
    print(f"Files skipped: {skipped_files}")

if __name__ == "__main__":
    print("Starting GML file distribution...")
    distribute_gml_files()
    print("Done!")