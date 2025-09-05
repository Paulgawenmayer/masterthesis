"""
This script renames folders and contained JPG files in a specified directory:
- Replaces spaces with underscores
- Removes double underscores
- Renames JPG files inside the folder to match the folder name
"""

import os
import sys

def normalize_name(name):
    """Replaces spaces with underscores and removes double underscores"""
    normalized = name.replace(" ", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized

def rename_folders_and_files(base_path, verbose=True):
    """
    Renames folders and contained JPG files in a directory:
    - Replaces spaces with underscores
    - Removes double underscores
    - Renames JPG files to match the folder name
    """
    if not os.path.exists(base_path):
        print(f"The path {base_path} does not exist!")
        return 0, 0

    renamed_folders = 0
    renamed_files = 0
    
    # List all folders in the specified path
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        
        # 1. Normalize the folder name
        new_folder_name = normalize_name(folder)
        if new_folder_name != folder:
            new_folder_path = os.path.join(base_path, new_folder_name)
            
            # Check if target folder already exists
            if os.path.exists(new_folder_path):
                if verbose:
                    print(f"Warning: Target folder '{new_folder_name}' already exists, skipping '{folder}'")
                continue
                
            if verbose:
                print(f"Renaming folder: '{folder}' -> '{new_folder_name}'")
            os.rename(folder_path, new_folder_path)
            folder_path = new_folder_path
            renamed_folders += 1
        elif verbose:
            print(f"Folder name '{folder}' remains unchanged")
        
        # 2. Rename JPG files in the folder
        jpg_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
        
        for jpg_file in jpg_files:
            jpg_path = os.path.join(folder_path, jpg_file)
            new_jpg_name = f"{new_folder_name}.jpg"
            new_jpg_path = os.path.join(folder_path, new_jpg_name)
            
            if jpg_file.lower() != new_jpg_name.lower():  # Case-insensitive check
                if verbose:
                    print(f"  Renaming file: '{jpg_file}' -> '{new_jpg_name}'")
                # If target file already exists, delete it
                if os.path.exists(new_jpg_path) and jpg_path.lower() != new_jpg_path.lower():
                    os.remove(new_jpg_path)
                os.rename(jpg_path, new_jpg_path)
                renamed_files += 1
            elif verbose:
                print(f"  File name '{jpg_file}' remains unchanged")
    
    return renamed_folders, renamed_files

def run_interactive():
    """Runs the program in interactive mode"""
    
    # Path input prompt
    print("=== Folder and File Name Corrector ===")
    path = input("Enter the relative or absolute path to the directory\n(ENTER for directory): ")
    
    if not path:
        path = "."
        
    # Resolve path
    abs_path = os.path.abspath(path)
    
    # Ask about verbose mode
    verbose_input = input("Show detailed output? (y/n, default: y): ")
    verbose = verbose_input.lower() not in ('n', 'no', '0')
    
    print(f"\nProcessing folder: {abs_path}")
    renamed_folders, renamed_files = rename_folders_and_files(abs_path, verbose=verbose)
    print(f"\nDone! Renamed {renamed_folders} folders and {renamed_files} files.")

# When the script is executed directly (not imported)
if __name__ == "__main__":
    run_interactive()