"""
This script copies the training datasets from 'training_datasets' to 'selective_training_datasets',
filters the datasets based on user-defined criteria, and allows the user to choose between filtering by image,
by attributes in the address data CSV files, or by specific file conditions. 

Filtering by image: Removes directories that do not contain any images from the 'images' directory, 
while filtering by attributes checks for specific criteria in the CSV files such as insulation methods and glazing types.

Filtering by attributes: Checks for specific criteria in the CSV files such as insulation methods and glazing types.
Only directories that meet the criteria will be kept in the 'colored' and 'BW' directories.

Filtering by condition: Checks for the presence of specific files (e.g., "StreetView.png") in each directory.
Only directories that contain the required files will be kept in the output directories.

This script is designed to be run in a directory structure where the 'training_datasets' directory
is located at the same level as this script.    
"""


import os
import shutil
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DATASETS = os.path.join(BASE_DIR, 'training_datasets')
DST_DATASETS = os.path.join(BASE_DIR, 'selective_training_datasets')

# --- Progress bar for copying training_datasets ---
def copy_with_progress(src, dst):
    # Collect all files to copy
    files_to_copy = []
    for root, dirs, files in os.walk(src):
        for file in files:
            files_to_copy.append(os.path.join(root, file))
    total_files = len(files_to_copy)
    # print("total files to copy:", total_files)
    for idx, src_file in enumerate(files_to_copy, 1):
        rel_path = os.path.relpath(src_file, src)
        dst_file = os.path.join(dst, rel_path)
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        shutil.copy2(src_file, dst_file)
        percent = int((idx / total_files) * 100)
        bar = ('#' * (percent // 2)).ljust(50)
        print(f"Copying training_datasets: |{bar}| {percent}% ({idx}/{total_files})", end='\r' if idx < total_files else '\n')
    print("Copy complete.")

# Remove and copy with progress bar
if os.path.exists(DST_DATASETS):
    shutil.rmtree(DST_DATASETS)
os.makedirs(DST_DATASETS, exist_ok=True)
copy_with_progress(SRC_DATASETS, DST_DATASETS)

COLORED = os.path.join(DST_DATASETS, 'colored')
BW = os.path.join(DST_DATASETS, 'BW')
IMAGES = os.path.join(DST_DATASETS, 'images')

image_filenames = set(os.listdir(IMAGES)) if os.path.exists(IMAGES) else set()


#this function filters the datasets based on the presence of images
# in the 'images' directory. It removes any subdirectory in both COLORED and BW directories
# that does not contain any image files that match the filenames in the 'images'
# directory. This is useful to ensure that only datasets with corresponding images are kept.
def choose_by_image():
    for dataset_dir in [COLORED, BW]:
        if not os.path.exists(dataset_dir):
            continue
        subdirs = [subdir for subdir in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, subdir))]
        total = len(subdirs)
        for idx, subdir in enumerate(subdirs, 1):
            subdir_path = os.path.join(dataset_dir, subdir)
            # Check if any image in subdir matches one in images
            found = False
            for file in os.listdir(subdir_path):
                if file in image_filenames:
                    found = True
                    break
            if not found:
                shutil.rmtree(subdir_path)
                print(f"Removed {subdir_path} (no matching image)")
            # Progress bar (global for filtering)
            percent = int((idx / total) * 100)
            bar = ('#' * (percent // 2)).ljust(50)
            print(f"Filtering Progress: |{bar}| {percent}% ({idx}/{total}) [{dataset_dir}]", end='\r' if idx < total else '\n')



def choose_by_attributes():
    allowed_criteria = [
        "Aufsparrendämmung?", "Dach gedämmt?", "Fassadendämmung", "Sockeldämmung", "Fenster fassadenbündig", "Kommentar",
        "Einfachverglasung", "Zweifachverglasung", "Dreifachverglasung", "Dach saniert?"
    ]
    glazing_types = ["Einfachverglasung", "Zweifachverglasung", "Dreifachverglasung"]
    criteria_str = input("Select criteria (separated by comma): Aufsparrendämmung?, Dach gedämmt?, Fassadendämmung, Sockeldämmung, Fenster fassadenbündig, Kommentar, Einfachverglasung, Zweifachverglasung, Dreifachverglasung, Dach saniert?)\n> ")
    if not criteria_str:
        print("Error: no criteria given")
        return
    criteria = [c.strip() for c in criteria_str.split(',') if c.strip() and c.strip() in allowed_criteria]
    if not criteria:
        print("Error: no valid criteria.")
        return
    # Filtering progress bar
    for dataset_dir in [COLORED, BW]:
        if not os.path.exists(dataset_dir):
            continue
        subdirs = [subdir for subdir in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, subdir))]
        total = len(subdirs)
        for idx, subdir in enumerate(subdirs, 1):
            subdir_path = os.path.join(dataset_dir, subdir)
            address_csv = os.path.join(subdir_path, 'address_data.csv')
            if not os.path.exists(address_csv):
                print(f"address_data.csv NOT FOUND in {subdir_path}")
                shutil.rmtree(subdir_path)
                print(f"Removed {subdir_path} (no address_data.csv)")
                # Filtering progress bar
                percent = int((idx / total) * 100)
                bar = ('#' * (percent // 2)).ljust(50)
                print(f"Filtering Progress: |{bar}| {percent}% ({idx}/{total}) [{dataset_dir}]", end='\r' if idx < total else '\n')
                continue
            else:
                print(f"address_data.csv FOUND in {subdir_path}")
            try:
                df = pd.read_csv(address_csv, delimiter=';')
            except Exception:
                shutil.rmtree(subdir_path)
                print(f"Removed {subdir_path} (CSV read error)")
                percent = int((idx / total) * 100)
                bar = ('#' * (percent // 2)).ljust(50)
                print(f"Filtering Progress: |{bar}| {percent}% ({idx}/{total}) [{dataset_dir}]", end='\r' if idx < total else '\n')
                continue
            if df.empty:
                shutil.rmtree(subdir_path)
                print(f"Removed {subdir_path} (empty CSV)")
                percent = int((idx / total) * 100)
                bar = ('#' * (percent // 2)).ljust(50)
                print(f"Filtering Progress: |{bar}| {percent}% ({idx}/{total}) [{dataset_dir}]", end='\r' if idx < total else '\n')
                continue
            row = df.iloc[0]
            all_ok = True
            for crit in criteria:
                if crit == "Kommentar":
                    if "Kommentar" not in df.columns or pd.isnull(row["Kommentar"]) or str(row["Kommentar"]).strip() == '' or str(row["Kommentar"]).lower() == 'nan':
                        all_ok = False
                        break
                elif crit in glazing_types:
                    # Check if the glazing type is present in the column 'Verglasungstyp'
                    if "Verglasungstyp" not in df.columns or str(row["Verglasungstyp"]).strip().lower() != crit.lower():
                        all_ok = False
                        break
                else:
                    if crit not in df.columns or str(row[crit]).strip().lower() != 'checked':
                        all_ok = False
                        break
            if not all_ok:
                shutil.rmtree(subdir_path)
                print(f"Removed {subdir_path} (criteria not met)")
            percent = int((idx / total) * 100)
            bar = ('#' * (percent // 2)).ljust(50)
            print(f"Filtering Progress: |{bar}| {percent}% ({idx}/{total}) [{dataset_dir}]", end='\r' if idx < total else '\n')

    # Cleanup progress bar
    print("\nCleaning up images folder...")
    colored_images = set()
    if os.path.exists(COLORED):
        for subdir in os.listdir(COLORED):
            subdir_path = os.path.join(COLORED, subdir)
            if os.path.isdir(subdir_path):
                for file in os.listdir(subdir_path):
                    colored_images.add(file)
    if os.path.exists(IMAGES):
        images_list = list(os.listdir(IMAGES))
        total_cleanup = len(images_list)
        for idx, img_file in enumerate(images_list, 1):
            if img_file not in colored_images:
                img_path = os.path.join(IMAGES, img_file)
                os.remove(img_path)
                print(f"Removed {img_path} (not in any colored subfolder)")
            percent = int((idx / total_cleanup) * 100)
            bar = ('#' * (percent // 2)).ljust(50)
            print(f"Cleanup Progress: |{bar}| {percent}% ({idx}/{total_cleanup})", end='\r' if idx < total_cleanup else '\n')
    print("Image cleanup complete.")

def choose_by_condition():
    """
    Filters the datasets based on the presence of a StreetView.png file in each directory.
    Only keeps directories that contain a StreetView.png file.
    """
    print("Filtering datasets by presence of StreetView.png...")
    
    for dataset_dir in [COLORED, BW]:
        if not os.path.exists(dataset_dir):
            continue
            
        subdirs = [subdir for subdir in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, subdir))]
        total = len(subdirs)
        
        for idx, subdir in enumerate(subdirs, 1):
            subdir_path = os.path.join(dataset_dir, subdir)
            
            # Check if any file in the directory ends with StreetView.png
            has_streetview = False
            for file in os.listdir(subdir_path):
                if file.endswith("StreetView.png"):
                    has_streetview = True
                    break
                    
            if not has_streetview:
                shutil.rmtree(subdir_path)
                print(f"Removed {subdir_path} (no StreetView.png file)")
                
            # Progress bar
            percent = int((idx / total) * 100)
            bar = ('#' * (percent // 2)).ljust(50)
            print(f"Filtering Progress: |{bar}| {percent}% ({idx}/{total}) [{dataset_dir}]", end='\r' if idx < total else '\n')
    
    # Cleanup images folder to match the filtered datasets
    print("\nCleaning up images folder...")
    colored_images = set()
    if os.path.exists(COLORED):
        for subdir in os.listdir(COLORED):
            subdir_path = os.path.join(COLORED, subdir)
            if os.path.isdir(subdir_path):
                for file in os.listdir(subdir_path):
                    colored_images.add(file)
                    
    if os.path.exists(IMAGES):
        images_list = list(os.listdir(IMAGES))
        total_cleanup = len(images_list)
        
        for idx, img_file in enumerate(images_list, 1):
            if img_file not in colored_images:
                img_path = os.path.join(IMAGES, img_file)
                os.remove(img_path)
                print(f"Removed {img_path} (not in any colored subfolder)")
                
            percent = int((idx / total_cleanup) * 100)
            bar = ('#' * (percent // 2)).ljust(50)
            print(f"Cleanup Progress: |{bar}| {percent}% ({idx}/{total_cleanup})", end='\r' if idx < total_cleanup else '\n')
            
    print("Image cleanup complete.")
    print("Datasets filtered by StreetView.png presence.")

def main():
    print("Choose mode: 'choose_by_image', 'choose_by_attributes', or 'choose_by_condition'")
    choice = input("> ").strip()
    if choice == 'choose_by_image':
        choose_by_image()
        print("Datasets filtered by image presence.")
    elif choice == 'choose_by_attributes':
        choose_by_attributes()
        print("Datasets filtered by attributes.")
    elif choice == 'choose_by_condition':
        choose_by_condition()
        print("Datasets filtered by StreetView.png presence.")
    else:
        print("Error: no valid choice.")

if __name__ == "__main__":
    main()