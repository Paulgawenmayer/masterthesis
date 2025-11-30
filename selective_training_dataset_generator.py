"""
This script copies the training datasets from 'training_datasets' to 'selective_training_datasets',
filters the datasets based on user-defined criteria, and allows the user to choose between filtering by image,
by attributes in the address data CSV files, or by specific file conditions. 

Filtering by image: Removes individual image files from 'colored' and 'BW' directories that are not present
in the 'images' directory, while keeping the directory structure intact.

Filtering by attributes: Checks for specific criteria in the CSV files such as insulation methods and glazing types.
Only directories that meet the criteria will be kept in the 'colored' and 'BW' directories.

Filtering by condition: Checks for the presence of specific files (e.g., "StreetView.png") in each directory.
Only directories that contain the required files will be kept in the output directories.

Filtering by source: Removes directories from 'colored' and 'BW' that don't contain any images from 'images',
and removes all files except source images and CSV files from remaining directories.

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
# in the 'images' directory. It removes individual image files from 'colored' and 'BW'
# directories that are not present in the 'images' directory, while keeping the directory structure.
def choose_by_image():
    total_removed = 0
    total_files_checked = 0
    
    for dataset_dir in [COLORED, BW]:
        if not os.path.exists(dataset_dir):
            continue
        
        # Collect all image files to check
        files_to_check = []
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                # Skip CSV files and other non-image files
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    files_to_check.append(os.path.join(root, file))
        
        total = len(files_to_check)
        dataset_name = os.path.basename(dataset_dir)
        
        for idx, file_path in enumerate(files_to_check, 1):
            filename = os.path.basename(file_path)
            
            # Remove file if not in images directory
            if filename not in image_filenames:
                try:
                    os.remove(file_path)
                    total_removed += 1
                    print(f"Removed {file_path} (not in images directory)")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
            
            total_files_checked += 1
            
            # Progress bar
            percent = int((idx / total) * 100)
            bar = ('#' * (percent // 2)).ljust(50)
            print(f"Filtering [{dataset_name}]: |{bar}| {percent}% ({idx}/{total})", end='\r' if idx < total else '\n')
    
    print(f"\nFiltering complete. Checked {total_files_checked} files, removed {total_removed} files.")



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


def remove_data_from_training_dataset():
    """
    Removes directories and files from 'colored' and 'BW' based on images in 'images' folder.
    
    Logic:
    1. Get all image filenames from 'images' folder (source reference)
    2. For each subdirectory in 'colored' and 'BW':
       - Check if it contains at least one image from 'images' folder
       - If NO: Delete the entire subdirectory
       - If YES: Keep subdirectory but remove all files EXCEPT:
         * Images that are in 'images' folder
         * CSV files
    """
    print("\n" + "="*60)
    print("  REMOVE DATA FROM TRAINING DATASET")
    print("="*60)
    print("\nSource: 'images' folder (reference images)")
    print("Target: 'colored' and 'BW' folders")
    print("\nLogic:")
    print("  1. Remove directories WITHOUT any source images")
    print("  2. In remaining directories, keep ONLY:")
    print("     - Images from 'images' folder")
    print("     - CSV files")
    print("="*60)
    
    # Get source image names from 'images' folder
    if not os.path.exists(IMAGES):
        print(f"❌ Images folder does not exist: {IMAGES}")
        return
    
    source_image_names = set(os.listdir(IMAGES))
    print(f"\n✅ Found {len(source_image_names)} images in source folder")
    
    total_dirs_removed = 0
    total_dirs_kept = 0
    total_files_removed = 0
    total_files_kept = 0
    
    # Process each target directory
    for dataset_dir in [COLORED, BW]:
        if not os.path.exists(dataset_dir):
            print(f"⚠️  Directory does not exist: {dataset_dir}")
            continue
        
        dataset_name = os.path.basename(dataset_dir)
        print(f"\n📁 Processing: {dataset_name}")
        
        subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        total_subdirs = len(subdirs)
        
        if total_subdirs == 0:
            print(f"  No subdirectories found in {dataset_name}")
            continue
        
        print(f"  Found {total_subdirs} subdirectories")
        
        # Step 1: Check each subdirectory
        for idx, subdir in enumerate(subdirs, 1):
            subdir_path = os.path.join(dataset_dir, subdir)
            
            # Find images in subdirectory that match source
            found_source_images = set()
            all_files = []
            
            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)
                if os.path.isfile(file_path):
                    all_files.append(file)
                    if file in source_image_names:
                        found_source_images.add(file)
            
            # Progress bar
            percent = int((idx / total_subdirs) * 100)
            bar = ('#' * (percent // 2)).ljust(50)
            print(f"  Checking: |{bar}| {percent}% ({idx}/{total_subdirs})", end='\r')
            
            # Decision: Remove directory or clean it?
            if len(found_source_images) == 0:
                # No source images found -> Remove entire directory
                try:
                    shutil.rmtree(subdir_path)
                    total_dirs_removed += 1
                except Exception as e:
                    print(f"\n  ⚠️  Error removing directory {subdir}: {e}")
            else:
                # Source images found -> Keep directory, but clean files
                total_dirs_kept += 1
                files_removed = 0
                files_kept = 0
                
                for file in all_files:
                    file_path = os.path.join(subdir_path, file)
                    
                    # Keep file if:
                    # 1. It's in source images
                    # 2. It's a CSV file
                    should_keep = (
                        file in source_image_names or
                        file.lower().endswith('.csv')
                    )
                    
                    if should_keep:
                        files_kept += 1
                    else:
                        try:
                            os.remove(file_path)
                            files_removed += 1
                        except Exception as e:
                            print(f"\n  ⚠️  Error removing file {file}: {e}")
                
                total_files_removed += files_removed
                total_files_kept += files_kept
        
        print(f"\n  ✓ {dataset_name} processed")
    
    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"Directories removed: {total_dirs_removed}")
    print(f"Directories kept: {total_dirs_kept}")
    print(f"Files removed: {total_files_removed}")
    print(f"Files kept: {total_files_kept}")
    print("="*60)
    print("✅ Dataset cleaning complete!")


def main():
    print("Choose mode:")
    print("  1. 'choose_by_image' - Remove images not in 'images' folder")
    print("  2. 'choose_by_attributes' - Filter by CSV attributes")
    print("  3. 'choose_by_condition' - Filter by StreetView.png presence")
    print("  4. 'remove_data' - Remove dirs without source images + clean remaining dirs")
    choice = input("> ").strip()
    
    if choice == 'choose_by_image' or choice == '1':
        choose_by_image()
        print("Datasets filtered by image presence.")
    elif choice == 'choose_by_attributes' or choice == '2':
        choose_by_attributes()
        print("Datasets filtered by attributes.")
    elif choice == 'choose_by_condition' or choice == '3':
        choose_by_condition()
        print("Datasets filtered by StreetView.png presence.")
    elif choice == 'remove_data' or choice == '4':
        remove_data_from_training_dataset()
    else:
        print("Error: no valid choice.")

if __name__ == "__main__":
    main()