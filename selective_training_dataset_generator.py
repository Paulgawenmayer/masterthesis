"""
This script copies the training datasets from 'training_datasets' to 'selective_training_datasets',
filters the datasets based on user-defined criteria, and allows the user to choose between filtering by image
or by attributes in the address data CSV files. Filtering by image removes directories that do not contain any images
from the 'images' directory, while filtering by attributes checks for specific criteria in the CSV files
such as insulation methods and glazing types.
"""


import os
import shutil
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DATASETS = os.path.join(BASE_DIR, 'training_datasets')
DST_DATASETS = os.path.join(BASE_DIR, 'selective_training_datasets')

# Copy the whole training_datasets folder
if os.path.exists(DST_DATASETS):
    shutil.rmtree(DST_DATASETS)
shutil.copytree(SRC_DATASETS, DST_DATASETS)

COLORED = os.path.join(DST_DATASETS, 'colored')
BW = os.path.join(DST_DATASETS, 'BW')
IMAGES = os.path.join(DST_DATASETS, 'images')

# Helper: get all image filenames in images dir
image_filenames = set(os.listdir(IMAGES)) if os.path.exists(IMAGES) else set()

def choose_by_image():
    for dataset_dir in [COLORED, BW]:
        if not os.path.exists(dataset_dir):
            continue
        for subdir in os.listdir(dataset_dir):
            subdir_path = os.path.join(dataset_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            # Check if any image in subdir matches one in images
            found = False
            for file in os.listdir(subdir_path):
                if file in image_filenames:
                    found = True
                    break
            if not found:
                shutil.rmtree(subdir_path)
                print(f"Removed {subdir_path} (no matching image)")

def choose_by_attributes():
    allowed_criteria = [
        "Aufsparrendämmung", "Fassadendämmung", "Sockeldämmung", "Fenster fassadenbündig", "Kommentar",
        "Einfachverglasung", "Zweifachverglasung", "Dreifachverglasung", "Dach saniert?"
    ]
    glazing_types = ["Einfachverglasung", "Zweifachverglasung", "Dreifachverglasung"]
    criteria_str = input("Criteria (separated by comma: Aufsparrendämmung, Fassadendämmung, Sockeldämmung, Fenster fassadenbündig, Kommentar, Einfachverglasung, Zweifachverglasung, Dreifachverglasung, Dach saniert?)\n> ")
    if not criteria_str:
        print("Error: no criteria given")
        return
    criteria = [c.strip() for c in criteria_str.split(',') if c.strip() and c.strip() in allowed_criteria]
    if not criteria:
        print("Error: no valid criteria.")
        return
    for dataset_dir in [COLORED, BW]:
        if not os.path.exists(dataset_dir):
            continue
        for subdir in os.listdir(dataset_dir):
            subdir_path = os.path.join(dataset_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            address_csv = os.path.join(subdir_path, 'address_data.csv')
            if not os.path.exists(address_csv):
                print(f"address_data.csv NOT FOUND in {subdir_path}")
                shutil.rmtree(subdir_path)
                print(f"Removed {subdir_path} (no address_data.csv)")
                continue
            else:
                print(f"address_data.csv FOUND in {subdir_path}")
            try:
                df = pd.read_csv(address_csv, delimiter=';')
            except Exception:
                shutil.rmtree(subdir_path)
                print(f"Removed {subdir_path} (CSV read error)")
                continue
            if df.empty:
                shutil.rmtree(subdir_path)
                print(f"Removed {subdir_path} (empty CSV)")
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

def main():
    print("Choose mode: 'choose_by_image' oder 'choose_by_attributes'")
    choice = input("> ").strip()
    if choice == 'choose_by_image':
        choose_by_image()
        print("Datasets filtered by image presence.")
    elif choice == 'choose_by_attributes':
        choose_by_attributes()
        print("Datasets filtered by attributes.")
    else:
        print("Error: no valid choice.")

if __name__ == "__main__":
    main()
