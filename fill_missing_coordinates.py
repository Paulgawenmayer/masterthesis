"""
This script completes the missing coordinates in field survey CSV files (using transform_address_to_coordinates.py), 
which occurred, when added manually to the dataset in the field. It therefore is mandatory to complete the dataset. 
It should be executed before survey_data_processor.py.
"""
import os
import pandas as pd
from transform_address_to_coordinates import get_coordinates

def fill_missing_coordinates(base_dir="field_survey", target_file="survey_results.csv"):
    # Traverse all subdirectories
    files_to_process = []
    for root, dirs, files in os.walk(base_dir):
        if target_file in files:
            files_to_process.append(os.path.join(root, target_file))
    total_files = len(files_to_process)
    for i, file_path in enumerate(files_to_process, 1):
        try:
            df = pd.read_csv(file_path)
            # Check if columns exist
            if "Koordinaten" in df.columns and "Adresse" in df.columns:
                # Find rows with missing coordinates
                missing_mask = df["Koordinaten"].isna() | (df["Koordinaten"].astype(str).str.strip() == "")
                for idx in df[missing_mask].index:
                    address = df.at[idx, "Adresse"]
                    #print(f"{file_path} → Address without coordinates: {address}")
                    coords = get_coordinates(address)
                    if coords:
                        coord_str = f"{coords[0]}, {coords[1]}"
                        df.at[idx, "Koordinaten"] = coord_str
                        #print(f"  → Coordinates found: {coord_str}")
                    else:
                        print(f"  → No coordinates found!")
                # Save the updated CSV if any changes were made
                if missing_mask.any():
                    df.to_csv(file_path, index=False)
                    print(f"Updated: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        # Progress bar
        percent = int((i / total_files) * 100)
        bar = ('#' * (percent // 2)).ljust(50)
        print(f"Progress: |{bar}| {percent}% ({i}/{total_files})", end='\r' if i < total_files else '\n')

if __name__ == "__main__":
    fill_missing_coordinates()

