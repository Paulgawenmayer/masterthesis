"""
This script allows interactive editing of the insulation attribute 'Aufsparrendämmung' for each address in the training dataset.
It displays each image in 'training_datasets/colored/<address>' using matplotlib, asks the user for input, and updates the corresponding CSV files.

- if __name__ == "__main__": is not nesessary, as the script runs on import and does not contain any functions that need to be called directly.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import shutil
import subprocess
import locale

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COLORED_DIR = os.path.join(BASE_DIR, 'training_datasets', 'colored')
FIELD_SURVEY_DIR = os.path.join(BASE_DIR, 'field_survey')

# Set German locale for sorting
locale.setlocale(locale.LC_COLLATE, 'de_DE.UTF-8')

for subdir in sorted(os.listdir(COLORED_DIR), key=locale.strxfrm):
    subdir_path = os.path.join(COLORED_DIR, subdir)
    if not os.path.isdir(subdir_path):
        continue
    # Format address name
    name_directory = subdir.replace('_', ',')
    name_directory = ', '.join([part.strip() for part in name_directory.split(',')])
    print(name_directory)
    # Show all jpg images in subdir
    for file in os.listdir(subdir_path):
        if file.lower().endswith('.jpg'):
            img_path = os.path.join(subdir_path, file)
            img = Image.open(img_path)
            fig = plt.figure(figsize=(12, 9))  # Feste große Fenstergröße
            plt.imshow(img)
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Maximiert die Bildfläche
            plt.tight_layout()
            plt.title(name_directory)
            plt.show(block=False)
            # User input erscheint sofort, während das Bild offen ist
            answer = input('Aufsparrendämmung y/n ? ')
            plt.close(fig)
            if answer.strip().lower() == 'y':
                # Update address_data.csv in subdir
                csv_path = os.path.join(subdir_path, 'address_data.csv')
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path, delimiter=';')
                    if 'Aufsparrendämmung?' in df.columns:
                        df['Aufsparrendämmung?'] = 'checked'
                        df.to_csv(csv_path, index=False, sep=';')
                # Update survey_results.csv in field_survey subfolders
                for survey_subdir in os.listdir(FIELD_SURVEY_DIR):
                    survey_subdir_path = os.path.join(FIELD_SURVEY_DIR, survey_subdir)
                    survey_csv = os.path.join(survey_subdir_path, 'survey_results.csv')
                    if os.path.exists(survey_csv):
                        try:
                            df_survey = pd.read_csv(survey_csv)
                            # Find row by address
                            mask = df_survey['Adresse'].astype(str).str.strip() == name_directory
                            if mask.any() and 'Aufsparrendämmung?' in df_survey.columns:
                                df_survey.loc[mask, 'Aufsparrendämmung?'] = 'checked'
                                df_survey.to_csv(survey_csv, index=False)
                        except Exception as e:
                            print(f'Error updating {survey_csv}: {e}')
            # If 'n', continue to next image/subdir
            break  # Only one image per subdir shown

# After all subdirectories have been processed
BW_DIR = os.path.join(BASE_DIR, 'training_datasets', 'BW')
if os.path.exists(BW_DIR):
    shutil.rmtree(BW_DIR)
    print(f"Deleted folder: {BW_DIR}")

convert_script = os.path.join(BASE_DIR, 'convert_to_bw.py')
if os.path.exists(convert_script):
    print("Running convert_to_bw.py...")
    result = subprocess.run([os.sys.executable, convert_script])
    if result.returncode == 0:
        print("convert_to_bw.py completed successfully.")
    else:
        print("Error running convert_to_bw.py.")
else:
    print("convert_to_bw.py not found.")


