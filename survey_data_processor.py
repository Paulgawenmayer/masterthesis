""" 
This script processes field survey_result CSV files, extracts image URLs from the 'Bild' column,
downloads the images, and organizes the data into structured directories for training datasets. 
Thereby, it creates a directory for each unique address found in the CSV files,
and saves the corresponding data and images in that directory.
"""

import os
import csv
import re
import requests
from fill_missing_coordinates import fill_missing_coordinates

FIELD_SURVEY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'field_survey')
TRAINING_DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_datasets', 'colored')

os.makedirs(TRAINING_DATASETS_DIR, exist_ok=True)

def safe_dirname(name):
    # Replace invalid characters for folder names
    return re.sub(r'[^\w\-_\. ]', '_', name)

def extract_image_url(bild_value):
    # Extract URL from the Bild column (format: 'date_time.jpg (URL)')
    if not bild_value:
        return None, None
    match = re.match(r'(.*?)\s*\((https?://[^)]+)\)', bild_value)
    if match:
        filename = match.group(1).strip()
        url = match.group(2).strip()
        return filename, url
    return None, None

def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded image to {save_path}")
        else:
            print(f"Failed to download image: {url} (Status: {response.status_code})")
    except Exception as e:
        print(f"Error downloading image: {url} ({e})")

def process_survey_csv(csv_path):
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            address = row.get('Adresse') or row.get('Adresse'.lower()) or 'unknown_address'
            folder_name = safe_dirname(address)
            address_dir = os.path.join(TRAINING_DATASETS_DIR, folder_name)
            os.makedirs(address_dir, exist_ok=True)
            # Write a small CSV for this address
            address_csv_path = os.path.join(address_dir, 'address_data.csv')
            with open(address_csv_path, 'w', newline='', encoding='utf-8') as out_csv:
                writer = csv.DictWriter(out_csv, fieldnames=row.keys(), delimiter=';')
                writer.writeheader()
                writer.writerow(row)
            # Download image if link exists in 'Bild' column
            bild_value = row.get('Bild') or row.get('bild')
            filename, url = extract_image_url(bild_value)
            if url:
                # Use the folder name as the image filename
                image_filename = f"{folder_name}.jpg"
                image_save_path = os.path.join(address_dir, image_filename)
                download_image(url, image_save_path)

def main():
    # Fill missing coordinates in the CSVs first
    print("Filling missing coordinates in address lists...")
    fill_missing_coordinates()
    print("Coordinate completion finished.")
    # Collect all survey_results.csv files for progress bar
    csv_files = []
    for root, dirs, files in os.walk(FIELD_SURVEY_DIR):
        if 'survey_results.csv' in files:
            csv_files.append(os.path.join(root, 'survey_results.csv'))
    total_files = len(csv_files)
    for i, csv_path in enumerate(csv_files, 1):
        print(f'Processing: {csv_path}')
        process_survey_csv(csv_path)
        # Progress bar
        percent = int((i / total_files) * 100)
        bar = ('#' * (percent // 2)).ljust(50)
        print(f"Progress: |{bar}| {percent}% ({i}/{total_files})", end='\r' if i < total_files else '\n')

if __name__ == '__main__':
    main()
