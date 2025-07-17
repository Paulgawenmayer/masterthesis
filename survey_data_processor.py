"""
Created on Tue Jun 10 12:19:02 2025

@author: paulmayer

This script processes field survey_result CSV files, extracts image URLs from the 'Bild' column,
downloads the images, and organizes the data into structured directories for training datasets. 
Thereby, it creates a directory for each unique address found in the CSV files,
and saves the corresponding data and images in that directory.
"""

import os
import csv
import re
import requests

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
                writer = csv.DictWriter(out_csv, fieldnames=row.keys())
                writer.writeheader()
                writer.writerow(row)
            # Download image if link exists in 'Bild' column
            bild_value = row.get('Bild') or row.get('bild')
            filename, url = extract_image_url(bild_value)
            if url:
                # Use the filename from the CSV, or fallback to 'image.jpg'
                image_filename = filename if filename else 'image.jpg'
                image_save_path = os.path.join(address_dir, image_filename)
                download_image(url, image_save_path)

def main():
    for root, dirs, files in os.walk(FIELD_SURVEY_DIR):
        if 'survey_results.csv' in files:
            csv_path = os.path.join(root, 'survey_results.csv')
            print(f'Processing: {csv_path}')
            process_survey_csv(csv_path)

if __name__ == '__main__':
    main()
