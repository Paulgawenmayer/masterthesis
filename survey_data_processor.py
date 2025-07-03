import os
import csv
import re

FIELD_SURVEY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'field_survey')
TRAINING_DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_datasets')

os.makedirs(TRAINING_DATASETS_DIR, exist_ok=True)

def safe_dirname(name):
    # Replace invalid characters for folder names
    return re.sub(r'[^\w\-_\. ]', '_', name)

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

def main():
    for root, dirs, files in os.walk(FIELD_SURVEY_DIR):
        if 'survey_results.csv' in files:
            csv_path = os.path.join(root, 'survey_results.csv')
            print(f'Processing: {csv_path}')
            process_survey_csv(csv_path)

if __name__ == '__main__':
    main()
