import os
import csv
from transform_address_to_coordinates import get_coordinates

FIELD_SURVEY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'field_survey')

def fill_missing_coordinates(csv_path):
    rows = []
    changed = False
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',')
        fieldnames = reader.fieldnames
        for row in reader:
            coords = row.get('Koordinaten') or row.get('koordinaten')
            address = row.get('Adresse') or row.get('adresse')
            if (not coords or coords.strip() == '') and address:
                latlon = get_coordinates(address)
                if latlon:
                    row['Koordinaten'] = f"{latlon[0]}, {latlon[1]}"
                    changed = True
            rows.append(row)
    if changed:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Updated: {csv_path}")
    else:
        print(f"No changes needed: {csv_path}")

def main():
    for root, dirs, files in os.walk(FIELD_SURVEY_DIR):
        if 'survey_results.csv' in files:
            csv_path = os.path.join(root, 'survey_results.csv')
            fill_missing_coordinates(csv_path)

if __name__ == '__main__':
    main()
