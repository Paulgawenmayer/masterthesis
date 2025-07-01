import csv
import re
import os

def extract_street_and_number(address):
    match = re.match(r"^(.*?)(?:\s+(\d+[a-zA-Z]?))?(,|$)", address)
    if match:
        street = match.group(1).strip() if match.group(1) else ''
        number = match.group(2) if match.group(2) else ''
        return street, number
    return '', ''

def house_number_key(num):
    if not num:
        return float('inf')
    match = re.match(r"(\d+)", str(num))
    if match:
        return int(match.group(1))
    return float('inf')

def sort_addresses_csv(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)
    # Remove duplicates
    seen = set()
    unique_rows = []
    for row in rows:
        key = (row['coordinates'], row['address'])
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)
    # Sort by street and house number
    unique_rows.sort(key=lambda row: (
        extract_street_and_number(row['address'])[0],
        house_number_key(extract_street_and_number(row['address'])[1])
    ))
    # Write sorted rows back
    with open(csv_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=['coordinates', 'address'])
        writer.writeheader()
        for row in unique_rows:
            writer.writerow(row)

if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "addresses_in_area.csv")
    sort_addresses_csv(csv_path)
    print(f"Sorted addresses in {csv_path}")
