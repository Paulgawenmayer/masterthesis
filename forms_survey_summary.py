"""
This script filters addresses from a CSV file, checking their validity using the Google Maps Geocoding API.
It saves valid addresses to a new CSV file in the 'survey_summary' directory, as only valid addresses can
be sensefully used in further research. 
It requires an API key for the Google Maps API, which should be stored in a 'config.py' file.

It further evaluates insulation methods from the filtered forms survey summary and saves the counts as insulation_methods_table.csv and a bar chart.
It adds average thickness rows for "Dachdämmung", "Fassadendämmung", and "Sockeldämmung".
It also creates additional bar charts for the insulation materials used in the facade, sockel, and roof.
"""

import os
import pandas as pd
import requests
from config import API_KEY

FIELD_SURVEY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'field_survey')
INPUT_CSV = os.path.join(FIELD_SURVEY_DIR, 'forms_survey_results.csv')
SUMMARY_DIR = os.path.join(FIELD_SURVEY_DIR, 'survey_summary')
OUTPUT_CSV = os.path.join(SUMMARY_DIR, 'forms_survey_summary.csv')
CHARTS_DIR = os.path.join(SUMMARY_DIR, 'charts')

os.makedirs(SUMMARY_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

ADDRESS_COLUMN = 'Wie lautet die Adresse Ihrer Immobilie [Straße, Hausnummer, PLZ, Ort]'


# Function to check if an address exists using Google Maps Geocoding API
def address_exists(address):
    url = 'https://maps.googleapis.com/maps/api/geocode/json'
    params = {'address': address, 'key': API_KEY}
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if response.status_code == 200 and data.get('status') == 'OK':
            result = data['results'][0]
            formatted = result.get('formatted_address', '')
            components = result.get('address_components', [])
            # Extract address parts
            street = None
            number = None
            city = None
            postal_code = None
            for comp in components:
                if 'route' in comp['types']:
                    street = comp['long_name']
                if 'street_number' in comp['types']:
                    number = comp['long_name']
                if 'locality' in comp['types']:
                    city = comp['long_name']
                if 'postal_code' in comp['types']:
                    postal_code = comp['long_name']
            # 1. Only postal code: not valid
            if postal_code and not street and not number and not city:
                return False, address
            # 2. Street + house number, and city is Stuttgart: valid
            if street and number and city and city.lower() == 'stuttgart':
                return True, formatted
            # 3. Only street and city: not valid
            if street and city and not number:
                return False, address
            # Fallback: not valid
            return False, address
        return False, address
    except Exception as e:
        print(f"Error validating address '{address}': {e}")
        return False, address


# Function to filter valid addresses from the input CSV and save them to the output CSV
def filter_valid_addresses():
    if not os.path.exists(INPUT_CSV):
        print(f"Input file not found: {INPUT_CSV}")
        return
    df = pd.read_csv(INPUT_CSV, delimiter=';')
    if ADDRESS_COLUMN not in df.columns:
        print(f"Address column not found: {ADDRESS_COLUMN}")
        return
    valid_rows = []
    for idx, row in df.iterrows():
        address = str(row[ADDRESS_COLUMN])
        is_valid, formatted_address = address_exists(address)
        if is_valid:
            row[ADDRESS_COLUMN] = formatted_address
            valid_rows.append(row)
            print(f"Valid address: {formatted_address}")
        else:
            print(f"Invalid address: {address}")
    if valid_rows:
        summary_df = pd.DataFrame(valid_rows)
        summary_df.to_csv(OUTPUT_CSV, index=False, sep=';')
        print(f"Filtered summary saved to {OUTPUT_CSV}")
    else:
        print("No valid addresses found.")


# Function to evaluate insulation methods in the forms survey and save as insulation_methods_table.csv
def evaluate_forms_survey():
    """
    Evaluates the insulation/glazing methods from the filtered forms survey summary and saves the counts as insulation_methods_table.csv and a bar chart.
    Adds average thickness rows for Dachdämmung, Fassadendämmung, and Sockeldämmämmung.
    """
    import matplotlib.pyplot as plt
    if not os.path.exists(OUTPUT_CSV):
        print(f"Filtered summary file not found: {OUTPUT_CSV}")
        return
    df = pd.read_csv(OUTPUT_CSV, delimiter=';')
    METHOD_COLUMN = 'Wählen sie unter den untenstehenden Dämmmaßnahmen solche aus, welche auf Ihre Immobilie zutreffen'
    methods = [
        "Dachdämmung (Aufsparrendämmung)",
        "Fassadendämmung",
        "Sockeldämmung",
        "Fenster (Zweifachverglasung)",
        "Fenster (Dreifachverglasung)",
        "Fenster (fassadenbündig)"
    ]
    thickness_columns = {
        "Dachdämmung (Aufsparrendämmung)": "OPTIONAL: Wie dick ist die Aufsparrendämmung? [cm]",
        "Fassadendämmung": "OPTIONAL: Wie dick ist die Fassadenämmung? [cm]",
        "Sockeldämmung":  "OPTIONAL: Wie dick ist die Sockeldämmung? [cm]"
    }
    counts = {method: 0 for method in methods}
    if METHOD_COLUMN not in df.columns:
        print(f"Method column not found: {METHOD_COLUMN}")
        return
    for entry in df[METHOD_COLUMN].dropna():
        for method in methods:
            if method in str(entry):
                counts[method] += 1
    # Prepare result table as one row per method, with 'Durchschnittliche Dämmdicke [cm]' as extra column
    rows = []
    for method in methods:
        avg = ''
        if method in thickness_columns:
            col = thickness_columns[method]
            if col in df.columns:
                cleaned = (
                    df[col]
                    .astype(str)
                    .str.replace(',', '.', regex=False)
                    .str.extract(r'([\d\.]+)')
                    .astype(float)
                )
                avg = round(cleaned.dropna().mean()[0], 2) if not cleaned.dropna().empty else ''
        rows.append({'Maßnahme': method, 'Häufigkeit': counts[method], 'Durchschnittliche Dämmdicke [cm]': avg})
    result_df = pd.DataFrame(rows)
    output_table_path = os.path.join(CHARTS_DIR, 'insulation_methods_table.csv')
    result_df.to_csv(output_table_path, index=False)
    print(f"Insulation methods summary saved to {output_table_path}")
    # Bar chart (only for main methods, not avg rows)
    plt.figure(figsize=(10, 6))
    haupt_df = result_df[result_df['Maßnahme'].isin(methods)]
    plt.bar(haupt_df['Maßnahme'], haupt_df['Häufigkeit'], color='skyblue')
    plt.xlabel('Maßnahme')
    plt.ylabel('Häufigkeit')
    plt.title('Häufigkeit der Dämmmaßnahmen')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    bar_chart_path = os.path.join(CHARTS_DIR, 'insulations_methods_bar_chart.png')
    plt.savefig(bar_chart_path)
    plt.close()
    print(f"Bar chart saved to {bar_chart_path}")

    # Neue Balkendiagramme für Dämmmaterialien
    material_columns = [
        ("OPTIONAL: Welches Dämmmaterial wurde für die Fassade verwendet? ", "fassade_material_bar_chart.png", "Dämmaterial Fassade"),
        ("OPTIONAL: Welches Dämmmaterial wurde für den Sockel verwendet?", "sockel_material_bar_chart.png", "Dämmaterial Sockel"),
        ("OPTIONAL: Welches Dämmmaterial wurde für das Dach verwendet? ", "dach_material_bar_chart.png", "Dämmaterial Dach")
    ]
    for col, filename, chart_title in material_columns:
        if col in df.columns:
            value_counts = df[col].dropna().astype(str).value_counts()
            if not value_counts.empty:
                plt.figure(figsize=(8, 5))
                ax = value_counts.plot(kind='bar', color='orange')
                plt.xlabel('Dämmmaterial')
                plt.ylabel('Häufigkeit')
                plt.title(chart_title)
                plt.xticks(rotation=0, ha='center')
                # Y-Achse ganzzahlig
                import matplotlib.ticker as mticker
                ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
                plt.tight_layout()
                out_path = os.path.join(CHARTS_DIR, filename)
                plt.savefig(out_path)
                plt.close()
                print(f"Bar chart saved to {out_path}")
            else:
                print(f"Keine Werte für {col} vorhanden.")
        else:
            print(f"Spalte nicht gefunden: {col}")

if __name__ == "__main__":
    filter_valid_addresses()
    evaluate_forms_survey()
