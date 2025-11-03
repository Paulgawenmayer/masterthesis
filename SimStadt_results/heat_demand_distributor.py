"""
This script processes CSV files containing heating demand data from SimStadt and updates corresponding
address_data.CSV files with this information. It performs the following steps:
1. Reads CSV files from a specified directory, which contain heating demand data from SimStadt.
2. Maps building IDs from the CSV files to addresses using GML filenames - for merging addresses to IDs.
3. Merges all CSV files into a single DataFrame named "Heating_demands_for_survey_area_buildings", retaining only rows with valid addresses.
4. Updates the "Jahresbedarf [kWh]" column in each address_data.CSV in training_datasets folder with the corresponding heating demand.
5. If multiple rows in the final CSV have the same value in "ParentGMLId", only the row with the highest "Yearly Heating demand" is kept.
6. Creates a CSV "unsimulated_adresses.csv" listing all addresses (folder names in gml_root_folder) that are not present in "Heating_demands_for_survey_area_buildings",
   as they could not be simulated in SimStadt.
7. Generates two bar charts visualizing the distribution of "Yearly Heating demand" and "Specific heating demand".
"""

import os
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
import matplotlib.pyplot as plt

# 1. Eingabe der Pfade
csv_folder = input("Pfad zum Ordner mit den CSV-Dateien: ").strip()
gml_root_folder = input("Pfad zum Wurzelordner mit den GML-Dateien: ").strip()

# 2. Hilfsfunktion: Mapping von ID zu Adresse (GML-Dateiname)
def build_id_to_address_map(gml_root_folder):
    id_to_address = {}
    for root, dirs, files in os.walk(gml_root_folder):
        for file in files:
            if file.lower().endswith('.gml'):
                gml_path = os.path.join(root, file)
                try:
                    tree = ET.parse(gml_path)
                    root_elem = tree.getroot()
                    for elem in root_elem.iter():
                        gml_id = elem.attrib.get('{http://www.opengis.net/gml}id')
                        if gml_id:
                            id_to_address[gml_id] = os.path.splitext(file)[0]
                except Exception as e:
                    print(f"Fehler beim Parsen von {gml_path}: {e}")
    return id_to_address

print("Baue Mapping von IDs zu Adressen auf (kann je nach Datenmenge dauern)...")
id_to_address = build_id_to_address_map(gml_root_folder)
print(f"{len(id_to_address)} IDs gefunden.")

# 3. Verarbeitung der CSV-Dateien und Sammeln der Pfade der neuen Dateien
new_csv_paths = []
for csv_file in os.listdir(csv_folder):
    if csv_file.lower().endswith('.csv'):
        csv_path = os.path.join(csv_folder, csv_file)
        print(f"Verarbeite {csv_file} ...")
        df = pd.read_csv(csv_path, sep=';', comment='#')
        id_col = None
        for col in df.columns:
            if 'id' in col.lower():
                id_col = col
                break
        if id_col is None:
            print(f"Keine ID-Spalte in {csv_file} gefunden. Überspringe Datei.")
            continue
        df['Adresse'] = df[id_col].map(id_to_address)
        out_path = os.path.join(csv_folder, f"{os.path.splitext(csv_file)[0]}_mit_Adresse.csv")
        df.to_csv(out_path, sep=';', index=False)
        new_csv_paths.append(out_path)
        print(f"Datei mit Adressen gespeichert: {out_path}")

# 4. Alle neuen CSVs zusammenführen und nur Zeilen mit Adresse behalten
dfs = []
for path in new_csv_paths:
    df = pd.read_csv(path, sep=';')
    df = df[df['Adresse'].notnull() & (df['Adresse'] != "")]
    dfs.append(df)
if dfs:
    final_df = pd.concat(dfs, ignore_index=True)
    final_path = os.path.join(csv_folder, "Heating_demands_for_survey_area_buildings.csv")
    final_df.to_csv(final_path, sep=';', index=False)
    print(f"Finale Datei gespeichert: {final_path}")
else:
    print("Keine Daten zum Zusammenführen gefunden.")

# 5. Zwischenschritt: ParentGMLId-Gruppierung und Filterung auf höchsten Heating Demand
final_csv = os.path.join(csv_folder, "Heating_demands_for_survey_area_buildings.csv")
df = pd.read_csv(final_csv, sep=';')

if "ParentGMLId" in df.columns:
    has_parent = df["ParentGMLId"].notnull() & (df["ParentGMLId"] != "")
    no_parent = df[~has_parent]
    parent_grouped = (
        df[has_parent]
        .sort_values("Yearly Heating demand", ascending=False)
        .groupby("ParentGMLId", as_index=False)
        .first()
    )
    # Wert aus ParentGMLId in GMLId übernehmen, ParentGMLId leeren
    parent_grouped["GMLId"] = parent_grouped["ParentGMLId"]
    parent_grouped["ParentGMLId"] = ""
    final_df = pd.concat([no_parent, parent_grouped], ignore_index=True)
    final_df.to_csv(final_csv, sep=';', index=False)
    print("Heating_demands_for_survey_area_buildings.csv wurde nach ParentGMLId gefiltert (nur höchster Heating Demand je Gruppe, GMLId übernommen).")
else:
    print("Spalte 'ParentGMLId' nicht gefunden, Zwischenschritt übersprungen.")

# 6. Erstelle unsimulated_adresses.csv
simulated_addresses = set(pd.read_csv(final_csv, sep=';')['Adresse'].dropna().astype(str))
all_address_folders = set(
    folder for folder in os.listdir(gml_root_folder)
    if os.path.isdir(os.path.join(gml_root_folder, folder))
)
unsimulated = sorted(all_address_folders - simulated_addresses)
unsimulated_path = os.path.join(csv_folder, "unsimulated_adresses.csv")
pd.DataFrame({"Adresse": unsimulated}).to_csv(unsimulated_path, sep=';', index=False)
print(f"Nicht simulierte Adressen gespeichert: {unsimulated_path}")

# 7. Update address_data.csv Dateien mit Heating Demand
for idx, row in final_df.iterrows():
    adresse = row['Adresse']
    heating_demand = row['Yearly Heating demand']
    address_folder = os.path.join(gml_root_folder, adresse)
    address_data_path = os.path.join(address_folder, "address_data.csv")
    if os.path.exists(address_data_path):
        address_df = pd.read_csv(address_data_path, sep=';')
        address_df["Jahresbedarf [kWh]"] = heating_demand
        address_df.to_csv(address_data_path, sep=';', index=False)
        print(f"Wert für {adresse} gesetzt.")
    else:
        print(f"address_data.csv für {adresse} nicht gefunden.")

# Lade die Daten für Diagrammerstellung:
df = pd.read_csv(final_path, sep=';', on_bad_lines='skip')

# 1. Balkendiagramm: Yearly Heating demand (Schrittweite 10.000)
bin_width_1 = 10000
# Starte bei der nächsten 1.000er-Stelle unterhalb des Minimums
min_val_1 = int(df['Yearly Heating demand'].min() // 1000 * 1000)
# max_val_1 wie gehabt, aber auf die nächste 10.000er-Stelle aufrunden
max_val_1 = int(df['Yearly Heating demand'].max() // bin_width_1 * bin_width_1 + bin_width_1)
bins_1 = range(min_val_1, max_val_1 + bin_width_1, bin_width_1)

plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(df['Yearly Heating demand'], bins=bins_1, edgecolor='black')
plt.xlabel('Verteilung jährlicher Heizwärmebedarf in kWh')
plt.ylabel('Anzahl Gebäude')
plt.xticks(bins_1, rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(csv_folder, 'Yearly_Heating_demand_distribution.png'))
plt.close()

# 2. Balkendiagramm: Specific space heating demand (Schrittweite 10, Potenz als Hochzahl)
bin_width_2 = 10
df['Specific space heating demand'] = (
    df['Specific space heating demand']
    .astype(str)
    .str.replace(',', '.')
    .astype(float)
)
min_val_2 = int(df['Specific space heating demand'].min() // 10 * 10)
max_val_2 = int(df['Specific space heating demand'].max() // 10 * 10 + bin_width_2)
bins_2 = range(min_val_2, max_val_2 + bin_width_2, bin_width_2)

plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(df['Specific space heating demand'], bins=bins_2, edgecolor='black')
plt.xlabel('Spezifischer Heizwärmebedarf [kWh/(m$^2$a)]')
plt.ylabel('Anzahl Gebäude')
plt.xticks(bins_2, rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(csv_folder, 'Specific_heating_demand_distribution.png'))
plt.close()