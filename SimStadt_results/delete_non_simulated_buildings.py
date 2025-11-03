"""
This script deletes folders listed in a CSV file named 'unsimulated_adresses.csv'. 
Target folder is meant to be a Training_dataset, that probably contains folders that are not simulated.
"""

import os
import pandas as pd
import shutil

source_folder = input("Pfad zum Quellordner mit unsimulated_adresses.csv: ").strip()
target_folder = input("Pfad zum Zielordner mit den zu löschenden Unterordnern: ").strip()

csv_path = os.path.join(source_folder, "unsimulated_adresses.csv")
if not os.path.exists(csv_path):
    print(f"Datei unsimulated_adresses.csv nicht gefunden in {source_folder}")
    exit(1)

df = pd.read_csv(csv_path, sep=';')
for adresse in df['Adresse'].dropna().astype(str):
    folder_to_delete = os.path.join(target_folder, adresse)
    if os.path.isdir(folder_to_delete):
        try:
            shutil.rmtree(folder_to_delete)
            print(f"Gelöscht: {folder_to_delete}")
        except Exception as e:
            print(f"Fehler beim Löschen von {folder_to_delete}: {e}")
    else:
        print(f"Ordner nicht gefunden: {folder_to_delete}")