"""
Dieses Skript fügt in allen 'address_data.csv' Dateien in allen Unterordnern
eines übergebenen Dateipfades eine neue Spalte 'Jahresbedarf [kWh]' mit einem
zufälligen Wert zwischen 5000 und 10000 hinzu.

Verwendung:
    python add_jahresbedarf_to_csv.py [Pfad]

Wenn kein Pfad angegeben wird, wird das aktuelle Verzeichnis verwendet.
"""

import os
import sys
import random
import pandas as pd
import argparse

root_dir= "/Users/paulmayer/Desktop/Notebooks/Masterthesis_code/DLM/Test_Training_DATA"
def add_jahresbedarf_to_csv_files(root_dir):
    """
    Fügt allen address_data.csv Dateien in allen Unterordnern von root_dir
    eine Spalte 'Jahresbedarf [kWh]' mit zufälligen Werten zwischen 5000-10000 hinzu.
    
    Args:
        root_dir (str): Der Pfad zum Stammverzeichnis
    """
    # Zähler für bearbeitete Dateien
    processed_files = 0
    
    # Durchsuche alle Unterordner rekursiv
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower() == "address_data.csv":
                file_path = os.path.join(dirpath, filename)
                try:
                    # Lese die CSV-Datei
                    # Versuche mehrere Trennzeichen für robusteres Einlesen
                    try:
                        df = pd.read_csv(file_path, delimiter=';')
                    except:
                        try:
                            df = pd.read_csv(file_path, delimiter=',')
                        except:
                            df = pd.read_csv(file_path)
                    
                    # Prüfe, ob die Spalte bereits existiert
                    if 'Jahresbedarf [kWh]' not in df.columns:
                        # Generiere zufällige Werte zwischen 5000 und 10000
                        rows_count = len(df)
                        if rows_count > 0:
                            jahresbedarf_values = [round(random.uniform(5000, 10000), 2) for _ in range(rows_count)]
                            
                            # Füge die neue Spalte hinzu
                            df['Jahresbedarf [kWh]'] = jahresbedarf_values
                            
                            # Speichere die aktualisierte Datei
                            # Verwende das gleiche Trennzeichen wie in der ursprünglichen Datei
                            delimiter = ';' if ';' in open(file_path, 'r').readline() else ','
                            df.to_csv(file_path, index=False, sep=delimiter)
                            
                            processed_files += 1
                            print(f"✓ Datei aktualisiert: {file_path}")
                        else:
                            print(f"⚠ Keine Zeilen in der Datei: {file_path}")
                    else:
                        print(f"⚠ Spalte existiert bereits in: {file_path}")
                        
                except Exception as e:
                    print(f"✗ Fehler beim Verarbeiten von {file_path}: {str(e)}")
    
    print(f"\nVerarbeitung abgeschlossen. {processed_files} Dateien wurden aktualisiert.")

def main():
    # Parse Kommandozeilenargumente
    parser = argparse.ArgumentParser(
        description='Fügt allen address_data.csv Dateien eine Jahresbedarf-Spalte hinzu.')
    parser.add_argument('directory', nargs='?', default=os.getcwd(),
                        help='Das Verzeichnis, in dem gesucht werden soll (Standard: aktuelles Verzeichnis)')
    
    args = parser.parse_args()
    
    # Prüfe, ob das angegebene Verzeichnis existiert
    if not os.path.isdir(args.directory):
        print(f"Fehler: Das Verzeichnis '{args.directory}' existiert nicht.")
        sys.exit(1)
    
    print(f"Verarbeite alle address_data.csv Dateien in '{args.directory}' und Unterverzeichnissen...")
    add_jahresbedarf_to_csv_files(args.directory)

if __name__ == "__main__":
    main()