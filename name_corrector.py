#!/usr/bin/env python3
import os
import sys

def normalize_name(name):
    """Ersetzt Leerzeichen mit _ und doppelte __ mit einzelnem _"""
    normalized = name.replace(" ", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized

def rename_folders_and_files(base_path, verbose=True):
    """
    Benennt Ordner und enthaltene JPG-Dateien in einem Verzeichnis um
    - Ersetzt Leerzeichen mit Unterstrichen
    - Entfernt doppelte Unterstriche
    - Benennt JPG-Dateien nach dem Ordnernamen
    """
    if not os.path.exists(base_path):
        print(f"Der Pfad {base_path} existiert nicht!")
        return 0, 0

    renamed_folders = 0
    renamed_files = 0
    
    # Liste aller Ordner im angegebenen Pfad
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        
        # 1. Normalisiere den Ordnernamen
        new_folder_name = normalize_name(folder)
        if new_folder_name != folder:
            new_folder_path = os.path.join(base_path, new_folder_name)
            
            # Prüfe, ob Zielordner bereits existiert
            if os.path.exists(new_folder_path):
                if verbose:
                    print(f"Warnung: Zielordner '{new_folder_name}' existiert bereits, überspringe '{folder}'")
                continue
                
            if verbose:
                print(f"Benenne Ordner um: '{folder}' -> '{new_folder_name}'")
            os.rename(folder_path, new_folder_path)
            folder_path = new_folder_path
            renamed_folders += 1
        elif verbose:
            print(f"Ordnername '{folder}' bleibt unverändert")
        
        # 2. Benenne JPG-Dateien im Ordner um
        jpg_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
        
        for jpg_file in jpg_files:
            jpg_path = os.path.join(folder_path, jpg_file)
            new_jpg_name = f"{new_folder_name}.jpg"
            new_jpg_path = os.path.join(folder_path, new_jpg_name)
            
            if jpg_file.lower() != new_jpg_name.lower():  # Case-insensitive check
                if verbose:
                    print(f"  Benenne Datei um: '{jpg_file}' -> '{new_jpg_name}'")
                # Falls die Zieldatei bereits existiert, lösche sie
                if os.path.exists(new_jpg_path) and jpg_path.lower() != new_jpg_path.lower():
                    os.remove(new_jpg_path)
                os.rename(jpg_path, new_jpg_path)
                renamed_files += 1
            elif verbose:
                print(f"  Dateiname '{jpg_file}' bleibt unverändert")
    
    return renamed_folders, renamed_files

def run_interactive():
    """Führt das Programm im interaktiven Modus aus"""
    
    # Eingabeaufforderung für den Pfad
    print("=== Ordner- und Dateinamen-Korrektor ===")
    path = input("Gib den relativen oder absoluten Pfad zum Verzeichnis ein\n(ENTER für aktuelles Verzeichnis): ")
    
    if not path:
        path = "."
        
    # Pfad auflösen
    abs_path = os.path.abspath(path)
    
    # Frage nach verbosem Modus
    verbose_input = input("Detaillierte Ausgabe anzeigen? (j/n, Standard: j): ")
    verbose = verbose_input.lower() not in ('n', 'nein', 'no', '0')
    
    print(f"\nVerarbeite Ordner: {abs_path}")
    renamed_folders, renamed_files = rename_folders_and_files(abs_path, verbose=verbose)
    print(f"\nFertig! {renamed_folders} Ordner und {renamed_files} Dateien umbenannt.")

# Wenn das Skript direkt ausgeführt wird (nicht importiert)
if __name__ == "__main__":
    run_interactive()