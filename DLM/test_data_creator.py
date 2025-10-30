"""
Datei-Bereinigungsskript

Dieses Skript fragt nach einem absoluten Pfad und löscht in diesem Verzeichnis und allen 
Unterverzeichnissen:
1. Alle Bilddateien, die "LGL" oder "GM" im Namen enthalten 
2. Alle .gml-Dateien

Unterstützte Bilddateiformate: .jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp
"""

import os
import sys

def get_absolute_path():
    """Fragt den Benutzer nach einem absoluten Pfad und validiert ihn."""
    while True:
        path = input("Bitte geben Sie einen absoluten Pfad ein: ").strip()
        
        # Prüfe, ob der Pfad absolut ist
        if not os.path.isabs(path):
            print("Fehler: Dies ist kein absoluter Pfad. Bitte geben Sie einen vollständigen Pfad ein.")
            continue
            
        # Prüfe, ob der Pfad existiert
        if not os.path.exists(path):
            print(f"Fehler: Der Pfad '{path}' existiert nicht.")
            continue
            
        # Prüfe, ob es sich um ein Verzeichnis handelt
        if not os.path.isdir(path):
            print(f"Fehler: '{path}' ist kein Verzeichnis.")
            continue
            
        return path

def confirm_deletion(file_count):
    """Bittet den Benutzer um Bestätigung vor dem Löschen."""
    if file_count == 0:
        print("Keine zu löschenden Dateien gefunden.")
        return False
        
    response = input(f"{file_count} Dateien werden gelöscht. Möchten Sie fortfahren? (y/n): ").strip().lower()
    return response in ['j', 'ja', 'y', 'yes']

def delete_files(root_path):
    """
    Löscht:
    1. Alle Bilddateien, die "LGL" oder "GM" im Namen enthalten
    2. Alle .gml-Dateien
    in root_path und Unterordnern.
    """
    # Liste der zu berücksichtigenden Bilddateierweiterungen
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    
    # Zu suchende Schlüsselwörter im Dateinamen
    keywords = ["LGL", "GM"]
    
    files_to_delete = []
    
    # Durchsuche alle Unterordner
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            
            # Fall 1: .gml-Datei
            if filename.lower().endswith('.gml'):
                files_to_delete.append(file_path)
                continue
                
            # Fall 2: Bilddatei mit Schlüsselwort
            _, ext = os.path.splitext(filename)
            if ext.lower() in image_extensions and any(keyword in filename for keyword in keywords):
                files_to_delete.append(file_path)
    
    # Zeige die gefundenen Dateien an
    if files_to_delete:
        print(f"\nGefundene zu löschende Dateien ({len(files_to_delete)}):")
        for file_path in files_to_delete[:10]:  # Zeige nur die ersten 10 Dateien
            print(f"- {file_path}")
            
        if len(files_to_delete) > 10:
            print(f"... und {len(files_to_delete) - 10} weitere Dateien")
    
    # Bestätigung einholen
    if not confirm_deletion(len(files_to_delete)):
        print("Vorgang abgebrochen.")
        return 0
    
    # Dateien löschen
    deleted_count = 0
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            deleted_count += 1
            
            # Fortschrittsanzeige für große Mengen an Dateien
            if deleted_count % 100 == 0:
                print(f"Gelöscht: {deleted_count}/{len(files_to_delete)} Dateien")
                
        except Exception as e:
            print(f"Fehler beim Löschen von {file_path}: {str(e)}")
    
    return deleted_count

def main():
    print("===== Datei-Bereinigungsskript =====")
    print("Dieses Skript löscht in einem Verzeichnis und seinen Unterverzeichnissen:")
    print("1. Alle Bilddateien, die 'LGL' oder 'GM' im Namen enthalten")
    print("2. Alle .gml-Dateien\n")
    
    try:
        # Absoluten Pfad vom Benutzer abfragen
        path = get_absolute_path()
        
        # Dateien löschen
        deleted_count = delete_files(path)
        
        if deleted_count > 0:
            print(f"\nVorgang abgeschlossen. {deleted_count} Dateien wurden gelöscht.")
        
    except KeyboardInterrupt:
        print("\nVorgang durch Benutzer abgebrochen.")
        return 1
    except Exception as e:
        print(f"\nEin Fehler ist aufgetreten: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())