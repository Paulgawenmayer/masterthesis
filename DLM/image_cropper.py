"""
Bild-Komprimierungsskript

Dieses Skript fragt nach einem Ordnerpfad und durchsucht diesen und alle seine Unterordner 
nach .jpg-Dateien, um sie zu komprimieren und Speicherplatz zu sparen. Die originalen Bilder 
werden durch die komprimierten Versionen ersetzt.
"""

import os
import sys
from PIL import Image
import tempfile
import shutil
from tqdm import tqdm

def get_directory_path():
    """Fragt den Benutzer nach einem Ordnerpfad und validiert ihn."""
    while True:
        path = input("Bitte geben Sie den Pfad zum zu durchsuchenden Ordner ein: ").strip()
        
        # Entferne Anführungszeichen, falls vorhanden (hilft bei kopierten Pfaden)
        if (path.startswith('"') and path.endswith('"')) or (path.startswith("'") and path.endswith("'")):
            path = path[1:-1]
        
        # Überprüfe, ob der Pfad existiert
        if not os.path.exists(path):
            print(f"Fehler: Der Pfad '{path}' existiert nicht.")
            continue
            
        # Überprüfe, ob es sich um ein Verzeichnis handelt
        if not os.path.isdir(path):
            print(f"Fehler: '{path}' ist kein Verzeichnis.")
            continue
            
        return path

def get_quality():
    """Fragt den Benutzer nach der gewünschten JPG-Qualität."""
    while True:
        try:
            quality = int(input("Gewünschte JPG-Qualität (1-95, empfohlen 70-85): ").strip())
            if 1 <= quality <= 95:
                return quality
            else:
                print("Fehler: Die Qualität muss zwischen 1 und 95 liegen.")
        except ValueError:
            print("Fehler: Bitte geben Sie eine ganze Zahl ein.")

def get_max_size():
    """Fragt den Benutzer nach der maximalen Bildgröße."""
    while True:
        size_input = input("Maximale Bildgröße in Pixeln (leer lassen für keine Begrenzung): ").strip()
        if not size_input:
            return None
        try:
            max_size = int(size_input)
            if max_size > 0:
                return max_size
            else:
                print("Fehler: Die Bildgröße muss größer als 0 sein.")
        except ValueError:
            print("Fehler: Bitte geben Sie eine ganze Zahl ein oder lassen Sie das Feld leer.")

def get_min_size():
    """Fragt den Benutzer nach der minimalen Dateigröße."""
    while True:
        size_input = input("Minimale Dateigröße in KB (leer lassen für alle Größen): ").strip()
        if not size_input:
            return 0
        try:
            min_size = int(size_input)
            if min_size >= 0:
                return min_size
            else:
                print("Fehler: Die Dateigröße darf nicht negativ sein.")
        except ValueError:
            print("Fehler: Bitte geben Sie eine ganze Zahl ein oder lassen Sie das Feld leer.")

def get_yes_no(prompt):
    """Fragt den Benutzer nach einer Ja/Nein-Antwort."""
    while True:
        response = input(prompt).strip().lower()
        if response in ['j', 'ja', 'y', 'yes']:
            return True
        elif response in ['n', 'nein', 'no']:
            return False
        else:
            print("Bitte antworten Sie mit 'j' für ja oder 'n' für nein.")

def find_jpg_files(directory):
    """Findet alle JPG-Dateien im angegebenen Verzeichnis und seinen Unterverzeichnissen."""
    jpg_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                jpg_files.append(os.path.join(root, file))
    
    return jpg_files

def compress_image(image_path, quality, max_size=None, backup=False, dry_run=False, 
                  min_size_kb=0, optimize=False, threshold=0.0):
    """
    Komprimiert ein Bild und speichert es.
    
    Args:
        image_path: Pfad zum Bild
        quality: JPG-Qualität (1-95)
        max_size: Maximale Breite/Höhe, größere Bilder werden proportional verkleinert
        backup: Wenn True, wird eine Backup-Kopie erstellt
        dry_run: Wenn True, werden keine tatsächlichen Änderungen vorgenommen
        min_size_kb: Nur Bilder größer als dieser Wert komprimieren
        optimize: Zusätzliche Optimierung aktivieren
        threshold: Minimale Verkleinerung in Prozent, um das Bild zu speichern
        
    Returns:
        (success, old_size, new_size): Erfolg und Größen vor/nach Komprimierung in KB
    """
    try:
        # Überprüfe die aktuelle Dateigröße
        old_size = os.path.getsize(image_path) / 1024  # KB
        
        # Überspringe, wenn die Datei kleiner als min_size_kb ist
        if old_size < min_size_kb:
            return True, old_size, old_size, 0, "Übersprungen (zu klein)"
        
        img = Image.open(image_path)
        
        # Skaliere das Bild, wenn es größer als max_size ist
        resized = False
        original_size = img.size
        if max_size and (img.width > max_size or img.height > max_size):
            if img.width > img.height:
                new_width = max_size
                new_height = int(img.height * (max_size / img.width))
            else:
                new_height = max_size
                new_width = int(img.width * (max_size / img.height))
            
            img = img.resize((new_width, new_height), Image.LANCZOS)
            resized = True
        
        # Erstelle ein temporäres Verzeichnis für die Komprimierung
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, os.path.basename(image_path))
        
        # Komprimiere das Bild
        img.save(temp_file, "JPEG", quality=quality, optimize=optimize)
        
        # Überprüfe die neue Dateigröße
        new_size = os.path.getsize(temp_file) / 1024  # KB
        
        # Berechne den Speicherplatzgewinn in Prozent
        saving_percent = ((old_size - new_size) / old_size) * 100 if old_size > 0 else 0
        
        # Wenn der Schwellenwert nicht erreicht wird, überspringe dieses Bild
        if saving_percent < threshold:
            os.remove(temp_file)
            return True, old_size, old_size, 0, "Übersprungen (Schwellenwert nicht erreicht)"
        
        # Erstelle ein Backup, wenn gewünscht
        if backup and not dry_run:
            backup_path = image_path + ".bak"
            if not os.path.exists(backup_path):
                shutil.copy2(image_path, backup_path)
        
        # Speichere das komprimierte Bild
        if not dry_run:
            shutil.move(temp_file, image_path)
        else:
            os.remove(temp_file)
        
        # Generiere eine Statusnachricht
        if resized:
            status = f"Komprimiert & skaliert ({original_size[0]}x{original_size[1]} → {img.width}x{img.height})"
        else:
            status = "Komprimiert"
        
        return True, old_size, new_size, saving_percent, status
    
    except Exception as e:
        print(f"Fehler bei {image_path}: {e}")
        return False, 0, 0, 0, f"Fehler: {e}"

def main():
    """Hauptfunktion des Skripts."""
    print("===== Bild-Komprimierungsskript =====")
    print("Dieses Skript komprimiert JPG-Bilder in einem Verzeichnis und seinen Unterordnern,")
    print("um Speicherplatz zu sparen.\n")
    
    try:
        # Einstellungen vom Benutzer abfragen
        directory = get_directory_path()
        quality = get_quality()
        max_size = get_max_size()
        min_size = get_min_size()
        optimize = get_yes_no("Zusätzliche Optimierung aktivieren? (j/n): ")
        backup = get_yes_no("Backup der Originalbilder erstellen? (j/n): ")
        dry_run = get_yes_no("Nur Simulation durchführen (keine Änderungen)? (j/n): ")
        threshold = 0
        if not dry_run:
            threshold_input = input("Minimale Speicherersparnis in Prozent (leer lassen für 0): ").strip()
            if threshold_input:
                threshold = float(threshold_input)
        
        # Finde alle JPG-Dateien
        jpg_files = find_jpg_files(directory)
        
        if not jpg_files:
            print(f"Keine JPG-Dateien im Verzeichnis '{directory}' und seinen Unterverzeichnissen gefunden.")
            return 0
        
        print(f"\n{len(jpg_files)} JPG-Dateien gefunden.")
        
        if dry_run:
            print("SIMULATION: Es werden keine tatsächlichen Änderungen vorgenommen.")
        
        # Zeige die geplanten Kompressionsparameter
        print(f"Kompressionsparameter:")
        print(f"  Qualität: {quality}")
        if max_size:
            print(f"  Maximale Größe: {max_size}x{max_size} Pixel")
        print(f"  Minimale Dateigröße: {min_size} KB")
        print(f"  Optimierung: {'Ein' if optimize else 'Aus'}")
        print(f"  Mindest-Speicherersparnis: {threshold}%")
        
        # Bestätigung einholen, wenn es keine Simulation ist
        if not dry_run:
            confirm = input(f"\nMöchten Sie fortfahren und {len(jpg_files)} Bilder komprimieren? (j/n): ").lower()
            if confirm not in ['j', 'ja', 'y', 'yes']:
                print("Vorgang abgebrochen.")
                return 0
        
        # Komprimiere die Bilder
        successful = 0
        failed = 0
        skipped = 0
        total_savings_kb = 0
        
        # Erstelle eine Tabelle für die Ergebnisse
        results = []
        
        for image_path in tqdm(jpg_files, desc="Bilder komprimieren"):
            success, old_size, new_size, saving_percent, status = compress_image(
                image_path, quality, max_size, backup, dry_run,
                min_size, optimize, threshold)
            
            if success:
                if old_size == new_size:
                    skipped += 1
                else:
                    successful += 1
                    total_savings_kb += (old_size - new_size)
            else:
                failed += 1
            
            # Speichere die Ergebnisse für die spätere Ausgabe
            if old_size != new_size or "Fehler" in status:
                results.append({
                    'path': image_path,
                    'old_size': old_size,
                    'new_size': new_size,
                    'saving_percent': saving_percent,
                    'status': status
                })
        
        # Ausgabe der Ergebnisse
        if dry_run:
            print(f"\nSIMULATION abgeschlossen. {len(jpg_files)} Bilder wurden analysiert.")
        else:
            print(f"\nVorgang abgeschlossen:")
            print(f"  {successful} Bilder erfolgreich komprimiert")
            print(f"  {skipped} Bilder übersprungen")
            print(f"  {failed} Bilder fehlgeschlagen")
            print(f"  Gesparter Speicherplatz: {total_savings_kb:.2f} KB ({total_savings_kb/1024:.2f} MB)")
        
        # Zeige die Top 10 Kompressionen
        if results:
            print("\nTop Kompressionen:")
            # Sortiere nach Einsparung in Prozent (absteigend)
            results.sort(key=lambda x: x['saving_percent'], reverse=True)
            
            # Zeige die Top 10 oder weniger, wenn nicht genügend Ergebnisse vorhanden sind
            for i, result in enumerate(results[:10]):
                if result['saving_percent'] > 0:  # Nur Bilder anzeigen, die tatsächlich komprimiert wurden
                    print(f"  {i+1}. {os.path.basename(result['path'])}: "
                        f"{result['old_size']:.1f}KB → {result['new_size']:.1f}KB "
                        f"({result['saving_percent']:.1f}% gespart) - {result['status']}")
        
    except KeyboardInterrupt:
        print("\nVorgang durch Benutzer abgebrochen.")
        return 1
    except Exception as e:
        print(f"\nEin Fehler ist aufgetreten: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())