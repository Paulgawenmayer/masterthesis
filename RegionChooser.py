import os
import traceback
import re
import copy
from shapely.wkt import loads
from shapely.geometry import Polygon, Point, LinearRing, MultiPoint
from lxml import etree
import pyproj
from shapely.ops import transform

def clip_citygml_by_polygon(wkt_polygon, citygml_path):
    """
    Stanzt ein Polygon aus einer CityGML-Datei aus und speichert den Inhalt als neue GML-Datei.
    
    Args:
        wkt_polygon (str): Das Polygon im WKT-Format
        citygml_path (str): Pfad zur CityGML-Datei
    """
    try:
        # WKT in Shapely-Geometrie umwandeln
        clip_polygon = loads(wkt_polygon)
        
        # Verzeichnis und Dateinamen ermitteln
        directory = os.path.dirname(citygml_path)
        base_filename = os.path.splitext(os.path.basename(citygml_path))[0]
        output_path = os.path.join(directory, f"{base_filename}_clipped.gml")
        
        print(f"Verarbeite {citygml_path}...")
        print(f"Ursprüngliches Clip-Polygon (WGS84): {clip_polygon}")
        
        # CityGML-Datei parsen
        parser = etree.XMLParser(remove_blank_text=True, huge_tree=True)
        tree = etree.parse(citygml_path, parser)
        root = tree.getroot()
        
        # Bestimme das Koordinatensystem der CityGML-Datei
        gml_ns = root.nsmap.get('gml', 'http://www.opengis.net/gml')
        srs_name = None
        
        # Suche nach srsName-Attribut
        for envelope in root.xpath(f'.//*[local-name()="Envelope"]'):
            if 'srsName' in envelope.attrib:
                srs_name = envelope.attrib['srsName']
                break
        
        if not srs_name:
            # Standard-Annahme für deutsche CityGML-Dateien: EPSG:25832 (UTM Zone 32N)
            srs_name = "EPSG:25832"
            print(f"Kein SRS gefunden, verwende Standard-Annahme: {srs_name}")
        
        # Extrahiere EPSG-Code
        if "EPSG:" in srs_name:
            target_epsg = srs_name
        else:
            # Fallback für andere Formate
            target_epsg = "EPSG:25832"
            print(f"Unbekanntes SRS-Format: {srs_name}, verwende {target_epsg}")
        
        # Transformiere das Polygon von WGS84 (EPSG:4326) zum Ziel-Koordinatensystem
        source_proj = pyproj.CRS('EPSG:4326')  # WGS84
        target_proj = pyproj.CRS(target_epsg)
        project = pyproj.Transformer.from_crs(source_proj, target_proj, always_xy=True).transform
        
        # Transformiere das Polygon
        transformed_polygon = transform(project, clip_polygon)
        
        print(f"Transformiertes Clip-Polygon ({target_epsg}): {transformed_polygon}")
        print(f"Transformiertes Polygon Bounds: {transformed_polygon.bounds}")
        
        # Neues CityModel erstellen mit gleichen Namespaces und Attributen
        citymodel = etree.Element(root.tag, nsmap=root.nsmap)
        for k, v in root.attrib.items():
            citymodel.set(k, v)
        
        # Metadaten kopieren (alles außer cityObjectMember/featureMember)
        for child in root:
            if not (child.tag.endswith('cityObjectMember') or child.tag.endswith('featureMember')):
                citymodel.append(copy.deepcopy(child))
        
        # Objekte finden und prüfen
        feature_count = 0
        processed_count = 0
        has_geometries = 0
        
        print("Suche nach Gebäudeobjekten innerhalb des transformierten Polygons...")
        
        # Alle cityObjectMember direkt unter dem Root-Element suchen
        city_object_members = root.xpath('./*[local-name()="cityObjectMember" or local-name()="featureMember"]')
        total_objects = len(city_object_members)
        print(f"Insgesamt {total_objects} Objekte gefunden")
        
        for city_object in city_object_members:
            processed_count += 1
            if processed_count % 100 == 0 or processed_count == total_objects:
                print(f"Verarbeitet: {processed_count}/{total_objects} Objekte, gefunden: {feature_count}, mit Geometrien: {has_geometries}")
            
            obj_in_polygon = False
            
            # 1. Versuch: Mittelpunkt aus boundedBy extrahieren
            bbox = extract_bounding_box(city_object)
            if bbox:
                # Berechne Mittelpunkt des Bounding Box
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                center_point = Point(center_x, center_y)
                if transformed_polygon.contains(center_point):
                    obj_in_polygon = True
                    if processed_count % 100 == 1:
                        print(f"Objekt {processed_count} im Polygon durch Mittelpunkttest: {center_point}")
            
            # 2. Versuch: Alle Geometrien prüfen
            if not obj_in_polygon:
                geometries = extract_geometries(city_object)
                if geometries:
                    has_geometries += 1
                    for geom in geometries:
                        if geom and (transformed_polygon.contains(geom) or transformed_polygon.intersects(geom)):
                            obj_in_polygon = True
                            if processed_count % 100 == 1:
                                print(f"Objekt {processed_count} im Polygon durch Geometrietest")
                            break
            
            # 3. Versuch: Koordinatentexte direkt prüfen
            if not obj_in_polygon:
                # Alle Koordinaten im Objekt finden
                all_coords = []
                for pos_elem in city_object.xpath('.//*[local-name()="pos" or local-name()="posList"]'):
                    if pos_elem.text and pos_elem.text.strip():
                        coords = extract_coords_from_text(pos_elem.text.strip())
                        if coords:
                            all_coords.extend(coords)
                
                # Prüfen, ob mindestens ein Punkt im Polygon liegt
                if all_coords:
                    for x, y in all_coords:
                        point = Point(x, y)
                        if transformed_polygon.contains(point):
                            obj_in_polygon = True
                            if processed_count % 100 == 1:
                                print(f"Objekt {processed_count} im Polygon durch Punkttest: {point}")
                            break
                    
                    # Wenn kein einzelner Punkt gefunden wurde, versuche mit MultiPoint
                    if not obj_in_polygon and all_coords:
                        multi_point = MultiPoint(all_coords)
                        if transformed_polygon.intersects(multi_point):
                            obj_in_polygon = True
                            if processed_count % 100 == 1:
                                print(f"Objekt {processed_count} im Polygon durch MultiPoint-Überschneidung")
            
            # Wenn das Objekt im Polygon liegt, zur neuen Datei hinzufügen
            if obj_in_polygon:
                citymodel.append(copy.deepcopy(city_object))
                feature_count += 1
        
        # Neuen Baum erstellen und speichern
        new_tree = etree.ElementTree(citymodel)
        new_tree.write(output_path, encoding="UTF-8", xml_declaration=True, pretty_print=True)
        
        print(f"\nClipping erfolgreich! {feature_count} von {processed_count} Objekten in '{output_path}' gespeichert.")
        print(f"Objekte mit erkannten Geometrien: {has_geometries}")
        return output_path
    
    except Exception as e:
        print(f"Fehler beim Verarbeiten: {str(e)}")
        traceback.print_exc()
        return None

def extract_bounding_box(element):
    """Versucht, eine BoundingBox aus einem CityGML-Element zu extrahieren"""
    try:
        # Suche nach gml:boundedBy
        bounded_by = element.xpath('.//*[local-name()="boundedBy"]')
        if bounded_by:
            # Suche nach lowerCorner und upperCorner
            lower_corner = bounded_by[0].xpath('.//*[local-name()="lowerCorner"]')
            upper_corner = bounded_by[0].xpath('.//*[local-name()="upperCorner"]')
            
            if lower_corner and upper_corner:
                lower_text = lower_corner[0].text.strip()
                upper_text = upper_corner[0].text.strip()
                
                lower_values = [float(val) for val in lower_text.split()]
                upper_values = [float(val) for val in upper_text.split()]
                
                # Rückgabe: [min_x, min_y, max_x, max_y]
                if len(lower_values) >= 2 and len(upper_values) >= 2:
                    return [lower_values[0], lower_values[1], upper_values[0], upper_values[1]]
    except Exception as e:
        pass
    return None

def extract_coords_from_text(text):
    """Extrahiert Koordinaten aus einem Text"""
    try:
        coords = []
        # Zahlen (einschließlich Dezimalzahlen mit Punkt) extrahieren
        values = [float(val) for val in re.findall(r"[-+]?\d*\.?\d+", text)]
        
        # Für 3D-Koordinaten (x,y,z) - nimm nur x,y für 2D-Geometrien
        if len(values) % 3 == 0:
            coords = [(values[i], values[i+1]) for i in range(0, len(values), 3)]
        # Für 2D-Koordinaten (x,y)
        elif len(values) % 2 == 0:
            coords = [(values[i], values[i+1]) for i in range(0, len(values), 2)]
        
        return coords
    except Exception as e:
        return []

def extract_geometries(element):
    """Extrahiert alle Geometrien aus einem CityGML-Element als Shapely-Objekte"""
    geometries = []
    
    try:
        # Alle posLists und pos-Elemente finden
        for coords_elem in element.xpath('.//*[local-name()="posList" or local-name()="pos"]'):
            if coords_elem.text and coords_elem.text.strip():
                coords_text = coords_elem.text.strip()
                coords = extract_coords_from_text(coords_text)
                
                # Wenn es genügend Punkte für ein Polygon gibt
                if len(coords) >= 3:
                    # Stelle sicher, dass das erste und letzte Koordinatenpaar gleich sind (geschlossener Ring)
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])
                    
                    try:
                        # Versuche ein gültiges Polygon zu erstellen
                        if len(coords) >= 4:  # Mindestens 3 Punkte + wiederholter Anfangspunkt
                            ring = LinearRing(coords)
                            if ring.is_valid:
                                polygon = Polygon(ring)
                                if polygon.is_valid and not polygon.is_empty:
                                    geometries.append(polygon)
                    except Exception:
                        # Versuche es als MultiPoint
                        try:
                            multi_point = MultiPoint(coords)
                            geometries.append(multi_point)
                        except:
                            pass
    except Exception:
        pass
    
    return geometries

if __name__ == "__main__":
    # Prüfen, ob pyproj installiert ist
    try:
        import pyproj
    except ImportError:
        print("Die Bibliothek 'pyproj' wird benötigt für Koordinatentransformationen.")
        print("Bitte installieren Sie sie mit: pip install pyproj")
        exit(1)
        
    # Eingabe über die IPython-Konsole
    print("Bitte geben Sie ein WKT-Polygon ein (z.B. 'POLYGON((x1 y1, x2 y2, ...))'):")
    wkt_polygon = input()
    
    # Prüfen und korrigieren des WKT-Formats
    if not wkt_polygon.startswith('POLYGON'):
        print("Warnung: WKT-String beginnt nicht mit 'POLYGON'. Korrekturversuch...")
        if '(' in wkt_polygon:
            wkt_polygon = 'POLYGON(' + wkt_polygon.split('(', 1)[1]
    
    print("Bitte geben Sie den Dateipfad zur CityGML-Datei ein:")
    citygml_path = input()
    
    # Pfad normalisieren
    citygml_path = os.path.expanduser(citygml_path)
    
    if os.path.exists(citygml_path):
        clip_citygml_by_polygon(wkt_polygon, citygml_path)
    else:
        print(f"Fehler: Die Datei '{citygml_path}' existiert nicht.")