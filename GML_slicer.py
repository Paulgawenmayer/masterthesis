"""
This script extracts specific building objects from large CityGML files based on a list of addresses.
It reads addresses from a CSV file, searches for matching address entries in the GML files,
and saves each found object as a separate GML file with the appropriate CityModel header.

Those GML files can then be used for further processing, e.g. in a 3D viewer or SIMSTADT etc.

Unavailable addresses (not found in any GML file) are recorded in a separate CSV file.
"""
import os
import pandas as pd
from lxml import etree
import copy  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADDRESS_CSV = os.path.join(BASE_DIR, "field_survey", "survey_summary", "address_list.csv")
GML_DIRS = [
    os.path.join(BASE_DIR, "field_survey", "Moehringen", "Möhringen.gml"),
    os.path.join(BASE_DIR, "field_survey", "Sillenbuch", "Sillenbuch.gml"),
    os.path.join(BASE_DIR, "field_survey", "Stammheim", "Stammheim.gml"),
]
OUTPUT_DIR = os.path.join(BASE_DIR, "field_survey", "Object_GMLs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

addresses = pd.read_csv(ADDRESS_CSV)["Address"].dropna().astype(str).tolist()

def extract_street_number_from_csv(address):
    return address.split(",")[0].strip()

def write_gml_with_citymodel(obj_elem, root, out_path):
    citymodel = etree.Element(root.tag, nsmap=root.nsmap)
    for k, v in root.attrib.items():
        citymodel.set(k, v)
    
    # search for direct children of <core:CityModel>
    gml_ns = root.nsmap.get("gml", "http://www.opengis.net/gml")
    for child in root:
        if child.tag == f"{{{gml_ns}}}name" or child.tag == f"{{{gml_ns}}}boundedBy":
            citymodel.append(copy.deepcopy(child))
    
    # If no Metadata found, create them manually
    if citymodel.find(f"{{{gml_ns}}}boundedBy") is None:
        # Create <gml:boundedBy> manually
        bounded_by = etree.Element(f"{{{gml_ns}}}boundedBy")
        envelope = etree.SubElement(bounded_by, f"{{{gml_ns}}}Envelope")
        envelope.set("srsName", "EPSG:25832")
        envelope.set("srsDimension", "3")
        lower_corner = etree.SubElement(envelope, f"{{{gml_ns}}}lowerCorner")
        lower_corner.text = "510000.0 5396000.0 410.0"  # Dummy-Werte
        upper_corner = etree.SubElement(envelope, f"{{{gml_ns}}}upperCorner")
        upper_corner.text = "511000.0 5397000.0 450.0"  # Dummy-Werte
        citymodel.append(bounded_by)
    
    # Add extracted object
    if obj_elem.tag.endswith("cityObjectMember"):
        citymodel.append(obj_elem)
    else:
        city_object_member = etree.Element("{http://www.opengis.net/citygml/1.0}cityObjectMember")
        city_object_member.append(obj_elem)
        citymodel.append(city_object_member)
    
    tree = etree.ElementTree(citymodel)
    tree.write(out_path, encoding="utf-8", xml_declaration=True, pretty_print=True)
    

def find_and_save_gml_objects(addresses, gml_files, output_dir):
    # followed which addresses were found
    found_addresses = set()
    
    for gml_file in gml_files:
        if not os.path.exists(gml_file):
            print(f"GML file not found: {gml_file}")
            continue
        print(f"Processing {gml_file}...")
        tree = etree.parse(gml_file)
        root = tree.getroot()
        nsmap = root.nsmap
        
        for address in addresses:
            # Skip already found addresses
            if address in found_addresses:
                continue
                
            found = False
            street_nr_csv = extract_street_number_from_csv(address)
            
            for elem in root.xpath(".//*[local-name()='address']", namespaces=nsmap):
                thoroughfare_name = elem.find(".//{urn:oasis:names:tc:ciq:xsdschema:xAL:2.0}ThoroughfareName")
                thoroughfare_number = elem.find(".//{urn:oasis:names:tc:ciq:xsdschema:xAL:2.0}ThoroughfareNumber")
                
                if thoroughfare_name is not None and thoroughfare_number is not None:
                    street_nr_gml = f"{thoroughfare_name.text.strip()} {thoroughfare_number.text.split(',')[0].strip()}"
                    
                    if street_nr_csv.strip().lower() == street_nr_gml.strip().lower():
                        # Go up the tree to find the <core:cityObjectMember>
                        parent_obj = elem
                        while parent_obj.getparent() is not None and not parent_obj.tag.endswith("cityObjectMember"):
                            parent_obj = parent_obj.getparent()
                            
                        # Write the entire object to a new GML file with CityModel header
                        safe_name = address.replace(",", "").replace(" ", "_").replace("/", "_")
                        out_path = os.path.join(output_dir, f"{safe_name}.gml")
                        write_gml_with_citymodel(parent_obj, root, out_path)
                        print(f"Saved: {out_path}")
                        
                        found = True
                        found_addresses.add(address)  # Mark this address as found
                        break
                        
            if not found:
                print(f"Address not found in {os.path.basename(gml_file)}: {address}")

    # After searching ALL files: Calculate not found addresses
    unavailable = [address for address in addresses if address not in found_addresses]
    
    # Write unavailable addresses to CSV
    unavailable_path = os.path.join(output_dir, "Unavailable_addresses_in_gml.csv")
    pd.DataFrame({"Unavailable_Address": unavailable}).to_csv(unavailable_path, index=False)
    print(f"\nFound {len(found_addresses)} addresses out of {len(addresses)}")
    print(f"{len(unavailable)} unavailable addresses saved to {unavailable_path}")

find_and_save_gml_objects(addresses, GML_DIRS, OUTPUT_DIR)