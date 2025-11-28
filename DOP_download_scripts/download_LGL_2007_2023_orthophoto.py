"""
This script downloads several DOP images from 2007-2023 for a given coordinate or bounding box
in Baden-Wuerttemberg.
"""
import os
import sys
from owslib.wms import WebMapService
import matplotlib.pyplot as plt
from pyproj import Transformer
import tempfile
import shutil
import re

# Set import path to path of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from image_has_content_checker import is_image_blank

def wgs84_to_utm32(lat, lon):
    """Convert WGS84 coordinates to UTM32/EPSG:25832"""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
    return transformer.transform(lon, lat)

def get_LGL_2007_2023_DOP_by_coord(latitude, longitude, output_dir=None):
    """
    Downloads DOP images from 2007-2023 for a single coordinate point.
    Creates a 20x20m bounding box around the point.
    """
    if output_dir is None:
        output_dir = os.path.join(script_dir, "Downloads/LGL/Historical/2007_2023")
    os.makedirs(output_dir, exist_ok=True)
    
    x_coordinate, y_coordinate = wgs84_to_utm32(latitude, longitude)
    found_usable_image = False  # Flag to track if any usable image was found
    
    # Parameters
    area = 20  # Meters
    
    # WMS URL for historical DOP
    wms_url = "https://owsproxy.lgl-bw.de/owsproxy/ows/WMS_LGL-BW_ATKIS_HIST_DOP_20_RGB?"
    
    try:
        # Initialize WMS
        wms = WebMapService(wms_url, version='1.3.0')
        # Dynamically get all layer names from the WMS
        layer_list = list(wms.contents.keys())
    
        for layer in layer_list:
            print(f"\nExamining layer {layer}...")
            
            try:
                # Request image from WMS server
                img = wms.getmap(
                    layers=[layer],
                    srs='EPSG:25832',
                    bbox=(x_coordinate - area, y_coordinate - area,
                          x_coordinate + area, y_coordinate + area),
                    size=(512, 512),
                    format='image/jpeg'
                )
                
                # Save image temporarily to check if it's empty
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    tmp_file.write(img.read())
                    tmp_path = tmp_file.name

                # Check if image contains actual content
                if is_image_blank(tmp_path):
                    print(f"Layer {layer} doesn't contain a usable image")
                    os.remove(tmp_path)
                else:
                    found_usable_image = True
                    # Extract year (first number in layer name)
                    match = re.match(r"(\d{4})", layer)
                    if match:
                        year = match.group(1)
                    else:
                        year = "unknown"
                    # Define final path and move the temporary file
                    filename = f"{latitude:.6f}_{longitude:.6f}_LGL_{year}.jpg"
                    filepath = os.path.join(output_dir, filename)
                    shutil.move(tmp_path, filepath)
                    print(f"✅ Layer {layer} saved as: {filepath}")
                    
            except Exception as e:
                print(f"❌ Error processing layer {layer}: {str(e)}")
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
        
        if not found_usable_image:
            print("\n❌ No usable images found for the given coordinates in any layer (2007-2023)")
            return False
        return True
        
    except Exception as e:
        print(f"\n❌ Error establishing WMS connection: {e}")
        print(f"Tip: Verify the WMS URL: {wms_url}")
        return False

def get_LGL_2007_2023_DOP_by_bbox(lat1, lon1, lat2, lon2, output_dir=None):
    """
    Downloads DOP images from 2007-2023 for a bounding box defined by two coordinates:
    (lat1, lon1) - top left corner
    (lat2, lon2) - bottom right corner
    """
    if output_dir is None:
        output_dir = os.path.join(script_dir, "Downloads/LGL/Historical/2007_2023")
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert WGS84 coordinates to UTM32
    x1, y1 = wgs84_to_utm32(lat1, lon1)  # Top-left
    x2, y2 = wgs84_to_utm32(lat2, lon2)  # Bottom-right
    
    found_usable_image = False  # Flag to track if any usable image was found
    
    # WMS URL for historical DOP
    wms_url = "https://owsproxy.lgl-bw.de/owsproxy/ows/WMS_LGL-BW_ATKIS_HIST_DOP_20_RGB?"
    
    try:
        # Initialize WMS
        wms = WebMapService(wms_url, version='1.3.0')
        # Dynamically get all layer names from the WMS
        layer_list = list(wms.contents.keys())
    
        for layer in layer_list:
            print(f"\nExamining layer {layer}...")
            
            try:
                # Request image from WMS server using the bbox
                img = wms.getmap(
                    layers=[layer],
                    srs='EPSG:25832',
                    bbox=(x1, y2, x2, y1),  # Note: WMS bbox format is (minx, miny, maxx, maxy)
                    size=(1024, 1024),  # Higher resolution for larger area
                    format='image/jpeg'
                )
                
                # Save image temporarily to check if it's empty
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    tmp_file.write(img.read())
                    tmp_path = tmp_file.name

                # Check if image contains actual content
                if is_image_blank(tmp_path):
                    print(f"Layer {layer} doesn't contain a usable image")
                    os.remove(tmp_path)
                else:
                    found_usable_image = True
                    # Extract year (first number in layer name)
                    match = re.match(r"(\d{4})", layer)
                    if match:
                        year = match.group(1)
                    else:
                        year = "unknown"
                    # Define final path and move the temporary file
                    filename = f"{lat1:.6f}_{lon1:.6f}__{lat2:.6f}_{lon2:.6f}_LGL_{year}.jpg"
                    filepath = os.path.join(output_dir, filename)
                    shutil.move(tmp_path, filepath)
                    print(f"✅ Layer {layer} saved as: {filepath}")
                    
            except Exception as e:
                print(f"❌ Error processing layer {layer}: {str(e)}")
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
        
        if not found_usable_image:
            print("\n❌ No usable images found for the given bounding box in any layer (2007-2023)")
            return False
        return True
        
    except Exception as e:
        print(f"\n❌ Error establishing WMS connection: {e}")
        print(f"Tip: Verify the WMS URL: {wms_url}")
        return False

def get_input():
    """Gets user input for coordinates"""
    print("Enter coordinates in one of the following formats:")
    print("1. Single point: 'latitude, longitude'")
    print("2. Bounding box: 'lat1, lon1, lat2, lon2' (top-left and bottom-right corners)")
    
    input_str = input("Enter coordinates (format: 'lat,lon' for point or 'lat1,lon1,lat2,lon2' for bbox): ").strip()
    
    try:
        parts = [float(x.strip()) for x in input_str.split(',')]
        
        if len(parts) == 2:
            # Single coordinate mode
            lat, lon = parts
            print(f"\nDownloading 2007-2023 DOPs for coordinate: {lat}, {lon}")
            return get_LGL_2007_2023_DOP_by_coord(lat, lon)
            
        elif len(parts) == 4:
            # Bounding box mode
            lat1, lon1, lat2, lon2 = parts
            print(f"\nDownloading 2007-2023 DOPs for bounding box: ({lat1}, {lon1}) to ({lat2}, {lon2})")
            return get_LGL_2007_2023_DOP_by_bbox(lat1, lon1, lat2, lon2)
            
        else:
            print("Invalid input format. Please provide either 2 or 4 comma-separated coordinates.")
            return False
            
    except ValueError:
        print("Wrong format! Input needs to match pattern: '48.12345, 10.12345' or")
        print("'48.12345, 10.12345, 48.12000, 10.12500'")
        return False

if __name__ == "__main__":
    success = get_input()
    if not success:
        print("No valid images found or error occurred during processing.")