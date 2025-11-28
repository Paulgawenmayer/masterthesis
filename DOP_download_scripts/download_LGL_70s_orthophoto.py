"""
This script downloads DOP images from the 1970s for a given coordinate or bounding box
in Baden-Wuerttemberg.
"""

import sys
import os
import tempfile
import shutil
import matplotlib.pyplot as plt
from owslib.wms import WebMapService
from pyproj import Transformer

# set import path to path of this script to avoid path-related-import-problems for custom scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from image_has_content_checker import is_image_blank 

def wgs84_to_utm32(lat, lon):
    """Convert WGS84 coordinates to UTM32/EPSG:25832"""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
    return transformer.transform(lon, lat)

def get_LGL_1970s_DOP_by_coord(latitude, longitude, output_dir=None):
    """
    Downloads DOP images from the 1970s for a single coordinate point.
    Creates a 20x20m bounding box around the point.
    """
    if output_dir is None:
        output_dir = os.path.join(script_dir, "Downloads/LGL/Historical/1970_1979")
    x_coordinate, y_coordinate = wgs84_to_utm32(latitude, longitude)
    image_width = 20  # meter
    found_usable_image = False  # Flag to track if any usable image was found

    # prepare paths
    os.makedirs(output_dir, exist_ok=True)

    wms_url = "https://owsproxy.lgl-bw.de/owsproxy/ows/WMS_LGL-BW_HIST_DOP_1970-1979?"

    try:
        wms = WebMapService(wms_url, version='1.3.0')

        for year in range(1970, 1980):
            print(f"\nexamine layer {year}...")
            try:
                # download image
                img = wms.getmap(
                    layers=[str(year)],
                    srs='EPSG:25832',
                    bbox=(x_coordinate - image_width, y_coordinate - image_width,
                          x_coordinate + image_width, y_coordinate + image_width),
                    size=(512, 512),
                    format='image/jpeg'
                )

                # save image temporarily to check, whether it is empty or contains information
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    tmp_file.write(img.read())
                    tmp_path = tmp_file.name

                # check image
                if is_image_blank(tmp_path):
                    print(f"Layer {year} doesn´t contain a usable image.")
                    os.remove(tmp_path)
                else:
                    found_usable_image = True
                    # define target directory to save image
                    filename = f"{latitude:.6f}_{longitude:.6f}_LGL_{year}.jpg"
                    final_path = os.path.join(output_dir, filename)
                    shutil.move(tmp_path, final_path)
                    print(f"✅ Layer {year} saved at: {final_path}")

            except Exception as e:
                print(f"❌ Error caused by layer {year}: {e}")

        if not found_usable_image:
            print("\n❌ No usable image found for given coordinates in any year of the 70s")
            return False
        return True

    except Exception as e:
        print(f"\n❌ Error building WMS-connection: {e}")
        print(f"Hint: double-check WMS-URL {wms_url}")
        return False

def get_LGL_1970s_DOP_by_bbox(lat1, lon1, lat2, lon2, output_dir=None):
    """
    Downloads DOP images from the 1970s for a bounding box defined by two coordinates:
    (lat1, lon1) - top left corner
    (lat2, lon2) - bottom right corner
    """
    if output_dir is None:
        output_dir = os.path.join(script_dir, "Downloads/LGL/Historical/1970_1979")
    
    # Convert WGS84 coordinates to UTM32
    x1, y1 = wgs84_to_utm32(lat1, lon1)  # Top-left
    x2, y2 = wgs84_to_utm32(lat2, lon2)  # Bottom-right
    
    found_usable_image = False  # Flag to track if any usable image was found

    # prepare paths
    os.makedirs(output_dir, exist_ok=True)

    wms_url = "https://owsproxy.lgl-bw.de/owsproxy/ows/WMS_LGL-BW_HIST_DOP_1970-1979?"

    try:
        wms = WebMapService(wms_url, version='1.3.0')

        for year in range(1970, 1980):
            print(f"\nexamine layer {year}...")
            try:
                # download image - using the bbox directly from the converted coordinates
                img = wms.getmap(
                    layers=[str(year)],
                    srs='EPSG:25832',
                    bbox=(x1, y2, x2, y1),  # Note: WMS bbox format is (minx, miny, maxx, maxy)
                    size=(1024, 1024),  # Higher resolution for larger area
                    format='image/jpeg'
                )

                # save image temporarily to check, whether it is empty or contains information
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    tmp_file.write(img.read())
                    tmp_path = tmp_file.name

                # check image
                if is_image_blank(tmp_path):
                    print(f"Layer {year} doesn´t contain a usable image.")
                    os.remove(tmp_path)
                else:
                    found_usable_image = True
                    # define target directory to save image
                    filename = f"{lat1:.6f}_{lon1:.6f}__{lat2:.6f}_{lon2:.6f}_LGL_{year}.jpg"
                    final_path = os.path.join(output_dir, filename)
                    shutil.move(tmp_path, final_path)
                    print(f"✅ Layer {year} saved at: {final_path}")

            except Exception as e:
                print(f"❌ Error caused by layer {year}: {e}")

        if not found_usable_image:
            print("\n❌ No usable image found for given bbox in any year of the 70s")
            return False
        return True

    except Exception as e:
        print(f"\n❌ Error building WMS-connection: {e}")
        print(f"Hint: double-check WMS-URL {wms_url}")
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
            print(f"\nDownloading 1970s DOP for coordinate: {lat}, {lon}")
            return get_LGL_1970s_DOP_by_coord(lat, lon)
            
        elif len(parts) == 4:
            # Bounding box mode
            lat1, lon1, lat2, lon2 = parts
            print(f"\nDownloading 1970s DOP for bounding box: ({lat1}, {lon1}) to ({lat2}, {lon2})")
            return get_LGL_1970s_DOP_by_bbox(lat1, lon1, lat2, lon2)
            
        else:
            print("Invalid input format. Please provide either 2 or 4 comma-separated coordinates.")
            return False
            
    except ValueError:
        print("Wrong format! Input needs to match pattern: '47.9924817937077, 7.82889116037526' or")
        print("'47.9924817937077, 7.82889116037526, 47.9920000000000, 7.82900000000000'")
        return False

if __name__ == "__main__":
    success = get_input()