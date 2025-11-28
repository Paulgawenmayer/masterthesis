"""
CIR = COLOR-Infrared
This script downloads the latest DOP for a given coordinate or bounding box 
in Baden-Wuerttemberg as CIR (Color-Infrared).
"""
import os
import sys
from owslib.wms import WebMapService
import matplotlib.pyplot as plt
from pyproj import Transformer

# Set import path to path of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from image_has_content_checker import is_image_blank

# Function for coordinate conversion: WGS84 (EPSG:4326) → UTM Zone 32N (EPSG:25832)
def wgs84_to_utm32(lat, lon):
    """Convert WGS84 coordinates to UTM32/EPSG:25832"""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
    return transformer.transform(lon, lat)
    
def get_LGL_CIR_DOP_by_coord(latitude, longitude, output_dir=None):
    """
    Downloads CIR DOP for a single coordinate point.
    Creates a 20x20m bounding box around the point.
    """
    if output_dir is None:
        # Prepare absolute path of this script to ensure downloaded data will be saved in correct directory
        output_dir = os.path.join(script_dir, "Downloads/LGL/CIR")
    os.makedirs(output_dir, exist_ok=True)

    x_coordinate, y_coordinate = wgs84_to_utm32(latitude, longitude)

    # Parameters
    area = 20  # Meters
    
    # WMS URL of LGL-BW
    wms_url = "https://owsproxy.lgl-bw.de/owsproxy/ows/WMS_LGL-BW_ATKIS_DOP_20_CIR?"
    
    try:
        # Initialize WMS client
        wms = WebMapService(wms_url, version='1.3.0')
    
        # Request image from WMS server
        img = wms.getmap(
            layers=['IMAGES_DOP_20_CIR'],
            srs='EPSG:25832',
            bbox=(x_coordinate - area, y_coordinate - area,
                  x_coordinate + area, y_coordinate + area),
            size=(512, 512),
            format='image/jpeg'
        )
    
        # Filename with coordinates
        filename = f"{latitude:.6f}_{longitude:.6f}_LGL_CIR.jpg"
        filepath = os.path.join(output_dir, filename)
    
        # Save image
        with open(filepath, "wb") as f:
            f.write(img.read())
    
        # Display metadata
        print("\nCoordinates successfully recognized:")
        print(f'Latitude: "{latitude:.6f}"')
        print(f'Longitude: "{longitude:.6f}"')
        print(f'Image saved at: "{filepath}"\n')
        
        return True
    
    except Exception as e:
        print(f"❌ Error retrieving map: {e}")
        print(f"Tip: Check WMS URL: {wms_url}")
        return False

def get_LGL_CIR_DOP_by_bbox(lat1, lon1, lat2, lon2, output_dir=None):
    """
    Downloads CIR DOP for a bounding box defined by two coordinates:
    (lat1, lon1) - top left corner
    (lat2, lon2) - bottom right corner
    """
    if output_dir is None:
        output_dir = os.path.join(script_dir, "Downloads/LGL/CIR")
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert WGS84 coordinates to UTM32
    x1, y1 = wgs84_to_utm32(lat1, lon1)  # Top-left
    x2, y2 = wgs84_to_utm32(lat2, lon2)  # Bottom-right
    
    # WMS URL of LGL-BW
    wms_url = "https://owsproxy.lgl-bw.de/owsproxy/ows/WMS_LGL-BW_ATKIS_DOP_20_CIR?"
    
    try:
        # Initialize WMS client
        wms = WebMapService(wms_url, version='1.3.0')
    
        # Request image from WMS server using the bbox
        img = wms.getmap(
            layers=['IMAGES_DOP_20_CIR'],
            srs='EPSG:25832',
            bbox=(x1, y2, x2, y1),  # Note: WMS bbox format is (minx, miny, maxx, maxy)
            size=(1024, 1024),  # Higher resolution for larger area
            format='image/jpeg'
        )
    
        # Filename with bbox coordinates
        filename = f"{lat1:.6f}_{lon1:.6f}__{lat2:.6f}_{lon2:.6f}_LGL_CIR.jpg"
        filepath = os.path.join(output_dir, filename)
    
        # Save image
        with open(filepath, "wb") as f:
            f.write(img.read())
    
        # Display metadata
        print("\nBounding box coordinates successfully processed:")
        print(f'Top-left: "{lat1:.6f}, {lon1:.6f}"')
        print(f'Bottom-right: "{lat2:.6f}, {lon2:.6f}"')
        print(f'Image saved at: "{filepath}"\n')
        
        return True
    
    except Exception as e:
        print(f"❌ Error retrieving map: {e}")
        print(f"Tip: Check WMS URL: {wms_url}")
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
            print(f"\nDownloading CIR DOP for coordinate: {lat}, {lon}")
            return get_LGL_CIR_DOP_by_coord(lat, lon)
            
        elif len(parts) == 4:
            # Bounding box mode
            lat1, lon1, lat2, lon2 = parts
            print(f"\nDownloading CIR DOP for bounding box: ({lat1}, {lon1}) to ({lat2}, {lon2})")
            return get_LGL_CIR_DOP_by_bbox(lat1, lon1, lat2, lon2)
            
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
        print("Error occurred during processing.")