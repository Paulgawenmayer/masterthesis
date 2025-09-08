"""
This script downloads several 20x20 m DOP's of different years for a given coordinate in Baden-Wuerttemberg.
"""
import os
import sys
from owslib.wms import WebMapService
import matplotlib.pyplot as plt
from pyproj import Transformer
import tempfile
import shutil

# Set import path to path of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from image_has_content_checker import is_image_blank

def wgs84_to_utm32(lat, lon):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
    return transformer.transform(lon, lat)

def get_coordinates():
    input_str = input("Enter coordinates (Format: 'latitude, longitude'): ")
    try:
        latitude, longitude = map(float, [x.strip() for x in input_str.split(',')])
        return latitude, longitude
    except ValueError:
        print("Invalid format! Please use format '48.12345, 10.12345'.")
        return None, None    

def get_LGL_2007_2023_DOP(latitude, longitude, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(script_dir, "Downloads/LGL/Historical/2007_2023")
    os.makedirs(output_dir, exist_ok=True)
    
    x_coordinate, y_coordinate = wgs84_to_utm32(latitude, longitude)
    found_usable_image = False  # Flag to track if any usable image was found
    
    # Parameters
    area = 20  # Meters
    # base_dir = os.path.join(script_dir, "Downloads", "LGL", "Historical")
    # os.makedirs(base_dir, exist_ok=True)
    
    # WMS URL for historical DOP
    wms_url = "https://owsproxy.lgl-bw.de/owsproxy/ows/WMS_LGL-BW_ATKIS_HIST_DOP_20_RGB?"
    # layer_list = [ ... ]
    
    try:
        # Initialize WMS
        wms = WebMapService(wms_url, version='1.3.0')
        # Dynamically get all layer names from the WMS
        layer_list = list(wms.contents.keys())
    
        for layer in layer_list:
            print(f"\nExamining layer {layer}...")
            if output_dir is not None:
                year_dir = output_dir
            else:
                year_dir = os.path.join(script_dir, "Downloads/LGL/Historical/2007_2023", layer)
                os.makedirs(year_dir, exist_ok=True)
            
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
                    import re
                    match = re.match(r"(\d{4})", layer)
                    if match:
                        year = match.group(1)
                    else:
                        year = "unknown"
                    # Define final path and move the temporary file
                    filename = f"{latitude:.6f}_{longitude:.6f}_LGL_{year}.jpg"
                    filepath = os.path.join(year_dir, filename)
                    shutil.move(tmp_path, filepath)
                    print("Coordinates successfully recognized:")
                    print(f'Latitude: "{latitude:.6f}"')
                    print(f'Longitude: "{longitude:.6f}"')
                    print(f'Image saved at: "{filepath}"')
                   
                    # Display image
                    #plt.figure(figsize=(6, 6))
                    #plt.imshow(plt.imread(final_path))
                    #plt.axis('off')
                    #plt.show()
                    
            except Exception as e:
                print(f"❌ Error processing layer {layer}: {str(e)}")
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        
        if not found_usable_image:
            print("\n❌ No usable images found for the given coordinates in any layer (2007-2023)")
            return False
        return True
        
    except Exception as e:
        print(f"\n❌ Error establishing WMS connection: {e}")
        print(f"Tip: Verify the WMS URL: {wms_url}")
        return False

if __name__ == "__main__":
    lat, lon = get_coordinates()
    if lat is not None and lon is not None:
        success = get_LGL_2007_2023_DOP(lat, lon)
        if not success:
            print("No valid images found for the provided coordinates.")
