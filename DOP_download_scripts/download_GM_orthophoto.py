"""
This script downloads a Google-Maps-Tile of a given coordinate or bbox at the highest resolution & zoom possible. 
As Google has changed its Terms and conditions (only projects created before 8th of July. 2025 can request images from Static Maps API), 
another API_KEY (from an older Project) has to be used here. This one is NOT free in usage though, so do not use it too much!
"""

import requests
import matplotlib.pyplot as plt
from PIL import Image
import io
import sys
import os
from math import cos, pi
from geopy.distance import geodesic

# Set import path to path of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from config import API_KEY_GM

# Set export path
export_folder = os.path.join(script_dir, "Downloads", "Maps")
os.makedirs(export_folder, exist_ok=True)

def get_GM_DOP_by_coord(api_key, latitude, longitude, zoom=20, width=200, height=200, folder=None):
    """
    This function downloads a Google Maps DOP image for a specific coordinate.
    """
    
    if folder is None:
        folder = export_folder

    filename = f"{latitude:.6f}_{longitude:.6f}_coord.png"
    full_path = os.path.join(folder, filename)

    url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={latitude},{longitude}&"
        f"zoom={zoom}&"
        f"size={width}x{height}&"
        f"maptype=satellite&"
        f"key={api_key}"
    )
    print(url)

    response = requests.get(url)

    if response.status_code == 200:
        with open(full_path, 'wb') as file:
            file.write(response.content)
        print(f"📍 Image saved at: {full_path}")

        # display the image
        #plt.figure(figsize=(5, 5))
        #plt.imshow(Image.open(io.BytesIO(response.content)))
        #plt.axis('off')
        #plt.show()
        
    else:
        print(f"HTTP Error: {response.status_code}")
        print(f"Response content: {response.text}")

def meters_per_pixel(zoom: int, latitude: float) -> float:
    earth_circumference = 40075016.686  # meters
    return earth_circumference * cos(latitude * pi / 180) / (2 ** (zoom + 8))

def calculate_pixel_dimensions(lat1, lon1, lat2, lon2, zoom=20, scale=2, max_dim=1280):
    center_lat = (lat1 + lat2) / 2
    mpp = meters_per_pixel(zoom, center_lat)

    lat_distance = geodesic((lat1, lon1), (lat2, lon1)).meters
    lon_distance = geodesic((lat1, lon1), (lat1, lon2)).meters

    width_px = min(int(lon_distance / mpp * scale), max_dim)
    height_px = min(int(lat_distance / mpp * scale), max_dim)

    return width_px, height_px

def get_GM_DOP_by_bbox(lat1, lon1, lat2, lon2, api_key=API_KEY_GM, zoom=20, scale=2, folder=None):
    """
    This function downloads a Google Maps DOP image for a specific bounding box.
    """
    if folder is None:
        folder = export_folder

    center_lat = (lat1 + lat2) / 2
    center_lon = (lon1 + lon2) / 2

    width_px, height_px = calculate_pixel_dimensions(lat1, lon1, lat2, lon2, zoom, scale)

    filename = f"{lat1:.6f}_{lon1:.6f}__{lat2:.6f}_{lon2:.6f}_GM.png"
    full_path = os.path.join(folder, filename)

    url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={center_lat},{center_lon}&"
        f"zoom={zoom}&"
        f"size={width_px}x{height_px}&"
        f"maptype=satellite&"
        f"scale={scale}&"
        f"key={api_key}"
    )

    response = requests.get(url)

    if response.status_code == 200:
        with open(full_path, 'wb') as file:
            file.write(response.content)
        print(f"BBOX Image saved at: {full_path}")

        # display the image
        #plt.figure(figsize=(6, 6))
        #plt.imshow(plt.imread(full_path))
        #plt.axis('off')
        #plt.show()
    else:
        print(f"HTTP Error: {response.status_code}")

def get_input_coordinates():
    input_str = input("Enter coordinates (format: 'lat,lon' for point or 'lat1,lon1,lat2,lon2' for bbox): ").strip()

    parts = [p.strip() for p in input_str.split(',')]
    
    if len(parts) == 2:
        try:
            lat, lon = map(float, parts)
            return (lat, lon)
        except:
            print("Invalid coordinate format. Expected 'lat,lon'")
            return None
    elif len(parts) == 4:
        try:
            lat1, lon1, lat2, lon2 = map(float, parts)
            return (lat1, lon1, lat2, lon2)
        except:
            print("Invalid coordinate format. Expected 'lat1,lon1,lat2,lon2'")
            return None
    else:
        print("Invalid input. Enter either 2 coordinates (lat,lon) or 4 coordinates (lat1,lon1,lat2,lon2)")
        return None

if __name__ == "__main__":
    coords = get_input_coordinates()

    if coords is None:
        print("Aborted due to input error.")
    elif len(coords) == 2:
        lat, lon = coords
        print(f"\n📍 coordinates registered: {lat}, {lon}")
        get_GM_DOP_by_coord(API_KEY_GM, lat, lon)
    elif len(coords) == 4:
        lat1, lon1, lat2, lon2 = coords
        print(f"\nBBOX registered: NW=({lat1},{lon1}), SE=({lat2},{lon2})")
        get_GM_DOP_by_bbox(lat1, lon1, lat2, lon2)