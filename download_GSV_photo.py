"""
This script downloads the nearest Google-Street-View image to a given coordinate (which should belong to a house),
adjusting heading, fov, to capture the building the best way possible. Pitch is set to 10, as a default value.
"""

import requests
import matplotlib.pyplot as plt
from PIL import Image
import io
import math
from geopy.distance import geodesic
from datetime import datetime
import os
import sys

# Set import path to path of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from config import API_KEY

# Set export path
export_folder = os.path.join(script_dir, "Downloads", "StreetView")


def check_streetview_availability(api_key, latitude, longitude):
    url = (
        f"https://maps.googleapis.com/maps/api/streetview/metadata?"
        f"location={latitude},{longitude}&"
        f"key={api_key}"
    )
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data  # Return complete metadata object
    else:
        print(f"Error in metadata request: HTTP {response.status_code}")
        return None

def get_GSV_photo(api_key, latitude, longitude, heading=0, fov=70, pitch=10, width=600, height=400, folder=None):
    if folder is None:
        folder = export_folder

    # Check availability and get metadata
    metadata = check_streetview_availability(api_key, latitude, longitude)
    
    if not metadata or metadata.get("status") != "OK":
        print("No Street View image available for these coordinates.")
        return

    # Print metadata
    print("\nStreet View Metadata:")
    print(f"• Location: {metadata['location']['lat']:.6f}, {metadata['location']['lng']:.6f}")
    print(f"• Panorama ID: {metadata.get('pano_id', 'N/A')}")
    print(f"• Date: {metadata.get('date', 'N/A')}")
    print(f"• Copyright Info: {metadata.get('copyright', 'N/A')}\n")

    # Create filename with actual coordinates from metadata
    filename = f"{metadata['location']['lat']:.6f}_{metadata['location']['lng']:.6f}_StreetView.png"
    
    # Full path
    full_path = os.path.join(folder, filename)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    # Calculate heading between GSV and target coordinate
    def calculate_heading(lat1, lon1, lat2, lon2):
        """Calculate heading from point 1 to point 2. All inputs/outputs in degrees."""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lon_rad = math.radians(lon2 - lon1)

        x = math.sin(delta_lon_rad) * math.cos(lat2_rad)
        y = math.cos(lat1_rad) * math.sin(lat2_rad) - \
            math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon_rad)

        initial_bearing = math.atan2(x, y)
        initial_bearing_deg = math.degrees(initial_bearing)
        compass_bearing = (initial_bearing_deg + 360) % 360

        return compass_bearing

    heading = calculate_heading(float(f"{metadata['location']['lat']:.6f}"), float(f"{metadata['location']['lng']:.6f}"), latitude, longitude)
    print("Calculated heading: ", round(heading,2), "°")

    # Print pitch
    print("Calculated pitch: ", round(pitch,2), "°")
    
    # Calculate distance between GSV and coordinate
    def calculate_distance(lat1, lon1, lat2, lon2):
        point1 = (lat1, lon1)
        point2 = (lat2, lon2)
        return round(geodesic(point1, point2).meters, 2)

    distance = calculate_distance(float(f"{metadata['location']['lat']:.6f}"), float(f"{metadata['location']['lng']:.6f}"), latitude, longitude)
    print("Distance to target coordinate: ", distance, "m")
    
    # Calculate FOV
    def calculate_fov(distance):
        """Calculate optimal FOV value for a 10x10m object based on distance."""
        object_size = 10  # 10 meters
        fov_rad = 2 * math.atan(object_size / (1 * distance))
        fov_deg = math.degrees(fov_rad)
        return max(20, min(90, round(fov_deg)))

    fov = calculate_fov(distance)
    print("Selected FOV = ", fov)
    
    # Calculate image age
    date_str = metadata.get('date', '')
    current_date = datetime.now()
    try:
        if '-' in date_str:
            streetview_date = datetime.strptime(date_str, "%Y-%m")
        else:
            streetview_date = datetime.strptime(date_str, "%Y")
        
        delta = current_date - streetview_date
        age = delta.days / 365.2425
        print(f"Image age: {age:.2f} year(s)")
    
    except (ValueError, AttributeError):
        age = None
        print("Image age: Could not be calculated (missing/incomplete date information)")

    # Get Street View image
    url = (
        f"https://maps.googleapis.com/maps/api/streetview?"
        f"size={width}x{height}&"
        f"location={latitude},{longitude}&"
        f"heading={heading}&"
        f"fov={fov}&"
        f"pitch={pitch}&"
        f"key={api_key}"
    )
    
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(full_path, 'wb') as file:
            file.write(response.content)
        print(f"Street View image successfully saved as:\n{full_path}")
             
        # Display image
        #plt.figure(figsize=(6, 5))
        #plt.imshow(Image.open(io.BytesIO(response.content)))
        #plt.axis('off')
        #plt.show()
        
    else:
        print(f"Error retrieving image: HTTP {response.status_code}")

def get_coordinates():
    input_str = input("Enter coordinates (Format: 'latitude, longitude'): ")
    
    try:
        latitude, longitude = map(float, [x.strip() for x in input_str.split(',')])
        return latitude, longitude
    except ValueError:
        print("Invalid format! Please use format '48.12345, 10.12345'.")
        return None, None

if __name__ == "__main__":
    latitude, longitude = get_coordinates()
    
    if latitude is not None and longitude is not None:
        print("\nCoordinates successfully recognized:")
        print(f"Latitude: {latitude}")
        print(f"Longitude: {longitude}")
        get_GSV_photo(API_KEY, latitude, longitude)
    else:
        print("Error processing coordinates.")



