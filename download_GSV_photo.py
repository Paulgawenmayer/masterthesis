"""
This script downloads the nearest Google-Street-View image to a given coordinate,
using Google Maps' default parameters for heading, field of view and pitch.
"""

import requests
import matplotlib.pyplot as plt
from PIL import Image
import io
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

def get_GSV_photo(api_key, latitude, longitude, width=600, height=400, folder=None):
    """
    Downloads Google Street View image using default Google parameters.
    
    Parameters:
    - api_key: Google API key
    - latitude, longitude: Coordinates of the target location
    - width, height: Image dimensions in pixels
    - folder: Destination folder for the downloaded image
    """
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

    # Get Street View image with default parameters
    # Note: By not specifying heading, fov and pitch, the API will use defaults
    url = (
        f"https://maps.googleapis.com/maps/api/streetview?"
        f"size={width}x{height}&"
        f"location={latitude},{longitude}&"
        f"key={api_key}"
    )
    
    print("Using Google Maps default parameters for heading, field of view, and pitch")
    
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(full_path, 'wb') as file:
            file.write(response.content)
        print(f"Street View image successfully saved as:\n{full_path}")
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