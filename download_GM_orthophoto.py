#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 12:37:17 2025

@author: paulmayer

This script downloads a Google-Maps Tile of a given coordinate at the highest resolution & zoom possible.  
"""

import requests
import matplotlib.pyplot as plt
from PIL import Image
import io
import sys
import os

# Set import path to path of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from config import API_KEY


# Set export path
export_folder = script_dir + "/Downloads/Maps"

def get_GM_DOP(api_key, latitude, longitude, zoom=20, width=200, height=200, folder=export_folder):
    # zoom = 20 --> Maximum resolution/zoom level for satellite images (up to 5 cm per pixel in cities)
    # width & height in pixels
    
    # Create filename from coordinates
    filename = f"{latitude:.6f}_{longitude:.6f}.png"
    
    # Full path for Mac (~/Downloads/Maps/)
    full_path = os.path.expanduser(f"{folder}/{filename}")
    os.makedirs(os.path.dirname(export_folder), exist_ok=True)
    
    # Build URL
    url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={latitude},{longitude}&"
        f"zoom={zoom}&"
        f"size={width}x{height}&"
        f"maptype=satellite&"
        f"key={api_key}"
    )
    
    response = requests.get(url)
    
    if response.status_code == 200:
        # Save image
        with open(full_path, 'wb') as file:
            file.write(response.content)
        print(f"Image saved at: {full_path}")
        
        # Display image
        plt.figure(figsize=(5, 5))
        plt.imshow(Image.open(io.BytesIO(response.content)))
        plt.axis('off')
        plt.show()
        
    else:
        print(f"Error: HTTP {response.status_code}")

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
        print(f"Latitude:  {latitude}")
        print(f"Longitude: {longitude}")
        get_GM_DOP(API_KEY, latitude, longitude)
    else:
        print("Error processing coordinates.")
