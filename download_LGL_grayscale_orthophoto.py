#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 15:11:08 2025

@author: paulmayer

This script downloads the latest DOP as 20x20 m for a given coordinate in in Baden-Wuerttemberg in grayscale format
"""
import os
from owslib.wms import WebMapService
import matplotlib.pyplot as plt
from pyproj import Transformer

# Function for coordinate conversion: WGS84 (EPSG:4326) → UTM Zone 32N (EPSG:25832)
def wgs84_to_utm32(lat, lon):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
    return transformer.transform(lon, lat)

def get_coordinates():
    # Open input field
    input_str = input("Enter coordinates (Format: 'latitude, longitude'): ")
   
    try:
        # Split string and convert to float
        latitude, longitude = map(float, [x.strip() for x in input_str.split(',')])
        return latitude, longitude
    except ValueError:
        print("Invalid format! Please use format '48.12345, 10.12345'.")
        return None, None   
    
def get_LGL_grayscale_DOP(latitude, longitude, output_dir=None):
    if output_dir is None:
        # Prepare absolute path of this script to ensure downloaded data will be saved in correct directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "Downloads/LGL/Grayscale")
    os.makedirs(output_dir, exist_ok=True)
    
    x_coordinate, y_coordinate = wgs84_to_utm32(latitude, longitude)
    
    # Parameters
    area = 20  # Meters
    
    # WMS URL of LGL-BW
    wms_url = "https://owsproxy.lgl-bw.de/owsproxy/ows/WMS_LGL-BW_ATKIS_DOP_20_GRAU?"
    
    try:
        # Initialize WMS client
        wms = WebMapService(wms_url, version='1.3.0')
    
        # Request image from WMS server
        img = wms.getmap(
            layers=['IMAGES_DOP_20_GRAU'],
            srs='EPSG:25832',
            bbox=(x_coordinate - area, y_coordinate - area,
                  x_coordinate + area, y_coordinate + area),
            size=(512, 512),
            format='image/jpeg'
        )
    
        # Filename with coordinates
        filename = f"{latitude:.6f}_{longitude:.6f}_LGL_Grayscale.jpg"
        filepath = os.path.join(output_dir, filename)
    
        # Save image
        with open(filepath, "wb") as f:
            f.write(img.read())
    
        # Display metadata
        print("\nCoordinates successfully recognized:")
        print(f'Latitude: "{latitude:.6f}"')
        print(f'Longitude: "{longitude:.6f}"')
        print(f'Image saved at: "{filepath}"\n')
    
        # Display image
        #plt.figure(figsize=(6, 6))
        #plt.imshow(plt.imread(final_path))
        #plt.axis('off')
        #plt.show()
    
    except Exception as e:
        print(f"❌ Error retrieving map: {e}")
        print(f"Tip: Check WMS URL: {wms_url}")

if __name__ == "__main__":
    get_LGL_grayscale_DOP(*get_coordinates())
