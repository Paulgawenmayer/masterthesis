#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 15:54:44 2025

@author: paulmayer

This script downloads a 20x20 m DOP of a year in the 90´s for a given coordinate in Baden-Wuerttemberg.
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

def get_coordinates():
    input_str = input("Enter coordinates (Format: 'latitude, longitude'): ")
    try:
        latitude, longitude = map(float, [x.strip() for x in input_str.split(',')])
        return latitude, longitude
    except ValueError:
        print("Wrong format! Input needs to match following pattern: 47.9924817937077, 7.82889116037526")
        return None, None

def wgs84_to_utm32(lat, lon):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
    return transformer.transform(lon, lat)

def get_LGL_2000s_DOP(latitude, longitude, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(script_dir, "Downloads/LGL/Historical/2000_2009")
    x_coordinate, y_coordinate = wgs84_to_utm32(latitude, longitude)
    image_width = 20  # meter
    found_usable_image = False  # Flag to track if any usable image was found

    # prepare paths
    os.makedirs(output_dir, exist_ok=True)

    wms_url = "https://owsproxy.lgl-bw.de/owsproxy/ows/WMS_LGL-BW_HIST_DOP_2000-2009?"

    try:
        wms = WebMapService(wms_url, version='1.3.0')

        for year in range(2000, 2010):
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

                    # show image
                    #plt.figure(figsize=(6, 6))
                    #plt.imshow(plt.imread(final_path))
                    #plt.axis('off')
                    #plt.show()

            except Exception as e:
                print(f"❌ Error caused by layer {year}: {e}")

        if not found_usable_image:
            print("\n❌ No usable image found for given coordinates in any year of the 2000s")
            return False
        return True

    except Exception as e:
        print(f"\n❌ Error building WMS-connection: {e}")
        print(f"Hint: double-check WMS-URL {wms_url}")
        return False

if __name__ == "__main__":
    lat, lon = get_coordinates()
    if lat is not None and lon is not None:
        success = get_LGL_2000s_DOP(lat, lon)



