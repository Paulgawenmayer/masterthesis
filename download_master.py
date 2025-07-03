#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 12:19:02 2025

@author: paulmayer

This is the master-script. It enables you to download orthophotos and Street-View-photos 
from customer-choosen areas. Either at scale of single buildings or for all buildings 
within a choosen region/frame. Images can be downloaded for different instants of time (s. below). Each script 
can also be run independently on itself.

EXPLANATION OF SUBSCRIPTS: 
    config.py:                              This script provides the API-Key, nescssary for some scripts in this folder to get access to google APIs
    transform_address_to_coordinates.py:    This script transforms and returns a given address into coordinates in WGS84.
    download_GM_orthophoto.py               This script downloads a Google-Maps image of a given coordinate at the highest resolution & zoom possible.  
    download_GSV_photo.py                   This script downloads the nearest Google-Street-View image to a given coordinate (which should represent a house),
                                            adjusting heading, fov and pitch, to capture the building the best way possible. 
    download_LGL_1968_orthophoto.py         This script downloads a 20x20 m DOP of the year 1968 for a given coordinate in Baden-Wuerttemberg.
    download_LGL_1975_ohrthophoto.py        This script downloads a 20x20 m DOP of the year 1975 for a given coordinate in Baden-Wuerttemberg.
                                            UNFORTUNATELY there are currently some server issues at LGL, which causes  malfunction in data-provision.
    download_LGL_1984_orthophoto.py         This script downloads a 20x20 m DOP of the year 1984 for a given coordinate in Baden-Wuerttemberg.
    download_LGL_1995_orthophoto.py         This script downloads a 20x20 m DOP of the year 1995 for a given coordinate in Baden-Wuerttemberg.
    download_LGL_2000_orthophoto.py         This script downloads a 20x20 m DOP of the year 2000 for a given coordinate in Baden-Wuerttemberg.
    download_LGL_2007_2023_orthophoto.py    This script downloads several 20x20 m DOP's of different years for a given coordinate in Baden-Wuerttemberg.
    download_LGL_latest_orthophoto.py       This script downloads the latest 20x20 m DOP for a given coordinate in Baden-Wuerttemberg.
    download_LGL_CIR_orthophoto.py          This script downloads the latest 20x20m DOP for a given coordinate in Baden-Wuerttemberg as CIR.
    download_LGL_grayscale_orthophoto.py    This script downloads the latest DOP as 20x20 m for a given coordinate in in Baden-Wuerttemberg in grayscale format
    

USAGE: 
    1. Get a single building: 
            - address-based
            - coordinate-based
        
            
    2. Get all buildings within a chosen region/frame
            - coordinate-based
    
"""

import sys
import os
# import argparse -- create CLI when everything else is done

# set import path to path of this master script to avoid path-related-import-problems
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)


# import of modules
from transform_address_to_coordinates import get_coordinates, validate_address
from download_GM_orthophoto import get_GM_DOP_by_coord, get_GM_DOP_by_bbox
from download_GSV_photo import get_GSV_photo
from download_LGL_60s_orthophoto import get_LGL_1960s_DOP
from download_LGL_70s_orthophoto import get_LGL_1970s_DOP
from download_LGL_80s_orthophoto import get_LGL_1980s_DOP
from download_LGL_90s_orthophoto import get_LGL_1990s_DOP
from download_LGL_2000s_orthophoto import get_LGL_2000s_DOP
from download_LGL_2007_2023_orthophoto import get_LGL_2007_2023_DOP
from download_LGL_latest_orthophoto import get_latest_LGL_DOP
from download_LGL_CIR_orthophoto import get_LGL_CIR_DOP
from download_LGL_grayscale_orthophoto import get_LGL_grayscale_DOP
from get_OSM_building_bbox_for_coordinates import get_building_polygon_for_coords
from address_finder_from_coordinates import reverse_geocode
from double_image_deleter import delete_duplicate_images
# from data_augmentator import data_augmentator


from config import API_KEY



def create_directory(address_or_coords):
    """
    Creates a directory under script_dir + "/Downloads" with the name of the validated address
    (for address input) or the address determined by reverse_geocode (for coordinate input).
    """
    downloads_path = os.path.join(script_dir, "Downloads")
    if not os.path.exists(downloads_path):
        os.makedirs(downloads_path)

    # Prüfen ob Input Koordinaten sind
    def is_coords(val):
        try:
            lat, lon = map(float, [x.strip() for x in val.split(',')])
            return lat, lon
        except Exception:
            return None

    coords = is_coords(address_or_coords)
    if coords:
        # Koordinaten: reverse_geocode nutzen
        lat, lon = coords
        address = reverse_geocode(lat, lon, API_KEY, verbose=False)
        if not address:
            address = f"{lat}_{lon}"
    else:
        # Address: use validate_address
        address = validate_address(address_or_coords)
        if not address:
            address = address_or_coords

    # Clean directory name (remove invalid characters)
    import re
    safe_address = re.sub(r'[^\w\-_\. ]', '_', address)
    dir_path = os.path.join(downloads_path, safe_address)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def get_images_by_address(address): # transform address to coordinates
    print(f"\nLooking for address: {address}")
    coords = get_coordinates(address)
    if coords:
        validated = validate_address(address)
        dir_path = create_directory(address)
        print(f"Directory created: {dir_path}")
        get_images_by_coordinates(coords[0], coords[1], output_dir=dir_path)


def get_images_by_coordinates(latitude, longitude, output_dir=None): # Download images for given coordinates
    print(f"\nLooking for download for: Latitude {latitude}, Longitude {longitude}")    
    if output_dir is None:
        output_dir = create_directory(f"{latitude}, {longitude}")
    print(f"Directory created: {output_dir}")
    # get_GM_DOP_by_coord(API_KEY, latitude, longitude, folder=output_dir)  # --> load GM-DOP for only one coordinate and NOT for a building with its dimensions
    get_GM_DOP_by_bbox(*get_building_polygon_for_coords(latitude, longitude), folder=output_dir) # --> load GM-DOP for bbox  
    get_GSV_photo(API_KEY, latitude, longitude, folder=output_dir) # load GSV-image
    get_LGL_1960s_DOP(latitude, longitude, output_dir=output_dir) # load all available images for the 60s
    get_LGL_1970s_DOP(latitude, longitude, output_dir=output_dir) # load all available images for the 70s
    get_LGL_1980s_DOP(latitude, longitude, output_dir=output_dir) # load all available images for the 80s
    get_LGL_1990s_DOP(latitude, longitude, output_dir=output_dir) # load all available images for the 90s
    get_LGL_2000s_DOP(latitude, longitude, output_dir=output_dir) # load all available images for the 2000s
    get_LGL_2007_2023_DOP(latitude, longitude, output_dir=output_dir) # load all available images from 2007-2023
    get_latest_LGL_DOP(latitude, longitude, output_dir=output_dir) # load latest LGL-DOP
    get_LGL_CIR_DOP(latitude, longitude, output_dir=output_dir) # load latest LGL-CIR-DOP
    get_LGL_grayscale_DOP(latitude, longitude, output_dir=output_dir) # load latest LGL-Grayscale-DOP
    delete_duplicate_images(output_dir)  # delete double images in output_dir
    # data_augmentator(output_dir, 10)  # augment images in output_dir the number sets the number of augmentations per image
    

def is_coordinate_input(user_input):
    try:
        lat, lon = map(float, [x.strip() for x in user_input.split(',')])
        return True
    except ValueError:
        return False

def main():
    user_input = input("Please enter an address or coordinates in WGS84 (Latitude, Longitude): ").strip()
    
    if is_coordinate_input(user_input):
        lat, lon = map(float, user_input.split(','))
        get_images_by_coordinates(lat, lon)
    else:
        get_images_by_address(user_input)

if __name__ == "__main__":
    main()