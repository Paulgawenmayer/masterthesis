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
    download_LGL_1968_orthophoto.py:        This script downloads a 20x20 m DOP of the year 1968 for a given coordinate in Baden-Wuerttemberg.
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


from config import API_KEY



def get_images_by_address(address): # transform address to coordinates
    print(f"\nLooking for address: {address}")
    coords = get_coordinates(address)
    if coords:
        validate_address(address)
        get_images_by_coordinates(coords[0], coords[1])


def get_images_by_coordinates(latitude, longitude): # Download images for given coordinates
    print(f"\nLooking for download for: Latitude {latitude}, Longitude {longitude}")    
    #get_GM_DOP_by_coord(API_KEY, latitude, longitude)  # --> load GM-DOP for only one coordinate
    get_GM_DOP_by_bbox(*get_building_polygon_for_coords(latitude, longitude)) # --> load GM-DOP for bbox  
    #get_GSV_photo(API_KEY, latitude, longitude) # load GSV-image
    #get_LGL_1960s_DOP(latitude, longitude) # load all available images for the 60s
    #get_LGL_1970s_DOP(latitude, longitude) # Lload all available images for the 70s
    #get_LGL_1980s_DOP(latitude, longitude) # load all available images for the 80s
    #get_LGL_1990s_DOP(latitude, longitude) # load all available images for the 90s
    #get_LGL_2000s_DOP(latitude, longitude) # load all available images for the 2000s
    #get_LGL_2007_2023_DOP(latitude, longitude) # load all available images from 2007-2023
    #get_latest_LGL_DOP(latitude, longitude) # load latest LGL-DOP
    #get_LGL_CIR_DOP(latitude, longitude) # load latest LGL-CIR-DOP
    #get_LGL_grayscale_DOP(latitude, longitude) # load latest LGL-Grayscale-DOP

    

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