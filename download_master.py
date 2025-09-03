"""
This is the master-script. It enables you to download orthophotos and Street-View-photos 
from customer-choosen areas. Either at scale of single buildings or for a batch of buildings listed in 
address_list.csv. Images can be downloaded for different instants of time (s. below). Each script 
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
    download_data_distributor.py            This script distributes downloaded data from the "Downloads" folder to the "training_datasets/colored" directory.

USAGE: 
    1. Get a single building: 
            - address-based
            - coordinate-based
        
            
    2. Get all buildings within a chosen region/frame
            - coordinate-based
    
"""

import sys
import os
import pandas as pd
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
from download_data_distributor import distribute_downloaded_data

from config import API_KEY



def create_directory(address_or_coords):
    """
    Creates a directory under script_dir + "/Downloads" with the name of the validated address
    (for address input) or the address determined by reverse_geocode (for coordinate input).
    
    Directory name is normalized:
    - Spaces are replaced with underscores
    - Double underscores are replaced with single underscores
    - Invalid characters are replaced with underscores
    """
    downloads_path = os.path.join(script_dir, "Downloads")
    if not os.path.exists(downloads_path):
        os.makedirs(downloads_path)

    # Test if input is coordinates
    def is_coords(val):
        try:
            lat, lon = map(float, [x.strip() for x in val.split(',')])
            return lat, lon
        except Exception:
            return None

    coords = is_coords(address_or_coords)
    if coords:
        # Coordinates: use reverse_geocode
        lat, lon = coords
        address = reverse_geocode(lat, lon, API_KEY, verbose=False)
        if not address:
            address = f"{lat}_{lon}"
    else:
        # Address: use validate_address
        address = validate_address(address_or_coords)
        if not address:
            address = address_or_coords

    # Clean directory name and normalize it:
    # 1. Replace spaces with underscores
    # 2. Remove invalid characters
    # 3. Replace double underscores with single ones
    import re
    safe_address = address.replace(' ', '_')  # Replace spaces with underscores
    safe_address = re.sub(r'[^\w\-_\.]', '_', safe_address)  # Replace invalid chars (note: no space in allowed chars)
    
    # Replace double underscores with single ones until no more doubles exist
    while '__' in safe_address:
        safe_address = safe_address.replace('__', '_')
    
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
    else:
        print(f"Could not geocode address: {address}")


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
    distribute_downloaded_data(os.path.join(script_dir, "Downloads"), os.path.join(script_dir, "training_datasets", "colored")) # distributes image-data form downlaod-section to training_datasets

def is_coordinate_input(user_input):
    try:
        lat, lon = map(float, [x.strip() for x in user_input.split(',')])
        return True
    except ValueError:
        return False

def batch_process_addresses():
    """Process all addresses from the CSV file"""
    csv_path = os.path.join(script_dir, "field_survey", "survey_summary", "address_list.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: Address list not found at {csv_path}")
        return
    
    try:
        # Read the address list
        address_df = pd.read_csv(csv_path)
        
        if "Address" not in address_df.columns:
            print("Error: CSV file doesn't contain an 'Address' column")
            return
        
        addresses = address_df["Address"].dropna().astype(str).tolist()
        total_addresses = len(addresses)
        
        print(f"Found {total_addresses} addresses to process")
        confirm = input(f"Do you want to proceed with downloading data for all {total_addresses} addresses? (y/n): ")
        
        if confirm.lower() != "y":
            print("Batch processing cancelled")
            return
        
        # Process each address
        for idx, address in enumerate(addresses, 1):
            print(f"\n[{idx}/{total_addresses}] Processing address: {address}")
            try:
                get_images_by_address(address)
                print(f"Completed address {idx}/{total_addresses}")
            except Exception as e:
                print(f"Error processing address {address}: {str(e)}")
                continue
            
        print(f"\nBatch processing complete! Processed {total_addresses} addresses.")
        
    except Exception as e:
        print(f"Error during batch processing: {str(e)}")

def main():
    # First, ask if user wants batch processing or single address
    batch_mode = input("Do you want to execute the batch-process (y) or enter a single coordinate/address (n)? ").strip().lower()
    
    if batch_mode == "y":
        # Batch process all addresses from the CSV
        batch_process_addresses()
    else:
        # Single address/coordinate mode (original behavior)
        user_input = input("Please enter an address or coordinates in WGS84 (Latitude, Longitude): ").strip()
        
        if is_coordinate_input(user_input):
            lat, lon = map(float, user_input.split(','))
            get_images_by_coordinates(lat, lon)
        else:
            get_images_by_address(user_input)

if __name__ == "__main__":
    main()