"""
This script returns the address most likely belonging to given coordinates. This method is the foundation of addres_collecter_for_area.py.
"""

import requests
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from config import API_KEY


# Verbose toggles printing of additional information
def reverse_geocode(lat, lng, api_key, verbose=True):
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "latlng": f"{lat},{lng}",
        "key": api_key
    }
    response = requests.get(url, params=params)
    data = response.json()

    if response.status_code != 200:
        if verbose:
            print("Error fetching data:", response.status_code)
        return None

    if data.get("status") != "OK":
        if verbose:
            print("Geocoding error:", data.get("status"))
        return None

    results = data.get("results", [])
    if results:
        address = results[0]["formatted_address"]
        if verbose:
            print("Address:", address)
        return address
    else:
        if verbose:
            print("No address found.")
        return None


def get_coordinates():
    input_str = input("Enter coordinates (Format: 'latitude, longitude'): ")
    
    try:
        latitude, longitude = map(float, [x.strip() for x in input_str.split(',')])
        return latitude, longitude
    except ValueError:
        print("Invalid format! Please use format '48.12345, 10.12345'.")
        return None, None


# Beispielhafte Nutzung
if __name__ == "__main__":
    latitude, longitude = get_coordinates()
    api_key = API_KEY

    reverse_geocode(latitude, longitude, api_key)