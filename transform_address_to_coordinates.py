"""
This script transforms and returns a given address into coordinates in WGS84.
"""

import requests
from config import API_KEY


def validate_address(address):
    url = f"https://addressvalidation.googleapis.com/v1:validateAddress?key={API_KEY}"
    payload = {
        "address": {
            "addressLines": [address]
        }
    }

    response = requests.post(url, json=payload)

    if response.status_code != 200:
        print(f"Error with Address Validation API: {response.status_code} {response.text}")
        return None

    result = response.json()

    formatted_address = result.get("result", {}).get("address", {}).get("formattedAddress", None)

    # Explizite Ersetzung von "Deutschland" durch "Germany"
    if formatted_address and "Deutschland" in formatted_address:
        formatted_address = formatted_address.replace("Deutschland", "Germany")

    #print(f"\nValidated address: {formatted_address}")

    # return for Geocoding API:
    if formatted_address:
        return formatted_address
    else:
        return address  # fallback → original address

def get_coordinates(address):
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": address,
        "key": API_KEY
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"Geocoding API error: {response.status_code} {response.text}")
        return None

    result = response.json()

    if result["status"] != "OK":
        print(f"Geocoding failed: {result['status']} {result.get('error_message', '')}")
        return None

    location = result["results"][0]["geometry"]["location"]
    lat = location["lat"]
    lng = location["lng"]

    return lat, lng

def main():
    print("Enter an address (free format):")
    user_input = input("Address: ")

    # Step 1: validate address
    address_for_geocoding = validate_address(user_input)

    # Step 2: get coordinates for address
    coords = get_coordinates(address_for_geocoding)

    if coords:
        print("\n📍 Coordinates for Address:")
        print(f"Latitude (Lat): {coords[0]}")
        print(f"Longitude (Lng): {coords[1]}")
    else:
        print("No coordinates could be found")

if __name__ == "__main__":
    main()