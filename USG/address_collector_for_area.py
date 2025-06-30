"""
Script to collect all addresses within a given WKT polygon using the Google Geocoding API.
The API key is loaded from the config.py in the parent directory.
Addresses are saved as a CSV file.

Script works as follows: given a wkt polygon it iterates over a grid of points within the polygon,
queries the Google Geocoding API for each point, and saves the unique addresses found to a CSV file.
"""
import os
import sys
import csv
from shapely import wkt
from shapely.geometry import box, Point
import requests

# Load API key from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import API_KEY

# set path back to the script directory to ensure output is saved in this directory
script_dir = os.path.dirname(os.path.abspath(__file__))

def get_addresses_in_polygon(wkt_polygon, grid_step=0.00015):
    """
    Given a WKT polygon, queries the Google Geocoding API for all addresses within the polygon.
    Results are saved as a CSV file in the script directory.
    grid_step: step size in degrees for the grid (smaller = more precise, but more API calls)
    """
    output_csv = os.path.join(script_dir, "addresses_in_area.csv")
    polygon = wkt.loads(wkt_polygon)
    minx, miny, maxx, maxy = polygon.bounds
    addresses = set()
    results = []
    
    lat = miny
    while lat <= maxy:
        lon = minx
        while lon <= maxx:
            point = Point(lon, lat)
            if polygon.contains(point):
                address_components = reverse_geocode_components(lat, lon)
                if address_components:
                    addr_tuple = tuple(address_components.values())
                    if addr_tuple not in addresses:
                        addresses.add(addr_tuple)
                        results.append(address_components)
            lon += grid_step
        lat += grid_step

    # Sort by street, then house number (numerically if possible)
    def house_number_key(num):
        import re
        if num is None:
            return float('inf')
        match = re.match(r"(\d+)", str(num))
        if match:
            return int(match.group(1))
        return float('inf')
    results.sort(key=lambda row: (row['street'] or '', house_number_key(row['number'])))

    # Write results to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["street", "number", "postcode", "city", "country"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Saved {len(results)} addresses to {output_csv}")


def reverse_geocode_components(lat, lon):
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "latlng": f"{lat},{lon}",
        "key": API_KEY
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if response.status_code == 200 and data.get("status") == "OK":
            components = data["results"][0]["address_components"]
            def get_component(types):
                for comp in components:
                    if any(t in comp["types"] for t in types):
                        return comp.get("long_name")
                return None
            street = get_component(["route"])
            number = get_component(["street_number"])
            postcode = get_component(["postal_code"])
            city = get_component(["locality", "postal_town", "administrative_area_level_2"])
            country = get_component(["country"])
            return {"street": street, "number": number, "postcode": postcode, "city": city, "country": country}
    except Exception as e:
        print(f"Error in reverse geocoding: {e}")
    return None

if __name__ == "__main__":
    wkt_input = input("Please enter a WKT polygon: ").strip()
    get_addresses_in_polygon(wkt_input)
