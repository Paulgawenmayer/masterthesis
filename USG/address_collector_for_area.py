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
import re
import time
import json
from collections import deque

# Load API key from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import API_KEY

# set path back to the script directory to ensure output is saved in this directory
script_dir = os.path.dirname(os.path.abspath(__file__))

PROGRESS_FILE = os.path.join(script_dir, "address_collection_progress.json") # File to save progress in case of interruptions due to rate limiting or other issues
RATE_LIMIT = 1500  # max requests per minute
SLEEP_BETWEEN_REQUESTS = 60.0 / RATE_LIMIT  # seconds


def save_progress(lat, lon, wkt_polygon):
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump({"lat": lat, "lon": lon, "wkt": wkt_polygon}, f)

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def clear_progress():
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)


def get_addresses_in_polygon(wkt_polygon, grid_step=0.00015, resume=False):
    """
    Given a WKT polygon, queries the Google Geocoding API for all addresses within the polygon.
    Results are saved as a CSV file in the script directory.
    grid_step: step size in degrees for the grid (smaller = more precise, but more API calls)
    resume: if True, resume from last saved progress
    """
    output_csv = os.path.join(script_dir, "addresses_in_area.csv")
    polygon = wkt.loads(wkt_polygon)
    minx, miny, maxx, maxy = polygon.bounds
    addresses = set()
    results = []

    # Resume support
    start_lat = miny
    start_lon = minx
    if resume:
        progress = load_progress()
        if progress and progress.get("wkt") == wkt_polygon:
            start_lat = progress["lat"]
            start_lon = progress["lon"]
            print(f"Resuming from lat={start_lat}, lon={start_lon}")
        else:
            print("No matching progress found or WKT changed. Starting from beginning.")

    lat = start_lat
    while lat <= maxy:
        lon = start_lon if lat == start_lat else minx
        while lon <= maxx:
            point = Point(lon, lat)
            if polygon.contains(point):
                # Rate limiting: max 1500 requests/minute
                now = time.time()
                request_times = deque()
                while request_times and now - request_times[0] > 60:
                    request_times.popleft()
                if len(request_times) >= RATE_LIMIT:
                    sleep_time = 60 - (now - request_times[0]) + 0.01
                    print(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                request_times.append(time.time())

                address_components = reverse_geocode_components(lat, lon)
                if address_components:
                    addr_tuple = tuple(address_components.values())
                    if addr_tuple not in addresses:
                        addresses.add(addr_tuple)
                        results.append(address_components)
                # Fortschritt speichern
                save_progress(lat, lon, wkt_polygon)
            lon += grid_step
            time.sleep(SLEEP_BETWEEN_REQUESTS)
        lat += grid_step

    # Sort by street, then house number (numerically if possible)
    def extract_street_and_number(address):
        match = re.match(r"^(.*?)(?:\s+(\d+[a-zA-Z]?))?(,|$)", address)
        if match:
            street = match.group(1).strip() if match.group(1) else ''
            number = match.group(2) if match.group(2) else ''
            return street, number
        return '', ''
    def house_number_key(num):
        if not num:
            return float('inf')
        match = re.match(r"(\d+)", str(num))
        if match:
            return int(match.group(1))
        return float('inf')
    # Prepare list of unique addresses
    unique_addresses = list({row['address'] for row in results if 'address' in row and row['address']})
    # Sort by street and house number
    unique_addresses.sort(key=lambda addr: (extract_street_and_number(addr)[0], house_number_key(extract_street_and_number(addr)[1])))
    # Write results to CSV (single column)
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["address"])
        for addr in unique_addresses:
            writer.writerow([addr])
    print(f"Saved {len(unique_addresses)} addresses to {output_csv}")
    # Fortschrittsdatei löschen, wenn fertig
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)


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
            # Compose full address for sorting
            address = f"{street or ''} {number or ''}, {postcode or ''} {city or ''}, {country or ''}".strip()
            return {"street": street, "number": number, "postcode": postcode, "city": city, "country": country, "address": address}
        else:
            print(f"API error for ({lat}, {lon}): HTTP {response.status_code}, status: {data.get('status')}, message: {data.get('error_message')}")
    except Exception as e:
        print(f"Error in reverse geocoding: {e}")
    return None

if __name__ == "__main__":
    wkt_input = input("Please enter a WKT polygon: ").strip()
    resume = False
    if os.path.exists(PROGRESS_FILE):
        answer = input("Progress file found. Resume from last saved point? (y/n): ").strip().lower()
        if answer == 'y':
            resume = True
    get_addresses_in_polygon(wkt_input, resume=resume)
