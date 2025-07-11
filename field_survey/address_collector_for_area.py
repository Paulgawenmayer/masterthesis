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

PROGRESS_FILE = os.path.join(script_dir, "address_collection_progress.json")  # File to save progress in case of interruptions due to rate limiting or other issues
RATE_LIMIT = 2000  # max requests per minute
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


def get_addresses_in_polygon(wkt_polygon, grid_step=0.00015, resume=False, total_points=None):
    """
    Given a WKT polygon, queries the Google Geocoding API for all addresses within the polygon.
    Results are saved as a CSV file in the script directory.
    grid_step: step size in degrees for the grid (smaller = more precise, but more API calls)
    resume: if True, resume from last saved progress
    total_points: total number of grid points for progress display
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

    # Prepare progress bar
    if total_points is None:
        total_points = count_grid_points_in_polygon(wkt_polygon, grid_step)
    points_done = 0

    # Prepare CSV: write header if file does not exist
    if not os.path.exists(output_csv):
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["coordinates", "address"])

    lat = start_lat
    while lat <= maxy:
        lon = start_lon if lat == start_lat else minx
        while lon <= maxx:
            point = Point(lon, lat)
            if polygon.contains(point):
                # Rate limiting: max x requests/minute
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
                        # Write address immediately to CSV
                        coordinates = f"{address_components.get('address_latitude')}, {address_components.get('address_longitude')}" if address_components.get('address_latitude') is not None and address_components.get('address_longitude') is not None else ''
                        with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow([
                                coordinates,
                                address_components.get('address')
                            ])
                # Save progress
                save_progress(lat, lon, wkt_polygon)
                points_done += 1
                percent = (points_done / total_points) * 100
                print(f"Progress: {points_done}/{total_points} ({percent:.2f}%)", end='\r', flush=True)
            lon += grid_step
            time.sleep(SLEEP_BETWEEN_REQUESTS)
        lat += grid_step

    print()  # Newline after progress bar
    print(f"Saved addresses to {output_csv}")
    # Call address_sorter.py to sort the CSV
    import subprocess
    subprocess.run([sys.executable, os.path.join(script_dir, "address_sorter.py")], check=True)
    # Delete progress file when finished
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
            result = data["results"][0]
            components = result["address_components"]
            geometry = result.get("geometry", {})
            location = geometry.get("location", {})
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
            address = f"{street or ''} {number or ''}, {postcode or ''} {city or ''}, {country or ''}".strip()
            return {
                "street": street,
                "number": number,
                "postcode": postcode,
                "city": city,
                "country": country,
                "address": address,
                "address_latitude": location.get("lat"),
                "address_longitude": location.get("lng")
            }
        else:
            print(f"API error for ({lat}, {lon}): HTTP {response.status_code}, status: {data.get('status')}, message: {data.get('error_message')}")
    except Exception as e:
        print(f"Error in reverse geocoding: {e}")
    return None

def count_grid_points_in_polygon(wkt_polygon, grid_step=0.00015):
    from shapely import wkt
    from shapely.geometry import Point
    polygon = wkt.loads(wkt_polygon)
    minx, miny, maxx, maxy = polygon.bounds
    count = 0
    lat = miny
    while lat <= maxy:
        lon = minx
        while lon <= maxx:
            point = Point(lon, lat)
            if polygon.contains(point):
                count += 1
            lon += grid_step
        lat += grid_step
    return count

if __name__ == "__main__":
    resume = False
    wkt_input = None
    grid_step = 0.00015
    # Check if progress exists
    if os.path.exists(PROGRESS_FILE):
        progress = load_progress()
        if progress and progress.get("wkt"):
            print("Progress file found.")
            wkt_input = progress["wkt"]
            print("Calculating grid points for the saved polygon...")
            n_points = count_grid_points_in_polygon(wkt_input, grid_step)
            print(f"The polygon contains approx. {n_points} grid points (API calls).")
            answer = input("Resume from last saved point and continue? (y/n): ").strip().lower()
            if answer == 'y':
                resume = True
            else:
                wkt_input = input("Please enter a new WKT polygon: ").strip()
        else:
            wkt_input = input("Please enter a WKT polygon: ").strip()
    else:
        wkt_input = input("Please enter a WKT polygon: ").strip()
    # Always calculate grid points before starting and ask for confirmation
    n_points = count_grid_points_in_polygon(wkt_input, grid_step)
    print(f"The polygon contains approx. {n_points} grid points (API calls).")
    answer = input("Do you want to start the process now? (y/n): ").strip().lower()
    if answer == 'y':
        get_addresses_in_polygon(wkt_input, grid_step=grid_step, resume=resume, total_points=n_points)
    else:
        print("Aborted.")
