"""
This script returns a bounding-box capturing the building nearest to given coordinates. This is to 
prepare the best frame for the download_GM_orthophoto.py script in the download_master.py
"""

import osmnx as ox
import warnings
from shapely.geometry import Polygon, MultiPolygon, Point, box

warnings.filterwarnings('ignore', message='Geometry is in a geographic CRS')

def get_building_polygon_for_coords(latitude, longitude, tags={"building": True}, dist=20):
    """
    Get building polygon for given coordinates
    Args:
        latitude: float, latitude coordinate
        longitude: float, longitude coordinate
        tags: dict, OSM tags to filter buildings
        dist: int, search distance in meters
    Returns:
        Tuple of four coordinates (lat1, lon1, lat2, lon2) representing bounding box (NW, SE)
    """
    # Create point from input coordinates
    point = (latitude, longitude)
    
    # 1. Download buildings around location
    try:
        gdf = ox.features_from_point(point, tags=tags, dist=dist)
    except Exception as e:
        print(f"Overpass query failed: {e}")
        return None

    # 2. Select nearest building
    if gdf.empty:
        print("No buildings found in the area.")
        return None

    gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
    gdf['distance'] = gdf.geometry.centroid.distance(Point(longitude, latitude))
    gdf = gdf.sort_values("distance")

    geom = gdf.iloc[0].geometry
    if isinstance(geom, MultiPolygon):
        geom = list(geom.geoms)[0]

    coords = list(geom.exterior.coords)
    print(f"\n✅ Building found. Vertices ({len(coords)}):")
    for i, (lon, lat) in enumerate(coords):
        print(f"Corner {i+1}: Latitude {lat:.6f}, Longitude {lon:.6f}")

    # Calculate bounding box coordinates
    min_lon = min(lon for lon, lat in coords)
    max_lon = max(lon for lon, lat in coords)
    min_lat = min(lat for lon, lat in coords)
    max_lat = max(lat for lon, lat in coords)
    
    # Return as four separate values: lat1, lon1, lat2, lon2
    return (max_lat, min_lon, min_lat, max_lon)

def get_coordinates():
    input_str = input("Enter coordinates (Format: 'latitude, longitude'): ")
    
    try:
        latitude, longitude = map(float, [x.strip() for x in input_str.split(',')])
        return latitude, longitude
    except ValueError:
        print("Invalid format! Please use format '48.12345, 10.12345'.")
        return None, None
    
if __name__ == "__main__":
    latitude, longitude = get_coordinates()
    
    if latitude is not None and longitude is not None:
        result = get_building_polygon_for_coords(latitude, longitude)
        if result:
            lat1, lon1, lat2, lon2 = result
            print(f"\nBBox coordinates for nearest building:")
            print(f"Top-Left (NW): {lat1:.6f}, {lon1:.6f}")
            print(f"Bottom-Right (SE): {lat2:.6f}, {lon2:.6f}")