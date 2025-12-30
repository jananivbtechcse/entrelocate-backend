from flask import Flask, request, jsonify
from flask_cors import CORS
import math
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
import aiohttp
import asyncio
import concurrent.futures
import statistics
import overpy
import time
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point
import requests
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler
from shapely.ops import unary_union
import warnings
import random
import os

app = Flask(__name__)

# CORS configuration - Replace with your actual frontend URL
FRONTEND_URL = os.getenv('FRONTEND_URL', 'http://localhost:3000')
CORS(app, resources={r"/api/*": {"origins": [FRONTEND_URL, "https://*.vercel.app", "https://*.netlify.app"]}})

# ============================================================================
# FREE NOMINATIM API FOR GEOCODING (NO API KEY NEEDED)
# ============================================================================

async def get_city_bounding_box_nominatim(city_name):
    """
    Get city bounding box using free Nominatim API (OpenStreetMap)
    No API key required!
    """
    base_url = "https://nominatim.openstreetmap.org/search"
    
    headers = {
        'User-Agent': 'LocationAnalyzer/1.0'
    }
    
    # Try multiple search strategies
    search_strategies = [
        # Strategy 1: Search for the city specifically (not the union territory)
        {
            'q': f"{city_name} city",
            'format': 'json',
            'limit': 5,
            'addressdetails': 1,
            'featuretype': 'city'
        },
        # Strategy 2: Original search
        {
            'q': city_name,
            'format': 'json',
            'limit': 5,
            'addressdetails': 1
        }
    ]
    
    try:
        async with aiohttp.ClientSession() as session:
            for strategy_num, params in enumerate(search_strategies, 1):
                await asyncio.sleep(1)  # Rate limiting
                
                async with session.get(base_url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if not data:
                            continue
                        
                        # Find the best result - prefer city/town over state/region
                        best_result = None
                        for result in data:
                            result_type = result.get('type', '').lower()
                            osm_type = result.get('osm_type', '').lower()
                            display_name = result.get('display_name', '')
                            
                            # For Puducherry, avoid "state" or "administrative" results
                            # Prefer "city", "town", "municipality"
                            if result_type in ['city', 'town', 'municipality', 'administrative'] and osm_type == 'relation':
                                # Check if it's not the huge union territory
                                boundingbox = result['boundingbox']
                                min_lat = float(boundingbox[0])
                                max_lat = float(boundingbox[1])
                                min_lon = float(boundingbox[2])
                                max_lon = float(boundingbox[3])
                                
                                # Calculate the size of the bounding box
                                lat_diff = max_lat - min_lat
                                lon_diff = max_lon - min_lon
                                
                                # If the box is too large (> 1 degree), it's probably a state/territory
                                # A typical city should be < 0.5 degrees
                                if lat_diff < 1.0 and lon_diff < 1.0:
                                    best_result = result
                                    print(f"Strategy {strategy_num}: Found suitable result: {display_name}")
                                    break
                            
                            # Also accept smaller results
                            elif result_type in ['city', 'town', 'suburb', 'neighbourhood']:
                                best_result = result
                                print(f"Strategy {strategy_num}: Found {result_type}: {display_name}")
                                break
                        
                        if best_result:
                            boundingbox = best_result['boundingbox']
                            min_lat = float(boundingbox[0])
                            max_lat = float(boundingbox[1])
                            min_lon = float(boundingbox[2])
                            max_lon = float(boundingbox[3])
                            display_name = best_result.get('display_name', '')
                            
                            print(f"Found location: {display_name}")
                            print(f"Bounding box for {city_name}: {min_lat}, {min_lon}, {max_lat}, {max_lon}")
                            print(f"Box size: {max_lat - min_lat:.4f}° x {max_lon - min_lon:.4f}°")
                            
                            return min_lat, min_lon, max_lat, max_lon, display_name
                
                print(f"Strategy {strategy_num} failed, trying next...")
            
            print(f"All strategies failed for {city_name}")
            return None, None, None, None, None
                
    except Exception as e:
        print(f"Error getting bounding box: {e}")
        return None, None, None, None, None

# ============================================================================
# FREE OVERPASS API FOR LOCATION DATA (NO API KEY NEEDED)
# ============================================================================

async def fetch_places_from_overpass(city, category):
    """
    Fetch places using the free Overpass API with improved city handling
    """
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    # Map common business categories to OSM tags
    category_mapping = {
        'restaurant': 'amenity=restaurant',
        'cafe': 'amenity=cafe',
        'bar': 'amenity=bar',
        'pub': 'amenity=pub',
        'grocery': 'shop=supermarket',
        'supermarket': 'shop=supermarket',
        'pharmacy': 'amenity=pharmacy',
        'hotel': 'tourism=hotel',
        'gym': 'leisure=fitness_centre',
        'fitness': 'leisure=fitness_centre',
        'bank': 'amenity=bank',
        'hospital': 'amenity=hospital',
        'school': 'amenity=school',
        'store': 'shop',
        'shop': 'shop',
        'retail': 'shop',
    }
    
    # Get the OSM tag for the category
    osm_tag = category_mapping.get(category.lower(), 'amenity')
    
    # ALWAYS use bounding box approach for reliability
    print(f"Fetching places for {city} using bounding box method...")
    return await fetch_places_with_bbox_fallback(city, osm_tag)


async def fetch_places_with_bbox_fallback(city, osm_tag):
    """Primary method using bounding box from Nominatim"""
    bbox_data = await get_city_bounding_box_nominatim(city)
    min_lat, min_lon, max_lat, max_lon, display_name = bbox_data
    
    if min_lat is None:
        return pd.DataFrame(), f"Could not find location: {city}"
    
    # Verify we got the right city
    print(f"Using coordinates for: {display_name}")
    
    overpass_url = "http://overpass-api.de/api/interpreter"
    bbox_query = f"""
    [out:json][timeout:60];
    (
      node[{osm_tag}]({min_lat},{min_lon},{max_lat},{max_lon});
      way[{osm_tag}]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out center 150;
    """
    
    try:
        async with aiohttp.ClientSession() as session:
            await asyncio.sleep(2)  # Rate limiting
            timeout = aiohttp.ClientTimeout(total=90)
            async with session.get(
                overpass_url,
                params={'data': bbox_query},
                headers={"User-Agent": "LocationAnalyzer/1.0"},
                timeout=timeout
            ) as response:
                if response.status != 200:
                    return pd.DataFrame(), f"Overpass API failed with status {response.status}"
                
                data = await response.json()
                if 'elements' not in data or not data['elements']:
                    return pd.DataFrame(), f"No places found in {city} for this category"
                
                rows = []
                for place in data['elements']:
                    tags = place.get('tags', {})
                    
                    if 'center' in place:
                        lat = place['center']['lat']
                        lon = place['center']['lon']
                    else:
                        lat = place.get('lat')
                        lon = place.get('lon')
                    
                    if lat is None or lon is None:
                        continue
                    
                    # Verify the place is within our bounding box (double check)
                    if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
                        continue
                    
                    address_parts = []
                    for key in ['addr:housenumber', 'addr:street', 'addr:city', 'addr:postcode']:
                        if key in tags:
                            address_parts.append(tags[key])
                    
                    # If no address, use the display_name's city
                    if not address_parts and display_name:
                        city_part = display_name.split(',')[0] if ',' in display_name else city
                        address_parts = [city_part]
                    
                    address = ', '.join(address_parts) if address_parts else city
                    
                    place_name = tags.get('name', 'Unnamed')
                    rows.append({
                        "Name": place_name,
                        "Rating": round(random.uniform(3.0, 5.0), 1),
                        "Address": address,
                        "Latitude": lat,
                        "Longitude": lon
                    })
                
                print(f"\n{'='*60}")
                print(f"Found {len(rows)} places in {city}")
                print(f"{'='*60}")
                print("\nPlace Names:")
                for i, row in enumerate(rows[:20], 1):  # Show first 20
                    print(f"{i:2d}. {row['Name']:40s} | {row['Address'][:40]}")
                if len(rows) > 20:
                    print(f"... and {len(rows) - 20} more places")
                print(f"{'='*60}\n")
                
                return pd.DataFrame(rows), None
                
    except Exception as e:
        return pd.DataFrame(), str(e)


async def fetch_and_process_data(city, category):
    """
    Use Overpass API instead of Foursquare for free data
    """
    return await fetch_places_from_overpass(city, category)


async def get_data_for_city(city, business):
    return await fetch_and_process_data(city, business)


async def get_shops_in_bounding_box_by_city(city, limit=200):
    """
    Fetch shops for a city by first getting its bounding box
    """
    bbox_data = await get_city_bounding_box_nominatim(city)
    min_lat, min_lon, max_lat, max_lon, display_name = bbox_data
    
    if min_lat is None:
        print(f"Could not find bounding box for {city}")
        return []
    
    print(f"Fetching shops for: {display_name}")
    return await get_shops_in_bounding_box(min_lat, min_lon, max_lat, max_lon, limit)


async def get_shops_in_bounding_box(min_lat, min_lon, max_lat, max_lon, limit=200):
    """
    Fetch shops within a bounding box using the Overpass API.
    """
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    overpass_query = f"""
    [out:json][timeout:60];
    (
      node["shop"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["shop"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out center {limit};
    """

    headers = {
        "Accept-Encoding": "gzip, deflate",
        "User-Agent": "MyApp/1.0"
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                await asyncio.sleep(2)  # Rate limiting
                timeout = aiohttp.ClientTimeout(total=90)
                async with session.get(
                    overpass_url, 
                    params={'data': overpass_query}, 
                    headers=headers,
                    timeout=timeout
                ) as response:
                    if response.status == 504:
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 5
                            print(f"Timeout on attempt {attempt + 1}, retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            print(f"Failed after {max_retries} attempts")
                            return []
                    
                    if response.status != 200:
                        print(f"Failed to retrieve data. Status code: {response.status}")
                        return []
                    
                    data = await response.json()

                    if 'elements' not in data:
                        print("No data found in the response.")
                        return []

                    shops = data['elements']
                    print(f"Found {len(shops)} shops.")

                    shop_data = []
                    for shop in shops:
                        if 'tags' in shop:
                            if 'center' in shop:
                                lat = shop['center']['lat']
                                lon = shop['center']['lon']
                            else:
                                lat = shop.get('lat')
                                lon = shop.get('lon')

                            if lat is not None and lon is not None:
                                # Verify within bounding box
                                if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                                    shop_data.append({
                                        'name': shop['tags'].get('name', 'Unnamed shop'),
                                        'type': shop['tags'].get('shop', 'Unspecified'),
                                        'lat': lat,
                                        'lon': lon
                                    })

                    return shop_data

        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                print(f"Timeout on attempt {attempt + 1}, retrying...")
                await asyncio.sleep((attempt + 1) * 5)
                continue
            else:
                print("Request timed out after all retries.")
                return []
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep((attempt + 1) * 5)
                continue
            return []
    
    return []


# ============================================================================
# CLUSTERING AND ANALYSIS FUNCTIONS (KEEP YOUR EXISTING CODE)
# ============================================================================

def is_outlier(df, lat_col='Latitude', lon_col='Longitude', threshold=3):
    """Detect outliers using MAD"""
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    
    median_lat = df[lat_col].median()
    mad_lat = (df[lat_col] - median_lat).abs().median()
    
    median_lon = df[lon_col].median()
    mad_lon = (df[lon_col] - median_lon).abs().median()
    
    if mad_lat == 0:
        mad_lat = 1e-9
    if mad_lon == 0:
        mad_lon = 1e-9
    
    modified_z_score_lat = (df[lat_col] - median_lat).abs() / mad_lat
    modified_z_score_lon = (df[lon_col] - median_lon).abs() / mad_lon
    
    outliers_lat = modified_z_score_lat > threshold
    outliers_lon = modified_z_score_lon > threshold
    
    outliers = outliers_lat | outliers_lon
    outliers |= df[lat_col].isna() | df[lon_col].isna()
    
    return outliers


def clean_coordinates(df, lat_col='Latitude', lon_col='Longitude'):
    """Clean the DataFrame by removing invalid coordinates"""
    df = df.copy()
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    
    df = df[
        (df[lat_col].between(-90, 90)) &
        (df[lon_col].between(-180, 180))
    ].reset_index(drop=True)
    
    outliers = is_outlier(df, lat_col, lon_col)
    df_cleaned = df[~outliers].reset_index(drop=True)
    
    return df_cleaned


def perform_analysis(df):
    """Your existing analysis function"""
    df = df.dropna(subset=['Latitude', 'Longitude'])
    df = clean_coordinates(df)
    
    print(f"After cleaning: {len(df)} records")
    
    data_latlong = df[['Latitude', 'Longitude']].values
    
    if len(data_latlong) == 0:
        return "No data for clustering"
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_latlong)
    
    sil_scores = []
    K = range(2, min(20, len(data_latlong)))
    
    for k in K:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(data_scaled)
        score = silhouette_score(data_scaled, cluster_labels)
        sil_scores.append(score)
    
    optimal_k = K[np.argmax(sil_scores)]
    print(f"Optimal K: {optimal_k}")
    
    kmeans = KMeans(n_clusters=optimal_k, n_init=5, random_state=42)
    kmeans.fit(data_latlong)
    df['cluster'] = kmeans.labels_
    
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df['Rating'].fillna(df['Rating'].median(), inplace=True)
    
    average_ratings = df.groupby('cluster')['Rating'].mean()
    sorted_avg_ratings = average_ratings.sort_values()
    last_5_min_clusters = sorted_avg_ratings.head(max(math.ceil(optimal_k/1.5), 2))
    
    cluster_counts = Counter(df['cluster'])
    last_5_min_cluster_counts = {cluster: cluster_counts[cluster] for cluster in last_5_min_clusters.index}
    pop_d = max(math.ceil(optimal_k/3), 2)
    top_3_min_clusters = dict(sorted(last_5_min_cluster_counts.items(), key=lambda item: item[1])[:pop_d])
    
    top_3_cluster_data = [
        {'cluster': cluster, 'population': len(df[df['cluster'] == cluster]) * 100}
        for cluster in top_3_min_clusters.keys()
    ]
    
    top_3_cluster_df = pd.DataFrame(top_3_cluster_data)
    cluster_bounding_boxes_sorted = top_3_cluster_df.sort_values(by='population', ascending=False).reset_index(drop=True)
    cluster_bounding_boxes_sorted['score'] = [(i/pop_d)*100 for i in range(pop_d, 0, -1)]
    
    recommended_clusters = []

    for i, row in cluster_bounding_boxes_sorted.iterrows():
        cluster_id = int(row['cluster'])
        cluster_points = df[df['cluster'] == cluster_id]

        centroid_lat = float(cluster_points['Latitude'].mean())
        centroid_lng = float(cluster_points['Longitude'].mean())

        traffic_score = (
            len(cluster_points) * 2 +
            cluster_points['Rating'].mean() * 10
        )

        neighborhood_areas = (
            cluster_points['Address']
            .dropna()
            .unique()
            .tolist()
        )

        recommended_clusters.append({
            "cluster": f"Cluster {cluster_id + 1}",
            "latitude": round(centroid_lat, 6),
            "longitude": round(centroid_lng, 6),
            "neighborhood_areas": neighborhood_areas if neighborhood_areas else ["N/A"],
            "places_count": int(len(cluster_points)),
            "traffic_score": round(float(traffic_score), 1)
        })
    
    return recommended_clusters


# ============================================================================
# BUSINESS ANALYSIS HELPER FUNCTIONS
# ============================================================================

category_mapping = {
    "supermarket": "Retail",
    "convenience": "Retail",
    "grocery": "Retail",
    "greengrocer": "Retail",
    "department_store": "Retail",
    "bakery": "Food & Beverage",
    "butcher": "Food & Beverage",
    "beverages": "Food & Beverage",
    "seafood": "Food & Beverage",
    "dairy": "Food & Beverage",
    "pharmacy": "Health",
    "chemist": "Health",
    "optician": "Health",
    "hairdresser": "Apparel",
    "jewellery": "Apparel",
    "clothes": "Apparel",
    "shoe": "Apparel",
    "fashion_accessories": "Apparel",
    "electronics": "Electronics",
    "mobile_phone": "Electronics",
    "car_repair": "Automotive",
    "bicycle": "Automotive",
    "car": "Automotive",
    "hardware": "Hardware",
    "paint": "House needs",
    "dry_cleaning": "House needs",
    "houseware": "House needs",
    "furniture": "House needs",
    "stationary": "House needs"
}

def generalize_shop_category(shop_type):
    return category_mapping.get(shop_type, "Other")

def apply_generalization(shop_data):
    for shop in shop_data:
        specific_category = shop.get('type')
        broad_category = generalize_shop_category(specific_category)
        shop['generalized_category'] = broad_category
    return shop_data

def filter_out_other(shop_data):
    return [shop for shop in shop_data if shop['generalized_category'] != 'Other']

def count_shops_by_category(shop_data):
    categories = [shop['generalized_category'] for shop in shop_data]
    return Counter(categories)

def calculate_optimal_k(coordinates, max_k=10):
    def kmeans_inertia(k):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(coordinates)
        return kmeans.inertia_
    
    with ThreadPoolExecutor() as executor:
        wcss = list(executor.map(kmeans_inertia, range(1, max_k + 1)))
    
    optimal_k = np.argmax(np.diff(wcss)) + 2
    print(f"Optimal K: {optimal_k}")
    return optimal_k

def calculate_bounding_box(df):
    south = df['lat'].min()
    north = df['lat'].max()
    west = df['lon'].min()
    east = df['lon'].max()
    return south, north, west, east

def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    a = np.clip(a, 0.0, 1.0)
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371
    distance = R * c
    return float(distance) if np.isfinite(distance) else 0.0

def estimate_population(south, west, north, east):
    width = haversine_distance(south, west, south, east)
    height = haversine_distance(south, west, north, west)
    
    if not np.isfinite(width) or not np.isfinite(height) or width <= 0 or height <= 0:
        return 1000
    
    area = width * height
    if not np.isfinite(area) or area <= 0:
        return 1000
    
    estimated_population = max(int(area * 16), 100)
    return estimated_population

def prepare_neighborhoods(city):
    try:
        # Try different osmnx methods based on version
        tags = {
            'boundary': ['administrative'],
            'admin_level': ['10'],
            'place': ['neighbourhood', 'suburb']
        }
        
        # Try newer API first
        try:
            neighborhoods = ox.features_from_place(city, tags=tags)
        except AttributeError:
            # Fall back to older API
            try:
                neighborhoods = ox.geometries_from_place(city, tags=tags)
            except AttributeError:
                print("OSMnx version doesn't support geometries/features_from_place")
                return gpd.GeoDataFrame()
        
        if neighborhoods.empty:
            print("No neighborhoods found")
            return gpd.GeoDataFrame()
        
        neighborhoods = gpd.GeoDataFrame(neighborhoods, crs="EPSG:4326")
        
        # Ensure we have a geometry column
        if 'geometry' not in neighborhoods.columns:
            print("No geometry column found")
            return gpd.GeoDataFrame()
        
        neighborhoods = neighborhoods.to_crs("EPSG:32618")
        
        point_mask = neighborhoods.geometry.type == 'Point'
        if point_mask.any():
            neighborhoods.loc[point_mask, 'geometry'] = neighborhoods[point_mask].geometry.buffer(500)
        
        def get_area_name(row):
            for col in ['name', 'name:en', 'addr:city', 'place_name']:
                if col in row.index and pd.notna(row[col]):
                    return row[col]
            return None
        
        neighborhoods['area_name'] = neighborhoods.apply(get_area_name, axis=1)
        neighborhoods = neighborhoods.dropna(subset=['area_name', 'geometry'])
        
        print(f"Prepared {len(neighborhoods)} neighborhoods")
        return neighborhoods
        
    except Exception as e:
        print(f"Error preparing neighborhoods: {e}")
        return gpd.GeoDataFrame()

def find_neighborhoods_for_cluster(df, cluster_number, neighborhoods, max_neighborhoods=4):
    cluster_points = df[df['cluster'] == cluster_number]
    
    if cluster_points.empty:
        return []
    
    # Check if neighborhoods is empty or invalid
    if neighborhoods.empty or 'geometry' not in neighborhoods.columns:
        print(f"No valid neighborhoods data available for cluster {cluster_number}")
        return []
    
    try:
        gdf = gpd.GeoDataFrame(
            cluster_points,
            geometry=[Point(xy) for xy in zip(cluster_points.lon, cluster_points.lat)],
            crs="EPSG:4326"
        )
        
        gdf = gdf.to_crs(neighborhoods.crs)
        gdf['geometry'] = gdf.geometry.buffer(100)
        cluster_boundary = unary_union(gdf.geometry)
        
        intersecting = neighborhoods[neighborhoods.geometry.intersects(cluster_boundary)]
        
        if not intersecting.empty:
            intersecting['intersection_area'] = intersecting.geometry.intersection(cluster_boundary).area
            intersecting = intersecting.sort_values('intersection_area', ascending=False)
            unique_neighborhoods = intersecting['area_name'].head(max_neighborhoods).unique()
            return list(unique_neighborhoods)
        else:
            return []
            
    except Exception as e:
        print(f"Error in spatial analysis for cluster {cluster_number}: {e}")
        return []

def perform_analysis_business(new_york_df, k_optimal, category_to_clusters, neighborhoods):
    cluster_bounding_boxes = new_york_df.groupby('cluster').apply(
        lambda x: pd.Series(calculate_bounding_box(x), index=['south', 'north', 'west', 'east'])
    ).reset_index()
    
    # Calculate cluster centroids for map display
    cluster_centroids = new_york_df.groupby('cluster').agg({
        'lat': 'mean',
        'lon': 'mean'
    }).reset_index()
    
    populations = []
    for _, row in cluster_bounding_boxes.iterrows():
        pop = estimate_population(row['south'], row['west'], row['north'], row['east'])
        populations.append(pop)
    
    cluster_bounding_boxes['population'] = populations
    cluster_bounding_boxes_sorted = cluster_bounding_boxes.sort_values(by='population', ascending=False).reset_index(drop=True)
    cluster_bounding_boxes_sorted['score'] = [i * 10 for i in range(k_optimal, 0, -1)]
    
    score_dict = cluster_bounding_boxes_sorted.set_index('cluster')['score'].to_dict()
    sorted_score_dict = dict(sorted(score_dict.items()))
    
    category_counts = {category: len(clusters) for category, clusters in category_to_clusters.items()}
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    top_5_categories = [category for category, _ in sorted_categories[:5]]
    
    result = []
    for category in top_5_categories:
        clusters = category_to_clusters[category]
        max_score_cluster
