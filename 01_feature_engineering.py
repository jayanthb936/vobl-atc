import pandas as pd
import numpy as np
import sqlite3
import math
import os

# VOBL Airport Reference Point
ARP_LAT = 13.1989
ARP_LON = 77.7056

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in Nautical Miles using Haversine"""
    R = 3440.065
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat/2) * math.sin(dLat/2) + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing from point 1 to point 2"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dLon = lon2 - lon1
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360

def load_star_starts(db_path):
    """Extract the first waypoint (starting point) of every STAR route"""
    conn = sqlite3.connect(db_path)
    waypoints = pd.read_sql_query("SELECT * FROM waypoints", conn).set_index('identifier').to_dict('index')
    routes_df = pd.read_sql_query("SELECT * FROM star_routes WHERE sequence_num = 1", conn)
    conn.close()
    
    star_starts = {}
    for _, row in routes_df.iterrows():
        wp = row['waypoint_identifier']
        if wp in waypoints:
            star_starts[row['star_name']] = (waypoints[wp]['latitude'], waypoints[wp]['longitude'])
    return star_starts

def find_nearest_star(lat, lon, star_starts):
    """Find the closest STAR route for a given lat/lon"""
    closest_star = None
    min_dist = float('inf')
    for star_name, coords in star_starts.items():
        dist = calculate_distance(lat, lon, coords[0], coords[1])
        if dist < min_dist:
            min_dist = dist
            closest_star = star_name
    return closest_star

def convert_nm_to_latlon(x_nm, y_nm, ref_lat, ref_lon):
    """Approximate conversion from X,Y (NM) back to Lat,Lon relative to VOBL"""
    # 1 NM of latitude is approx 1/60th of a degree
    lat = ref_lat + (y_nm / 60.0)
    # 1 NM of longitude depends on latitude
    lon = ref_lon + (x_nm / (60.0 * math.cos(math.radians(ref_lat))))
    return lat, lon

def main():
    print("Step 1: Loading raw 3-day dataset...")
    input_file = r"C:\Users\Jayanth B\Downloads\ARRIVAL_WITH_ETA.csv"
    db_file = r"C:\Users\Jayanth B\PycharmProjects\PythonProject\star_routes.db"
    output_dir = r"C:\Users\Jayanth B\PycharmProjects\PythonProject\data"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows.")
    
    # Check if target variable exists
    if 'pms_eta_seconds_remaining' not in df.columns:
        print("ERROR: Target variable 'pms_eta_seconds_remaining' missing!")
        return

    # Clean missing data
    df = df.dropna(subset=['X (Nm)', 'Y (Nm)', 'gspdxy', 'Mode-C', 'pms_eta_seconds_remaining']).copy()

    # Sort sequentially by flight and time for rolling operations
    df = df.sort_values(by=['icao_address', 'abstime_sec'])

    print("Step 2: Reconstructing Spatial Lat/Lon from X,Y Coordinates...")
    df['lat'] = df.apply(lambda row: convert_nm_to_latlon(row['X (Nm)'], row['Y (Nm)'], ARP_LAT, ARP_LON)[0], axis=1)
    df['lon'] = df.apply(lambda row: convert_nm_to_latlon(row['X (Nm)'], row['Y (Nm)'], ARP_LAT, ARP_LON)[1], axis=1)

    print("Step 3: Extracting STAR Routes & One-Hot Encoding...")
    star_starts = load_star_starts(db_file)
    df['assigned_star'] = df.apply(lambda row: find_nearest_star(row['lat'], row['lon'], star_starts), axis=1)
    df = pd.get_dummies(df, columns=['assigned_star'], prefix='STAR')

    print("Step 4: Engineering Time-Based Cyclical Features...")
    # Map seconds in a day (0 to 86400) to sine and cosine waves
    df['time_sin'] = np.sin(2 * np.pi * df['abstime_sec'] / 86400)
    df['time_cos'] = np.cos(2 * np.pi * df['abstime_sec'] / 86400)

    print("Step 5: Kinematic Smoothing (Rolling Averages)...")
    # 5-ping rolling average to smooth radar jitter (grouped by flight)
    df['gspdxy_smooth'] = df.groupby('icao_address')['gspdxy'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['vrate_smooth'] = df.groupby('icao_address')['vrate'].transform(lambda x: x.rolling(5, min_periods=1).mean())

    print("Step 6: Mathematical Geometric Features...")
    # Bearing to VOBL Runway
    df['bearing_to_rwy'] = df.apply(lambda row: calculate_bearing(row['lat'], row['lon'], ARP_LAT, ARP_LON), axis=1)
    
    # Track Error: Difference between where they are pointing (mhdg) and where they should be pointing (runway)
    def calculate_track_error(heading, bearing):
        diff = abs(heading - bearing)
        return min(diff, 360 - diff)
        
    df['track_error'] = df.apply(lambda row: calculate_track_error(row['mhdg'], row['bearing_to_rwy']), axis=1)

    print("Step 7: Total Energy State...")
    # Formula: Potential Energy (Altitude) + Kinetic Energy (Speed)
    # Scaled to be roughly comparable numeric values
    df['energy_state'] = (df['Mode-C'] / 100) + (df['gspdxy_smooth'] ** 2) / 1000

    print("Step 8: Finalizing Feature Dataset...")
    # Select final features
    feature_columns = [
        'icao_address', 'r_callsign', 'abstime_sec', 'lat', 'lon',
        'X (Nm)', 'Y (Nm)', 'gspdxy', 'gspdxy_smooth', 'Mode-C', 'vrate', 'vrate_smooth',
        'distance_nm', 'pms_track_distance', 'bearing_to_rwy', 'track_error', 'energy_state',
        'time_sin', 'time_cos'
    ]
    # Add the One-Hot Encoded STAR columns
    star_cols = [c for c in df.columns if c.startswith('STAR_')]
    feature_columns.extend(star_cols)
    
    # Target
    feature_columns.append('pms_eta_seconds_remaining')
    
    final_df = df[feature_columns]
    
    output_file = os.path.join(output_dir, 'ML_READY_DATA.csv')
    final_df.to_csv(output_file, index=False)
    
    print(f"\nSUCCESS! Feature Engineering Complete.")
    print(f"Saved {len(final_df)} highly enriched rows to: {output_file}")
    print("\nFeatures Generated:")
    for col in final_df.columns:
        print(f" - {col}")

if __name__ == "__main__":
    main()
