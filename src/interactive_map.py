"""
CSE 557 — Interactive Map Visualizer
Plots the depot, customers, and routes on a real interactive OpenStreetMap
using the folium library. Allows toggling between points and routes.

Usage:
    python interactive_map.py <instance.json> [solution.json]
"""
import json
import argparse
import os
import requests

try:
    import folium
except ImportError:
    print("Please install folium first: python -m pip install folium")
    exit(1)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_color(index):
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
              'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 
              'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 
              'gray', 'black', 'lightgray']
    return colors[index % len(colors)]

def get_route_geometry(coords):
    """Fetches real road geometry from OSRM for a sequence of (lat, lon) coordinates."""
    # OSRM expects lon,lat format
    coord_str = ";".join([f"{lon},{lat}" for lat, lon in coords])
    url = f"http://router.project-osrm.org/route/v1/driving/{coord_str}?overview=full&geometries=geojson"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("code") == "Ok":
                # geojson is [lon, lat], folium needs [lat, lon]
                geom = data["routes"][0]["geometry"]["coordinates"]
                return [[lat, lon] for lon, lat in geom]
    except Exception as e:
        print(f"Warning: Failed to fetch geometry from OSRM: {e}")
    # Fallback to straight lines if API fails
    return coords

def main():
    parser = argparse.ArgumentParser(description="Interactive S-CVRPTW Map Visualizer")
    parser.add_argument("instance", help="Path to instance JSON file")
    parser.add_argument("solution", nargs="?", default=None, help="Path to optional solution JSON file")
    args = parser.parse_args()

    # Load data
    instance = load_json(args.instance)
    solution = load_json(args.solution) if args.solution else None

    # Center map on the Depot
    depot = instance["depot"]
    m = folium.Map(location=[depot["lat"], depot["lon"]], zoom_start=11)

    # --- Feature Group 1: Points (Customers + Depot) ---
    fg_points = folium.FeatureGroup(name="Points (Depot & Customers)")
    
    # Add Depot
    folium.Marker(
        location=[depot["lat"], depot["lon"]],
        popup="DEPOT",
        icon=folium.Icon(color="darkred", icon="home", prefix='fa')
    ).add_to(fg_points)

    # Add Customers with visible ID numbers
    for c in instance["customers"]:
        demand = c.get("demand", "?")
        district = c.get("district", "Unknown")
        tw = f"{c.get('tw_open','?')}-{c.get('tw_close','?')}"
        popup_text = f"ID: {c['id']}<br>District: {district}<br>Demand: {demand}<br>TW: {tw}"
        
        # Numbered icon so the customer index is always visible on the map
        icon_html = f'''<div style="
            background-color: #3388ff;
            color: white;
            border-radius: 50%;
            width: 22px; height: 22px;
            display: flex; align-items: center; justify-content: center;
            font-size: 9px; font-weight: bold;
            border: 2px solid white;
            box-shadow: 0 0 3px rgba(0,0,0,0.4);
        ">{c["id"]}</div>'''
        
        folium.Marker(
            location=[c["lat"], c["lon"]],
            popup=popup_text,
            icon=folium.DivIcon(
                html=icon_html,
                icon_size=(22, 22),
                icon_anchor=(11, 11)
            )
        ).add_to(fg_points)

    # Add Points group to map
    fg_points.add_to(m)

    # --- Feature Group 2: Routes ---
    if solution and "routes" in solution:
        fg_routes = folium.FeatureGroup(name="Routes (Solutions)")
        
        customers_dict = {c["id"]: c for c in instance["customers"]}
        routes = solution["routes"]
        
        for ri, route in enumerate(routes):
            if not route: continue
            
            # Extract coordinates for the path
            path_coords = [(depot["lat"], depot["lon"])]
            for cid in route:
                if cid in customers_dict:
                    c = customers_dict[cid]
                    path_coords.append((c["lat"], c["lon"]))
            path_coords.append((depot["lat"], depot["lon"]))
            
            # Fetch real turn-by-turn road geometry
            print(f"Fetching real geometry for Vehicle {ri+1}...")
            real_path = get_route_geometry(path_coords)
            
            # Determine color for this vehicle
            color = get_color(ri)
            
            # Draw line
            folium.PolyLine(
                locations=real_path,
                color=color,
                weight=4,
                opacity=0.8,
                tooltip=f"Vehicle {ri+1}"
            ).add_to(fg_routes)
            
        fg_routes.add_to(m)

    # Add Layer Control to allow toggling Points vs Routes
    folium.LayerControl().add_to(m)

    # Save to HTML file
    out_filename = f"interactive_map_{instance['name']}.html"
    m.save(out_filename)
    print(f"Success! Open '{out_filename}' in your web browser to view the interactive map.")

if __name__ == "__main__":
    main()
