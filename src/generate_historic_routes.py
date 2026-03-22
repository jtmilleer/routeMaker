import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
import gpxpy
import gpxpy.gpx
import pickle
import random
import folium
import webbrowser
import os
import math
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial import KDTree
from paths import DATA_DIR, ROUTES_DIR

# ── Config ────────────────────────────────────────────────────────────────────

START_LAT    = 41.6543043857067
START_LNG    = -91.52670199266414
NETWORK_DIST = 50000
NUM_ROUTES   = 50
TARGET_MILES = 35
TOLERANCE    = 0.25
HILLY_FACTOR = 0  # Keep hilly factor to get hilly historic loops!
DOWNTOWN_RADIUS = 1000
OUTPUT_DIR   = ROUTES_DIR
MODEL_FILE   = os.path.join(DATA_DIR, "route_model.pkl")
GRAPH_FILE   = os.path.join(DATA_DIR, "iowa_city_network.graphml")
HISTORIC_FILE = os.path.join(DATA_DIR, "iowa_city_historic_sites.csv")

# ── Load model ────────────────────────────────────────────────────────────────

def load_model():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)

def predict_score(bundle, route_features: dict) -> float:
    model         = bundle["model"]
    scaler        = bundle["scaler"]
    feature_names = bundle["feature_names"]

    df = pd.DataFrame([route_features])
    features = pd.DataFrame()
    features["distance_mi"]     = df["distance_mi"]
    features["elevation_ft"]    = df.get("elevation_ft",    pd.Series([0])).fillna(0)
    features["avg_speed_mph"]   = df.get("avg_speed_mph",   pd.Series([0])).fillna(0)
    features["moving_time_min"] = df.get("moving_time_min", pd.Series([0])).fillna(0)
    features["elev_per_mile"]   = features["elevation_ft"] / features["distance_mi"].replace(0, 1)
    features["distance_sq"]     = features["distance_mi"] ** 2
    features["elevation_sq"]    = features["elevation_ft"] ** 2
    features["suffer_score"]    = df.get("suffer_score",    pd.Series([0])).fillna(0)
    features["avg_watts"]       = df.get("avg_watts",       pd.Series([0])).fillna(0)
    features["pr_count"]        = df.get("pr_count",        pd.Series([0])).fillna(0)

    features  = features[feature_names]
    X_scaled  = scaler.transform(features)
    score     = model.predict(X_scaled)[0]
    return round(float(np.clip(score, 1, 10)), 2)

# ── Elevation ─────────────────────────────────────────────────────────────────

def add_elevation_to_graph(G):
    """Fetch elevation using USGS EPQS with parallel requests and retry logic."""
    node_ids = list(G.nodes)
    total    = len(node_ids)
    print(f"Fetching elevation for {total} nodes via USGS EPQS...")

    def fetch_elevation(node_id):
        lat = float(G.nodes[node_id]["y"])
        lng = float(G.nodes[node_id]["x"])
        for attempt in range(3):
            try:
                resp = requests.get(
                    "https://epqs.nationalmap.gov/v1/json",
                    params={"x": lng, "y": lat, "units": "Meters", "includeDate": False},
                    timeout=15,
                    proxies={"http": None, "https": None}
                )
                val = resp.json().get("value")
                return node_id, float(val) if val else None
            except Exception:
                time.sleep(0.5 * attempt)
        return node_id, None

    completed = 0
    with ThreadPoolExecutor(max_workers=40) as executor:
        futures = {executor.submit(fetch_elevation, n): n for n in node_ids}
        for future in as_completed(futures):
            node_id, elev = future.result()
            G.nodes[node_id]["elevation"] = elev
            completed += 1
            if completed % 1000 == 0:
                print(f"  {completed}/{total} nodes done...")

    missing = sum(1 for n in G.nodes if G.nodes[n].get("elevation") is None)
    print(f"Done. Missing: {missing}/{total}")
    return G

# ── Road network ──────────────────────────────────────────────────────────────

def mark_hilly_edges(G):
    print("Adjusting routing weights to favor hilly roads...")
    for u, v, k, d in G.edges(keys=True, data=True):
        grade_abs = d.get("grade_abs", 0.0)
        grade_abs = min(grade_abs, 0.15)
        d["routing_weight"] = d["length"] / (1 + grade_abs * HILLY_FACTOR)
    return G

def get_network():
    if os.path.exists(GRAPH_FILE):
        print("Loading cached road network...")
        G = ox.load_graphml(GRAPH_FILE)
        # Fix disjoint components from cached graphs
        if not nx.is_strongly_connected(G):
            print("Filtering to largest strongly connected component...")
            largest_scc = max(nx.strongly_connected_components(G), key=len)
            G = G.subgraph(largest_scc).copy()
        print(f"Network loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")
        return mark_hilly_edges(G)

    print("Downloading Iowa City road network...")
    G = ox.graph_from_point(
        (START_LAT, START_LNG),
        dist=NETWORK_DIST,
        network_type="bike",
        simplify=True,
    )

    edges_to_remove = [
        (u, v, k) for u, v, k, data in G.edges(keys=True, data=True)
        if data.get("highway") in ("motorway", "trunk", "primary", "motorway_link", "trunk_link")
    ]
    G.remove_edges_from(edges_to_remove)

    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    G = add_elevation_to_graph(G)

    for node_id in G.nodes:
        if G.nodes[node_id].get("elevation") is None:
            G.nodes[node_id]["elevation"] = 0.0

    G = ox.add_edge_grades(G)

    ox.save_graphml(G, GRAPH_FILE)
    print(f"Network cached to {GRAPH_FILE}")
    print(f"Network loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")
    return mark_hilly_edges(G)

# ── Spatial index ─────────────────────────────────────────────────────────────

def build_intersection_index(G):
    G_undir = G.to_undirected()
    nodes   = [(n, float(G.nodes[n]["y"]), float(G.nodes[n]["x"]))
               for n in G.nodes if G_undir.degree(n) >= 3]
    ids     = [n[0] for n in nodes]
    coords  = np.array([(n[1], n[2]) for n in nodes])
    tree    = KDTree(coords)
    return ids, coords, tree

def nearest_intersection(ids, coords, tree, lat, lng):
    _, idx = tree.query([lat, lng])
    return ids[idx]

# ── Route helpers ─────────────────────────────────────────────────────────────

def meters_to_miles(m):
    return m * 0.000621371

def get_start_node(G):
    return ox.distance.nearest_nodes(G, START_LNG, START_LAT)

def make_heuristic(G):
    coords = {n: (float(d["y"]), float(d["x"])) for n, d in G.nodes(data=True)}
    def heuristic(u, v):
        u_lat, u_lng = coords[u]
        v_lat, v_lng = coords[v]
        return math.sqrt((u_lat - v_lat)**2 + ((u_lng - v_lng)*math.cos(math.radians(u_lat)))**2) * 111320
    return heuristic

def remove_spurs(path):
    if len(path) < 3:
        return path

    stack = [path[0]]
    for node in path[1:]:
        if len(stack) >= 2 and stack[-2] == node:
            stack.pop()
        else:
            stack.append(node)
    return stack

def has_detours(G, path, max_detour_ratio=2.5):
    window = 10
    for i in range(0, len(path) - window, window // 2):
        segment = path[i:i + window]
        start   = G.nodes[segment[0]]
        end     = G.nodes[segment[-1]]
        lat1, lng1 = math.radians(float(start["y"])), math.radians(float(start["x"]))
        lat2, lng2 = math.radians(float(end["y"])),   math.radians(float(end["x"]))
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
        straight = 3958.8 * 2 * math.asin(math.sqrt(max(0, a)))

        if straight < 0.1:
            continue

        actual = sum(
            G.get_edge_data(u, v, 0).get("length", 0)
            for u, v in zip(segment[:-1], segment[1:])
        ) * 0.000621371

        if actual / straight > max_detour_ratio:
            return True
    return False

def estimate_elevation_gain(G, path):
    elevations = []
    for node in path:
        elev = G.nodes[node].get("elevation")
        try:
            elev = float(elev)
            elevations.append(elev if elev > 0 else None)
        except (TypeError, ValueError):
            elevations.append(None)

    gain = 0
    prev_elev = None
    for elev in elevations:
        if elev is None:
            prev_elev = None
            continue
        if prev_elev is not None:
            diff = elev - prev_elev
            if diff > 1.0:
                gain += diff
        prev_elev = elev
    return gain * 3.28084

# ── Route generation ──────────────────────────────────────────────────────────

def generate_loop(G, start_node, target_meters, int_ids, int_coords, int_tree, heuristic, valid_historic_nodes):
    start_lat = float(G.nodes[start_node]["y"])
    start_lng = float(G.nodes[start_node]["x"])

    for _ in range(100):
        modified = []
        try:
            hist_node, hist_name = random.choice(valid_historic_nodes)
            wp1 = hist_node
            wp1_lat = float(G.nodes[wp1]["y"])
            wp1_lng = float(G.nodes[wp1]["x"])
            
            d1 = ox.distance.great_circle(start_lat, start_lng, wp1_lat, wp1_lng)
            
            def get_wp2(distance, angle):
                rad  = math.radians(angle)
                dlat = (distance / 111320) * math.cos(rad)
                dlng = (distance / (111320 * math.cos(math.radians(wp1_lat)))) * math.sin(rad)
                return nearest_intersection(int_ids, int_coords, int_tree, wp1_lat + dlat, wp1_lng + dlng)

            hist_name2 = None
            hist_coords2 = None
            if d1 < DOWNTOWN_RADIUS:
                far_nodes = [(n, name) for n, name in valid_historic_nodes 
                             if ox.distance.great_circle(start_lat, start_lng, float(G.nodes[n]['y']), float(G.nodes[n]['x'])) > DOWNTOWN_RADIUS]
                if far_nodes:
                    wp2_node, hist_name2 = random.choice(far_nodes)
                    wp2 = wp2_node
                    wp2_lat = float(G.nodes[wp2]["y"])
                    wp2_lng = float(G.nodes[wp2]["x"])
                    hist_coords2 = (wp2_lat, wp2_lng)
                else:
                    rem_dist = max(5000, target_meters - d1)
                    wp2 = get_wp2(rem_dist / 2, random.uniform(0, 360))
            else:
                rem_dist = max(5000, target_meters - d1)
                wp2 = get_wp2(rem_dist / 2, random.uniform(0, 360))

            # Segment 1: city → wp1 (historic site)
            seg1 = nx.astar_path(G, start_node, wp1, heuristic=heuristic, weight="routing_weight")
            seg1 = remove_spurs(seg1)

            # Segment 2: wp1 → wp2
            seg2 = nx.astar_path(G, wp1, wp2, heuristic=heuristic, weight="routing_weight")
            seg2 = remove_spurs(seg2)

            # Segment 3: wp2 → city
            seg3 = nx.astar_path(G, wp2, start_node, heuristic=heuristic, weight="routing_weight")
            seg3 = remove_spurs(seg3)

            full_path = seg1 + seg2[1:] + seg3[1:]
            full_path = remove_spurs(full_path)

            total_dist = sum(
                G.get_edge_data(u, v, 0).get("length", 0)
                for u, v in zip(full_path[:-1], full_path[1:])
            )

            dist_miles = meters_to_miles(total_dist)
            low  = TARGET_MILES * (1 - TOLERANCE)
            high = TARGET_MILES * (1 + TOLERANCE)

            if not (low <= dist_miles <= high):
                continue

            # Reject high edge reuse
            all_edges    = list(zip(full_path[:-1], full_path[1:]))
            unique_edges = len(set(all_edges))
            reuse_ratio  = 1 - (unique_edges / len(all_edges)) if all_edges else 1
            if reuse_ratio > 0.15:
                continue

            if has_detours(G, full_path):
                continue

            return full_path, dist_miles, hist_name, (wp1_lat, wp1_lng), hist_name2, hist_coords2

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    return None, None, None, None, None, None

# ── Coords + GPX ─────────────────────────────────────────────────────────────

def path_to_coords(G, path):
    coords = []
    for u, v in zip(path[:-1], path[1:]):
        edge_data = G.get_edge_data(u, v)
        if edge_data is None:
            edge_data = G.get_edge_data(v, u)
        if edge_data is None:
            continue

        data = edge_data.get(0, list(edge_data.values())[0])

        if "geometry" in data:
            geom_coords = list(data["geometry"].coords)
            if u in G.nodes:
                u_lng    = float(G.nodes[u]["x"])
                geom_lng = geom_coords[0][0]
                if abs(geom_lng - u_lng) > 0.0001:
                    geom_coords = list(reversed(geom_coords))
            u_elev = G.nodes[u].get("elevation")
            v_elev = G.nodes[v].get("elevation")
            try:
                u_elev = float(u_elev) if u_elev else None
                v_elev = float(v_elev) if v_elev else None
            except (TypeError, ValueError):
                u_elev = v_elev = None
            elev = (u_elev + v_elev) / 2 if u_elev and v_elev else (u_elev or v_elev)
            for lng, lat in geom_coords:
                coords.append((lat, lng, elev))
        else:
            for node in [u, v]:
                data_n = G.nodes[node]
                elev   = data_n.get("elevation")
                try:
                    elev = float(elev) if elev is not None else None
                except (TypeError, ValueError):
                    elev = None
                coords.append((float(data_n["y"]), float(data_n["x"]), elev))

    return coords

def save_gpx(coords, historic_info, filename, route_name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack(name=route_name)
    gpx.tracks.append(track)
    segment = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(segment)
    for lat, lng, elev in coords:
        elev_clean = elev if (elev is not None and elev > 0) else None
        segment.points.append(gpxpy.gpx.GPXTrackPoint(lat, lng, elevation=elev_clean))
        
    # Mark historic site via GPX waypoint
    if historic_info:
        for hist_name, hist_lat, hist_lng in historic_info:
            wp = gpxpy.gpx.GPXWaypoint(hist_lat, hist_lng, name=hist_name)
            gpx.waypoints.append(wp)
        
    fpath = f"{OUTPUT_DIR}/{filename}"
    with open(fpath, "w") as f:
        f.write(gpx.to_xml())
    return fpath

def save_route_features(results, G):
    rows = []
    for i, r in enumerate(results[:5]):
        rows.append({
            "id":              f"generated_{i+1}_{int(time.time())}",
            "name":            f"Historic Route #{i+1} ({r['hist_name']})",
            "distance_mi":     r["dist_miles"],
            "elevation_ft":    r["elev_ft"],
            "moving_time_min": round(r["dist_miles"] / 15 * 60),
            "avg_speed_mph":   15,
            "avg_watts":       None,
            "suffer_score":    None,
            "pr_count":        0,
            "score":           r["score"],
            "polyline":        encode_path_to_polyline(G, r["path"]),
        })
    df  = pd.DataFrame(rows)
    out = os.path.join(DATA_DIR, "generated_routes.csv")
    if os.path.exists(out):
        existing = pd.read_csv(out)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"Saved route features to {out}")

def encode_path_to_polyline(G, path):
    import polyline as pl
    full_coords = path_to_coords(G, path)
    latlngs = [(lat, lng) for lat, lng, _ in full_coords]
    return pl.encode(latlngs)

# ── Map visualization ─────────────────────────────────────────────────────────

def show_routes_in_browser(G, results, valid_historic_nodes, top_n=5):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    m = folium.Map(
        location=[START_LAT, START_LNG],
        zoom_start=11,
        tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        attr="© OpenStreetMap contributors © CARTO"
    )

    # Plot all valid historic sites as small dots
    for node, name in valid_historic_nodes:
        lat = float(G.nodes[node]['y'])
        lng = float(G.nodes[node]['x'])
        folium.CircleMarker(
            location=[lat, lng],
            radius=3,
            color='gray',
            fill=True,
            fill_color='gray',
            fill_opacity=0.5,
            tooltip=f"{name}"
        ).add_to(m)

    folium.Marker(
        [START_LAT, START_LNG],
        popup="Start",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)

    # Draw the downtown radius boundary
    folium.Circle(
        location=[START_LAT, START_LNG],
        radius=DOWNTOWN_RADIUS,
        color="red",
        weight=2,
        fill=True,
        fill_opacity=0.05,
        tooltip=f"Downtown Border ({DOWNTOWN_RADIUS}m)"
    ).add_to(m)

    colors = ["green", "blue", "purple", "orange", "darkred"]

    for i, r in enumerate(results[:top_n]):
        coords  = path_to_coords(G, r["path"])
        latLngs = [(lat, lng) for lat, lng, _ in coords]
        label   = f"#{i+1} | Score: {r['score']}/10 | {r['dist_miles']}mi | Visited: {r['hist_name']}"
        folium.PolyLine(
            latLngs,
            color=colors[i % len(colors)],
            weight=4,
            opacity=0.8,
            tooltip=label,
            popup=label,
        ).add_to(m)
        
        # Add Historic Point marker
        hist_name, hist_lat, hist_lng = r['hist_name'], r['hist_lat'], r['hist_lng']
        folium.Marker(
            [hist_lat, hist_lng],
            popup=f"National Register: {hist_name}",
            icon=folium.Icon(color="purple", icon="star")
        ).add_to(m)
        if r.get('hist_name2'):
            folium.Marker(
                [r['hist_lat2'], r['hist_lng2']],
                popup=f"National Register: {r['hist_name2']}",
                icon=folium.Icon(color="purple", icon="star")
            ).add_to(m)

    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: white; padding: 12px 16px; border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2); font-family: sans-serif; font-size: 13px;">
      <b>Top Historic Routes</b><br>
    """
    for i, r in enumerate(results[:top_n]):
        legend_html += f'<span style="color:{colors[i]}">&#9644;</span> #{i+1} &nbsp;{r["score"]}/10 &nbsp;{r["dist_miles"]}mi &nbsp;{r["elev_ft"]}ft<br>'
    legend_html += "</div>"

    m.get_root().html.add_child(folium.Element(legend_html))

    map_file = os.path.join(OUTPUT_DIR, "top_historic_routes.html")
    m.save(map_file)
    print(f"\nOpening map: {map_file}")
    webbrowser.open(f"file:///{os.path.abspath(map_file)}")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bundle     = load_model()
    G          = get_network()
    start_node = get_start_node(G)
    target_m   = TARGET_MILES / 0.000621371

    if not os.path.exists(HISTORIC_FILE):
        print("Historic sites file not found. Run process_historic.py first.")
        exit(1)

    print("Loading historic sites...")
    df_hist = pd.read_csv(HISTORIC_FILE)
    max_h_dist = TARGET_MILES * 1609.34 / 2 * 0.90
    
    valid_historic_nodes = []
    print("Snapping historic sites to road network...")
    for idx, row in df_hist.iterrows():
        lat, lng, name = row['lat'], row['lng'], row['name']
        dist_from_start = ox.distance.great_circle(START_LAT, START_LNG, lat, lng)
        if dist_from_start < max_h_dist:
            node = ox.distance.nearest_nodes(G, lng, lat)
            valid_historic_nodes.append((node, name))

    print(f"  Found {len(valid_historic_nodes)} reachable historic sites.")

    print("Building intersection index...")
    int_ids, int_coords, int_tree = build_intersection_index(G)
    print(f"  {len(int_ids)} intersection nodes indexed")

    heuristic = make_heuristic(G)

    print(f"\nGenerating candidate historic routes (~{TARGET_MILES}mi ± {int(TOLERANCE*100)}%)...")

    results  = []
    attempts = 0

    while len(results) < NUM_ROUTES and attempts < NUM_ROUTES * 50:
        attempts += 1
        path, dist_miles, hist_name, hist_coords, hist_name2, hist_coords2 = generate_loop(
            G, start_node, target_m, int_ids, int_coords, int_tree, heuristic, valid_historic_nodes
        )
        if path is None:
            continue

        elev_ft  = estimate_elevation_gain(G, path)
        time_min = dist_miles / 15 * 60

        features = {
            "distance_mi":     dist_miles,
            "elevation_ft":    elev_ft,
            "avg_speed_mph":   15,
            "moving_time_min": time_min,
            "suffer_score":    0,
            "avg_watts":       0,
            "pr_count":        0,
        }

        score = predict_score(bundle, features)
        results.append({
            "path":       path,
            "score":      score,
            "dist_miles": round(dist_miles, 1),
            "elev_ft":    round(elev_ft),
            "hist_name":  hist_name,
            "hist_lat":   hist_coords[0],
            "hist_lng":   hist_coords[1],
            "hist_name2": hist_name2,
            "hist_lat2":  hist_coords2[0] if hist_coords2 else None,
            "hist_lng2":  hist_coords2[1] if hist_coords2 else None
        })

    # Sort primarily by score
    results.sort(key=lambda x: -x["score"])

    print(f"\nTop 5 historic routes out of {len(results)} generated:\n")
    for i, r in enumerate(results[:5]):
        h_name = r['hist_name'][:30] + "..." if len(r['hist_name']) > 30 else r['hist_name']
        print(f"  #{i+1}  Score: {r['score']}/10  | {r['dist_miles']}mi | Site: {h_name}")

    print("\nSaving top 5 as GPX files...")
    for i, r in enumerate(results[:5]):
        coords = path_to_coords(G, r["path"])
        name   = f"historic_route_{i+1}_{r['dist_miles']}mi"
        # pass historic info logic
        historic_info = [(r['hist_name'], r['hist_lat'], r['hist_lng'])]
        if r.get('hist_name2'):
            historic_info.append((r['hist_name2'], r['hist_lat2'], r['hist_lng2']))
        saved  = save_gpx(coords, historic_info, f"{name}.gpx", name)
        print(f"  Saved: {saved}")

    show_routes_in_browser(G, results, valid_historic_nodes, top_n=5)
    save_route_features(results, G)
