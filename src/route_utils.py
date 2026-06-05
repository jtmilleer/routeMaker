import os
import math
import time
import pickle
import requests
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial import KDTree
from paths import DATA_DIR
import srtm

MODEL_FILE = os.path.join(DATA_DIR, "route_model.pkl")
GRAPH_FILE = os.path.join(DATA_DIR, "iowa_city_network.graphml")
SRTM_CACHE = os.path.join(DATA_DIR, "srtm_cache")

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

#def add_elevation_to_graph(G):
#    """Fetch elevation using USGS EPQS with parallel requests and retry logic."""
#    node_ids = list(G.nodes)
#    total    = len(node_ids)
#    print(f"Fetching elevation for {total} nodes via USGS EPQS...")
#    start_time = time.time()
#
#    session = requests.Session()
#    adapter = requests.adapters.HTTPAdapter(pool_connections=40, pool_maxsize=40)
#    session.mount("https://", adapter)
#    session.mount("http://", adapter)
#
#    def fetch_elevation(node_id):
#        lat = float(G.nodes[node_id]["y"])
#        lng = float(G.nodes[node_id]["x"])
#        for attempt in range(3):
#            try:
#                resp = session.get(
#                    "https://epqs.nationalmap.gov/v1/json",
#                    params={"x": lng, "y": lat, "units": "Meters", "includeDate": False},
#                    timeout=15,
#                    proxies={"http": None, "https": None}
#                )
#                val = resp.json().get("value")
#                return node_id, float(val) if val else None
#            except Exception:
#                time.sleep(0.5 * attempt)
#        return node_id, None
#
#    completed = 0
#    with ThreadPoolExecutor(max_workers=40) as executor:
#        futures = {executor.submit(fetch_elevation, n): n for n in node_ids}
#        for future in as_completed(futures):
#            node_id, elev = future.result()
#            G.nodes[node_id]["elevation"] = elev
#            completed += 1
#            if completed % 1000 == 0:
#                print(f"  {completed}/{total} nodes done...")
#
#    missing = 0
#    for n in G.nodes:
#        if G.nodes[n].get("elevation") is None:
#            G.nodes[n]["elevation"] = 0.0
#            missing += 1
#
#    elapsed_time = time.time() - start_time
#    print(f"Done! Fetched {total} nodes in {elapsed_time:.2f} seconds. (Missing: {missing})")
#    return G

def add_elevation_to_graph(G):
    """Fetch elevation using local SRTM tiles for high-speed, offline access."""
    node_ids = list(G.nodes)
    total    = len(node_ids)
    print(f"Loading local elevation data for {total} nodes...")
    start_time = time.time()

    # Create the cache directory if it doesn't exist
    os.makedirs(SRTM_CACHE, exist_ok=True)
    
    # Initialize srtm.py (it will download tiles as needed)
    elevation_data = srtm.get_data(local_cache_dir=SRTM_CACHE)

    for i, node_id in enumerate(node_ids):
        lat = float(G.nodes[node_id]["y"])
        lng = float(G.nodes[node_id]["x"])
        
        # This is a local lookup which is extremely fast
        elev = elevation_data.get_elevation(lat, lng)
        G.nodes[node_id]["elevation"] = float(elev) if elev is not None else 0.0
        
        if (i + 1) % 5000 == 0:
            rate = (i + 1) / (time.time() - start_time)
            print(f"  {i+1}/{total} ({rate:.0f} nodes/s)...")

    missing = sum(1 for n in G.nodes if G.nodes[n].get("elevation", 0.0) == 0.0)
    print(f"Done! {total} nodes in {time.time() - start_time:.2f}s. (Missing Elevation: {missing})")
    return G
    

# ── Road network ──────────────────────────────────────────────────────────────

def mark_hilly_edges(G, hilly_factor):
    print("Adjusting routing weights to favor hilly roads...")
    for u, v, k, d in G.edges(keys=True, data=True):
        grade_abs = d.get("grade_abs", 0.0)
        grade_abs = min(grade_abs, 0.15)
        d["routing_weight"] = d["length"] / (1 + grade_abs * hilly_factor)
    return G

def get_network(start_lat, start_lng, network_dist, hilly_factor):
    if os.path.exists(GRAPH_FILE):
        print("Loading cached road network...")
        G = ox.load_graphml(GRAPH_FILE)
        print(f"Network loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")
        return mark_hilly_edges(G, hilly_factor)

    import threading
    import sys

    stop_event = threading.Event()
    def spinner():
        chars = ['|', '/', '-', '\\']
        idx = 0
        start_t = time.time()
        while not stop_event.is_set():
            elapsed = int(time.time() - start_t)
            sys.stdout.write(f"\rDownloading Iowa City road network... {chars[idx]} ({elapsed}s elapsed)")
            sys.stdout.flush()
            idx = (idx + 1) % len(chars)
            time.sleep(0.1)

    t = threading.Thread(target=spinner)
    t.start()

    try:
        G = ox.graph_from_point(
            (start_lat, start_lng),
            dist=network_dist,
            network_type="bike",
            simplify=True,
        )
    finally:
        stop_event.set()
        t.join()
        sys.stdout.write("\rDownloading Iowa City road network... Done!                  \n")

    print("\nProcessing geometry...")
    print(" [1/7] Removing highways...")
    edges_to_remove = [
        (u, v, k) for u, v, k, data in G.edges(keys=True, data=True)
        if data.get("highway") in ("motorway", "trunk", "primary", "motorway_link", "trunk_link")
    ]
    G.remove_edges_from(edges_to_remove)

    print(" [2/7] Adding edge speeds...")
    G = ox.add_edge_speeds(G)

    print(" [3/7] Adding edge travel times...")
    G = ox.add_edge_travel_times(G)

    print(" [4/7] Fetching elevations...")
    G = add_elevation_to_graph(G)

    print(" [5/7] Adding edge grades...")
    G = ox.add_edge_grades(G)

    print(" [6/7] Filtering to largest strongly connected component...")
    largest_scc = max(nx.strongly_connected_components(G), key=len)
    G.remove_nodes_from(set(G.nodes) - largest_scc)

    print(f" [7/7] Caching network to disk...")
    ox.save_graphml(G, GRAPH_FILE)
    
    print(f"Network processing finished: {len(G.nodes)} nodes, {len(G.edges)} edges")
    return mark_hilly_edges(G, hilly_factor)

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

def get_start_node(G, start_lat, start_lng):
    return ox.distance.nearest_nodes(G, start_lng, start_lat)

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

def encode_path_to_polyline(G, path):
    import polyline as pl
    full_coords = path_to_coords(G, path)
    latlngs = [(lat, lng) for lat, lng, _ in full_coords]
    return pl.encode(latlngs)
