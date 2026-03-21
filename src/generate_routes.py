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
OUTPUT_DIR   = ROUTES_DIR
MODEL_FILE   = os.path.join(DATA_DIR, "route_model.pkl")
GRAPH_FILE   = os.path.join(DATA_DIR, "iowa_city_network.graphml")

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
                    timeout=15
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

def get_network():
    if os.path.exists(GRAPH_FILE):
        print("Loading cached road network...")
        G = ox.load_graphml(GRAPH_FILE)
        print(f"Network loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")
        return G

    print("Downloading Iowa City road network...")
    G = ox.graph_from_point(
        (START_LAT, START_LNG),
        dist=NETWORK_DIST,
        network_type="bike",
        simplify=True,
    )

    # Remove highways and motorways
    edges_to_remove = [
        (u, v, k) for u, v, k, data in G.edges(keys=True, data=True)
        if data.get("highway") in ("motorway", "trunk", "primary", "motorway_link", "trunk_link")
    ]
    G.remove_edges_from(edges_to_remove)
    # No dead end removal — intersection index handles waypoint snapping instead

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
    return G

# ── Spatial index ─────────────────────────────────────────────────────────────

def build_intersection_index(G):
    """Build a KD-tree spatial index of intersection nodes (degree >= 3)."""
    G_undir = G.to_undirected()
    nodes   = [(n, float(G.nodes[n]["y"]), float(G.nodes[n]["x"]))
               for n in G.nodes if G_undir.degree(n) >= 3]
    ids     = [n[0] for n in nodes]
    coords  = np.array([(n[1], n[2]) for n in nodes])
    tree    = KDTree(coords)
    return ids, coords, tree

def nearest_intersection(ids, coords, tree, lat, lng):
    """Find nearest intersection node using KD-tree — O(log n)."""
    _, idx = tree.query([lat, lng])
    return ids[idx]

# ── Route helpers ─────────────────────────────────────────────────────────────

def meters_to_miles(m):
    return m * 0.000621371

def get_start_node(G):
    return ox.distance.nearest_nodes(G, START_LNG, START_LAT)

def make_heuristic(G):
    """Return an A* heuristic function using straight-line geographic distance."""
    coords = {n: (float(d["y"]), float(d["x"])) for n, d in G.nodes(data=True)}
    def heuristic(u, v):
        u_lat, u_lng = coords[u]
        v_lat, v_lng = coords[v]
        return math.sqrt((u_lat - v_lat)**2 + (u_lng - v_lng)**2) * 111320
    return heuristic

def remove_spurs(path):
    """Remove backtracking of any length by finding edges traversed in both directions."""
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
    """Reject routes where any local section deviates far from a straight line."""
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

def route_compactness(G, path):
    start = G.nodes[path[0]]
    end   = G.nodes[path[-1]]
    lat1, lng1 = math.radians(float(start["y"])), math.radians(float(start["x"]))
    lat2, lng2 = math.radians(float(end["y"])),   math.radians(float(end["x"]))
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
    straight_line_miles = 3958.8 * 2 * math.asin(math.sqrt(a))
    total_dist = sum(
        G.get_edge_data(u, v, 0).get("length", 0)
        for u, v in zip(path[:-1], path[1:])
    ) * 0.000621371
    return straight_line_miles / total_dist if total_dist > 0 else 1.0

# ── Route generation ──────────────────────────────────────────────────────────

def generate_loop(G, start_node, target_meters, int_ids, int_coords, int_tree, heuristic):
    """
    Generate a lollipop-style loop: out of the city into the country,
    loop around out there, and back into the city.
    Uses A* for fast routing and modifies/restores edge weights in place.
    """
    start_lat = float(G.nodes[start_node]["y"])
    start_lng = float(G.nodes[start_node]["x"])

    outbound_dist_m = target_meters / 3

    for _ in range(50):
        modified = []
        try:
            outbound_angle = random.uniform(0, 360)
            offset         = random.uniform(15, 90)

            def get_wp_node(distance, angle):
                rad  = math.radians(angle)
                dlat = (distance / 111320) * math.cos(rad)
                dlng = (distance / (111320 * math.cos(math.radians(start_lat)))) * math.sin(rad)
                return nearest_intersection(int_ids, int_coords, int_tree, start_lat + dlat, start_lng + dlng)

            def penalize(seg):
                for u, v in zip(seg[:-1], seg[1:]):
                    if G.has_edge(u, v):
                        for key in G[u][v]:
                            orig = G[u][v][key]["length"]
                            G[u][v][key]["length"] *= 10
                            modified.append((u, v, key, orig))

            def restore():
                for u, v, key, orig in modified:
                    if G.has_edge(u, v) and key in G[u][v]:
                        G[u][v][key]["length"] = orig

            wp1 = get_wp_node(outbound_dist_m, outbound_angle - offset)
            wp2 = get_wp_node(outbound_dist_m, outbound_angle + offset)

            # Segment 1: city → wp1
            seg1 = nx.astar_path(G, start_node, wp1, heuristic=heuristic, weight="length")
            seg1 = remove_spurs(seg1)
            penalize(seg1)

            # Segment 2: wp1 → wp2
            seg2 = nx.astar_path(G, wp1, wp2, heuristic=heuristic, weight="length")
            seg2 = remove_spurs(seg2)
            penalize(seg2)

            # Segment 3: wp2 → city
            seg3 = nx.astar_path(G, wp2, start_node, heuristic=heuristic, weight="length")
            seg3 = remove_spurs(seg3)

            # Restore edge weights before any further checks
            restore()
            modified = []

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
                outbound_dist_m *= (TARGET_MILES / dist_miles) ** 0.5 if dist_miles > 0 else 1
                continue

            # Reject if midpoint is too close to city center
            midpoint_idx = len(full_path) // 2
            mid_node     = full_path[midpoint_idx]
            mid_lat      = float(G.nodes[mid_node]["y"])
            mid_lng      = float(G.nodes[mid_node]["x"])
            dlat = math.radians(mid_lat - start_lat)
            dlng = math.radians(mid_lng - start_lng)
            a    = math.sin(dlat/2)**2 + math.cos(math.radians(start_lat)) * math.cos(math.radians(mid_lat)) * math.sin(dlng/2)**2
            mid_dist_miles = 3958.8 * 2 * math.asin(math.sqrt(a))
            if mid_dist_miles < 3.0:
                continue

            # Reject high edge reuse
            all_edges    = list(zip(full_path[:-1], full_path[1:]))
            unique_edges = len(set(all_edges))
            reuse_ratio  = 1 - (unique_edges / len(all_edges)) if all_edges else 1
            if reuse_ratio > 0.15:
                continue

            # Reject rectangular detours
            if has_detours(G, full_path):
                continue

            return full_path, dist_miles

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            for u, v, key, orig in modified:
                if G.has_edge(u, v) and key in G[u][v]:
                    G[u][v][key]["length"] = orig
            continue

    return None, None

# ── Coords + GPX ─────────────────────────────────────────────────────────────

def path_to_coords(G, path):
    """Convert node path to (lat, lng, elev) using full edge geometry."""
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

def save_gpx(coords, filename, route_name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack(name=route_name)
    gpx.tracks.append(track)
    segment = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(segment)
    for lat, lng, elev in coords:
        elev_clean = elev if (elev is not None and elev > 0) else None
        segment.points.append(gpxpy.gpx.GPXTrackPoint(lat, lng, elevation=elev_clean))
    fpath = f"{OUTPUT_DIR}/{filename}"
    with open(fpath, "w") as f:
        f.write(gpx.to_xml())
    return fpath

def save_route_features(results, G):
    rows = []
    for i, r in enumerate(results[:5]):
        rows.append({
            "id":              f"generated_{i+1}_{int(time.time())}",
            "name":            f"Generated Route #{i+1}",
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

def show_routes_in_browser(G, results, top_n=5):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    m = folium.Map(
        location=[START_LAT, START_LNG],
        zoom_start=11,
        tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        attr="© OpenStreetMap contributors © CARTO"
    )

    folium.Marker(
        [START_LAT, START_LNG],
        popup="Start",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)

    colors = ["green", "blue", "purple", "orange", "darkred"]

    for i, r in enumerate(results[:top_n]):
        coords  = path_to_coords(G, r["path"])
        latLngs = [(lat, lng) for lat, lng, _ in coords]
        label   = f"#{i+1} | Score: {r['score']}/10 | {r['dist_miles']}mi | {r['elev_ft']}ft"
        folium.PolyLine(
            latLngs,
            color=colors[i % len(colors)],
            weight=4,
            opacity=0.8,
            tooltip=label,
            popup=label,
        ).add_to(m)

    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: white; padding: 12px 16px; border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2); font-family: sans-serif; font-size: 13px;">
      <b>Top Routes</b><br>
    """
    for i, r in enumerate(results[:top_n]):
        legend_html += f'<span style="color:{colors[i]}">&#9644;</span> #{i+1} &nbsp;{r["score"]}/10 &nbsp;{r["dist_miles"]}mi &nbsp;{r["elev_ft"]}ft<br>'
    legend_html += "</div>"

    m.get_root().html.add_child(folium.Element(legend_html))

    map_file = os.path.join(OUTPUT_DIR, "top_routes.html")
    m.save(map_file)
    print(f"\nOpening map: {map_file}")
    webbrowser.open(f"file:///{os.path.abspath(map_file)}")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bundle     = load_model()
    G          = get_network()
    start_node = get_start_node(G)
    target_m   = TARGET_MILES / 0.000621371

    print("Building intersection index...")
    int_ids, int_coords, int_tree = build_intersection_index(G)
    print(f"  {len(int_ids)} intersection nodes indexed")

    heuristic = make_heuristic(G)

    print(f"\nGenerating candidate routes (~{TARGET_MILES}mi ± {int(TOLERANCE*100)}%)...")

    results  = []
    attempts = 0

    while len(results) < NUM_ROUTES and attempts < NUM_ROUTES * 10:
        attempts += 1
        path, dist_miles = generate_loop(
            G, start_node, target_m, int_ids, int_coords, int_tree, heuristic
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
        })

    results.sort(key=lambda x: -x["score"])

    print(f"\nTop 5 routes out of {len(results)} generated:\n")
    for i, r in enumerate(results[:5]):
        print(f"  #{i+1}  Score: {r['score']}/10  |  {r['dist_miles']}mi  |  {r['elev_ft']}ft gain")

    print("\nSaving top 5 as GPX files...")
    for i, r in enumerate(results[:5]):
        coords = path_to_coords(G, r["path"])
        name   = f"route_{i+1}_{r['dist_miles']}mi_{r['elev_ft']}ft"
        saved  = save_gpx(coords, f"{name}.gpx", name)
        print(f"  Saved: {saved}")

    show_routes_in_browser(G, results, top_n=5)
    save_route_features(results, G)