import os
import random
import folium
import webbrowser
import math
import time
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
import gpxpy
import gpxpy.gpx
from paths import DATA_DIR, ROUTES_DIR
from route_utils import (
    load_model, get_network, build_intersection_index, nearest_intersection,
    meters_to_miles, get_start_node, make_heuristic, remove_spurs,
    has_detours, estimate_elevation_gain, path_to_coords,
    encode_path_to_polyline, predict_score
)

# ── Config ────────────────────────────────────────────────────────────────────

START_LAT    = 41.6543043857067
START_LNG    = -91.52670199266414
NETWORK_DIST = 50000
NUM_ROUTES   = 50
TARGET_MILES = 35
TOLERANCE    = 0.25
HILLY_FACTOR = 100
OUTPUT_DIR   = ROUTES_DIR

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
                            orig = G[u][v][key]["routing_weight"]
                            G[u][v][key]["routing_weight"] *= 10
                            modified.append((u, v, key, orig))

            def restore():
                for u, v, key, orig in modified:
                    if G.has_edge(u, v) and key in G[u][v]:
                        G[u][v][key]["routing_weight"] = orig

            wp1 = get_wp_node(outbound_dist_m, outbound_angle - offset)
            wp2 = get_wp_node(outbound_dist_m, outbound_angle + offset)

            # Segment 1: city → wp1
            seg1 = nx.astar_path(G, start_node, wp1, heuristic=heuristic, weight="routing_weight")
            seg1 = remove_spurs(seg1)
            penalize(seg1)

            # Segment 2: wp1 → wp2
            seg2 = nx.astar_path(G, wp1, wp2, heuristic=heuristic, weight="routing_weight")
            seg2 = remove_spurs(seg2)
            penalize(seg2)

            # Segment 3: wp2 → city
            seg3 = nx.astar_path(G, wp2, start_node, heuristic=heuristic, weight="routing_weight")
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
                    G[u][v][key]["routing_weight"] = orig
            continue

    return None, None

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
            "name":            f"Generated Hilly Route #{i+1}",
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

# ── Map visualization ─────────────────────────────────────────────────────────

def show_routes_in_browser(G, results, top_n=5):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    m = folium.Map(
        location=[START_LAT, START_LNG],
        zoom_start=11,
        # tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        attr="OpenStreetMap contributors"
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

    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: white; padding: 12px 16px; border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2); font-family: sans-serif; font-size: 13px;">
      <b>Top Hilly Routes</b><br>
    """
    for i, r in enumerate(results[:top_n]):
        legend_html += f'<span style="color:{colors[i]}">&#9644;</span> #{i+1} &nbsp;{r["score"]}/10 &nbsp;{r["dist_miles"]}mi &nbsp;{r["elev_ft"]}ft<br>'
    legend_html += "</div>"

    m.get_root().html.add_child(folium.Element(legend_html))

    map_file = os.path.join(OUTPUT_DIR, "top_hilly_routes.html")
    m.save(map_file)
    print(f"\\nOpening map: {map_file}")
    webbrowser.open(f"file:///{os.path.abspath(map_file)}")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bundle     = load_model()
    G          = get_network(START_LAT, START_LNG, NETWORK_DIST, HILLY_FACTOR)
    start_node = get_start_node(G, START_LAT, START_LNG)
    target_m   = TARGET_MILES / 0.000621371

    print("Building intersection index...")
    int_ids, int_coords, int_tree = build_intersection_index(G)
    print(f"  {len(int_ids)} intersection nodes indexed")

    heuristic = make_heuristic(G)

    print(f"\\nGenerating candidate hilly routes (~{TARGET_MILES}mi ± {int(TOLERANCE*100)}%)...")

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

    # Sort to prioritize elevation gain, but keep score somewhat relevant
    results.sort(key=lambda x: -(x["elev_ft"]/50 + x["score"]))

    print(f"\\nTop 5 hilly routes out of {len(results)} generated:\\n")
    for i, r in enumerate(results[:5]):
        print(f"  #{i+1}  Score: {r['score']}/10  |  {r['dist_miles']}mi  |  {r['elev_ft']}ft gain")

    print("\\nSaving top 5 as GPX files...")
    for i, r in enumerate(results[:5]):
        coords = path_to_coords(G, r["path"])
        name   = f"hilly_route_{i+1}_{r['dist_miles']}mi_{r['elev_ft']}ft"
        saved  = save_gpx(coords, f"{name}.gpx", name)
        print(f"  Saved: {saved}")

    show_routes_in_browser(G, results, top_n=5)
    save_route_features(results, G)
