import json
import math
import os
import random
import uuid

import gpxpy
import gpxpy.gpx
import httpx
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import polyline as pl

from backend.core.config import settings
from backend.models.schemas import GenerateRequest, HistoricSite, RouteResult
from backend.services import graph_service, model_service

NUM_CANDIDATES = 30
MAX_ATTEMPTS_MULTIPLIER = 10

# Cache: city_key -> list of (node_id, name, lat, lng)
_historic_sites_cache: dict[str, list] = {}


def generate_routes(
    G: nx.MultiDiGraph,
    athlete_id: int,
    params: GenerateRequest,
    rides_df: pd.DataFrame,
) -> list[RouteResult]:
    G = G.copy()

    weight_attr = "length"
    historic_nodes = None

    if params.route_type == "hilly":
        graph_service.mark_hilly_edges(G, params.hilly_factor)
        weight_attr = "routing_weight"
    elif params.route_type == "novel":
        graph_service.mark_ridden_edges(G, rides_df, params.novelty_factor)
        weight_attr = "routing_weight"
    elif params.route_type == "historic":
        for u, v, k, d in G.edges(keys=True, data=True):
            d["routing_weight"] = d["length"]
        weight_attr = "routing_weight"
        historic_nodes = _load_historic_sites(
            G, params.start_lat, params.start_lng, params.distance_mi,
        )
        if not historic_nodes:
            raise ValueError("No historic sites found near this location")

    start_node = ox.distance.nearest_nodes(G, params.start_lng, params.start_lat)
    target_m = params.distance_mi / 0.000621371
    int_ids, int_coords, int_tree = graph_service.build_intersection_index(G)
    heuristic = _make_heuristic(G)

    results = []
    attempts = 0
    max_attempts = NUM_CANDIDATES * MAX_ATTEMPTS_MULTIPLIER

    while len(results) < NUM_CANDIDATES and attempts < max_attempts:
        attempts += 1

        if params.route_type == "historic":
            loop_result = _generate_historic_loop(
                G, start_node, target_m, params.distance_mi, params.tolerance,
                int_ids, int_coords, int_tree, heuristic, weight_attr,
                historic_nodes, params.downtown_radius,
            )
            if loop_result is None:
                continue
            path, dist_miles, sites_visited = loop_result
        else:
            path, dist_miles = _generate_loop(
                G, start_node, target_m, params.distance_mi, params.tolerance,
                int_ids, int_coords, int_tree, heuristic, weight_attr,
            )
            if path is None:
                continue
            sites_visited = None

        elev_ft = _estimate_elevation_gain(G, path)
        features = {
            "distance_mi": dist_miles,
            "elevation_ft": elev_ft,
            "avg_speed_mph": 15,
            "moving_time_min": dist_miles / 15 * 60,
            "suffer_score": 0,
            "avg_watts": 0,
            "pr_count": 0,
        }
        score = model_service.predict_score(athlete_id, features)

        novelty_pct = None
        route_segments = None
        if params.route_type == "novel":
            novelty_pct = _calculate_novelty_pct(G, path)
            route_segments = _encode_segments(G, path)

        polyline_str = _encode_polyline(G, path)
        elevation_profile = _build_elevation_profile(G, path)
        city_k = graph_service.city_key(params.start_lat, params.start_lng)
        route_id = str(uuid.uuid4())
        gpx_path = _save_gpx(G, path, route_id, sites_visited)

        results.append({
            "id": route_id,
            "path": path,
            "score": score,
            "dist_miles": round(dist_miles, 1),
            "elev_ft": round(elev_ft),
            "novelty_pct": novelty_pct,
            "polyline": polyline_str,
            "route_segments": route_segments,
            "elevation_profile": elevation_profile,
            "city_key": city_k,
            "gpx_path": gpx_path,
            "historic_sites": sites_visited,
        })

    if params.route_type == "hilly":
        results.sort(key=lambda x: -(x["elev_ft"] / 50 + x["score"]))
    elif params.route_type == "novel":
        results.sort(key=lambda x: -((x["novelty_pct"] or 0) / 20 + x["score"]))
    else:
        results.sort(key=lambda x: -x["score"])

    top = results[:5]
    return [
        RouteResult(
            id=r["id"],
            polyline=r["polyline"],
            route_segments=r["route_segments"],
            distance_mi=r["dist_miles"],
            elevation_ft=r["elev_ft"],
            predicted_score=r["score"],
            novelty_pct=r["novelty_pct"],
            historic_sites=[
                HistoricSite(name=s["name"], lat=s["lat"], lng=s["lng"])
                for s in r["historic_sites"]
            ] if r.get("historic_sites") else None,
            elevation_profile=r["elevation_profile"],
            city_key=r["city_key"],
            route_type=params.route_type,
            gpx_path=r["gpx_path"],
        )
        for r in top
    ]


# -- Historic sites ------------------------------------------------------------

def _load_historic_sites(
    G: nx.MultiDiGraph, start_lat: float, start_lng: float, distance_mi: float,
) -> list[dict]:
    city_k = graph_service.city_key(start_lat, start_lng)
    if city_k in _historic_sites_cache:
        return _historic_sites_cache[city_k]

    sites = _fetch_historic_sites_from_nps(start_lat, start_lng, distance_mi)
    if not sites:
        sites = _load_historic_sites_from_csv(start_lat, start_lng, distance_mi)

    max_dist = distance_mi * 1609.34 / 2 * 0.90
    valid = []
    for site in sites:
        d = ox.distance.great_circle(start_lat, start_lng, site["lat"], site["lng"])
        if d < max_dist:
            node = ox.distance.nearest_nodes(G, site["lng"], site["lat"])
            valid.append({
                "node": node,
                "name": site["name"],
                "lat": float(G.nodes[node]["y"]),
                "lng": float(G.nodes[node]["x"]),
            })

    _historic_sites_cache[city_k] = valid
    print(f"[route_service] {len(valid)} historic sites reachable near {city_k}")
    return valid


def _fetch_historic_sites_from_nps(lat: float, lng: float, distance_mi: float) -> list[dict]:
    url = "https://mapservices.nps.gov/arcgis/rest/services/cultural_resources/nrhp_locations/MapServer/0/query"
    radius_deg = distance_mi / 69.0
    envelope = {
        "xmin": lng - radius_deg,
        "ymin": lat - radius_deg,
        "xmax": lng + radius_deg,
        "ymax": lat + radius_deg,
        "spatialReference": {"wkid": 4326},
    }
    params = {
        "geometry": json.dumps(envelope),
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "RESNAME",
        "outSR": "4326",
        "f": "json",
        "returnGeometry": "true",
        "resultRecordCount": 2000,
    }
    try:
        resp = httpx.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        sites = []
        for f in data.get("features", []):
            geom = f.get("geometry", {})
            attrs = f.get("attributes", {})
            x, y = geom.get("x"), geom.get("y")
            name = attrs.get("RESNAME")
            if x and y and name:
                sites.append({"name": name, "lat": y, "lng": x})
        print(f"[route_service] Fetched {len(sites)} historic sites from NPS API")
        return sites
    except Exception as e:
        print(f"[route_service] NPS API failed: {e}, falling back to CSV")
        return []


def _load_historic_sites_from_csv(lat: float, lng: float, distance_mi: float) -> list[dict]:
    csv_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    sites = []
    for filename in os.listdir(csv_dir):
        if filename.endswith("_historic_sites.csv"):
            try:
                df = pd.read_csv(os.path.join(csv_dir, filename))
                if df.empty:
                    continue
                for _, row in df.iterrows():
                    if pd.notna(row.get("lat")) and pd.notna(row.get("lng")) and pd.notna(row.get("name")):
                        sites.append({"name": row["name"], "lat": row["lat"], "lng": row["lng"]})
            except Exception:
                continue
    print(f"[route_service] Loaded {len(sites)} historic sites from CSV files")
    return sites


# -- Historic loop generation --------------------------------------------------

def _generate_historic_loop(G, start_node, target_meters, target_miles, tolerance,
                            int_ids, int_coords, int_tree, heuristic, weight_attr,
                            historic_nodes, downtown_radius):
    start_lat = float(G.nodes[start_node]["y"])
    start_lng = float(G.nodes[start_node]["x"])

    for _ in range(100):
        try:
            site1 = random.choice(historic_nodes)
            wp1 = site1["node"]
            wp1_lat, wp1_lng = site1["lat"], site1["lng"]
            d1 = ox.distance.great_circle(start_lat, start_lng, wp1_lat, wp1_lng)

            sites_visited = [{"name": site1["name"], "lat": wp1_lat, "lng": wp1_lng}]
            site2 = None

            if d1 < downtown_radius:
                far = [s for s in historic_nodes
                       if ox.distance.great_circle(start_lat, start_lng, s["lat"], s["lng"]) > downtown_radius]
                if far:
                    site2 = random.choice(far)
                    wp2 = site2["node"]
                else:
                    rem = max(5000, target_meters - d1)
                    angle = random.uniform(0, 360)
                    rad = math.radians(angle)
                    dlat = (rem / 2 / 111320) * math.cos(rad)
                    dlng = (rem / 2 / (111320 * math.cos(math.radians(wp1_lat)))) * math.sin(rad)
                    wp2 = graph_service.nearest_intersection(
                        int_ids, int_coords, int_tree, wp1_lat + dlat, wp1_lng + dlng
                    )
            else:
                rem = max(5000, target_meters - d1)
                angle = random.uniform(0, 360)
                rad = math.radians(angle)
                dlat = (rem / 2 / 111320) * math.cos(rad)
                dlng = (rem / 2 / (111320 * math.cos(math.radians(wp1_lat)))) * math.sin(rad)
                wp2 = graph_service.nearest_intersection(
                    int_ids, int_coords, int_tree, wp1_lat + dlat, wp1_lng + dlng
                )

            if site2:
                sites_visited.append({"name": site2["name"], "lat": site2["lat"], "lng": site2["lng"]})

            seg1 = nx.astar_path(G, start_node, wp1, heuristic=heuristic, weight=weight_attr)
            seg1 = _remove_spurs(seg1)
            seg2 = nx.astar_path(G, wp1, wp2, heuristic=heuristic, weight=weight_attr)
            seg2 = _remove_spurs(seg2)
            seg3 = nx.astar_path(G, wp2, start_node, heuristic=heuristic, weight=weight_attr)
            seg3 = _remove_spurs(seg3)

            full_path = seg1 + seg2[1:] + seg3[1:]
            full_path = _remove_spurs(full_path)

            total_dist = sum(
                G.get_edge_data(u, v, 0).get("length", 0)
                for u, v in zip(full_path[:-1], full_path[1:])
            )
            dist_miles = total_dist * 0.000621371
            low = target_miles * (1 - tolerance)
            high = target_miles * (1 + tolerance)

            if not (low <= dist_miles <= high):
                continue

            all_edges = list(zip(full_path[:-1], full_path[1:]))
            unique_edges = len(set(all_edges))
            reuse_ratio = 1 - (unique_edges / len(all_edges)) if all_edges else 1
            if reuse_ratio > 0.15:
                continue

            if _has_detours(G, full_path):
                continue

            return full_path, dist_miles, sites_visited

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    return None


# -- Standard loop generation --------------------------------------------------

def _generate_loop(G, start_node, target_meters, target_miles, tolerance,
                   int_ids, int_coords, int_tree, heuristic, weight_attr):
    start_lat = float(G.nodes[start_node]["y"])
    start_lng = float(G.nodes[start_node]["x"])
    outbound_dist_m = target_meters / 3

    for _ in range(50):
        modified = []
        try:
            outbound_angle = random.uniform(0, 360)
            offset = random.uniform(15, 90)

            def get_wp_node(distance, angle):
                rad = math.radians(angle)
                dlat = (distance / 111320) * math.cos(rad)
                dlng = (distance / (111320 * math.cos(math.radians(start_lat)))) * math.sin(rad)
                return graph_service.nearest_intersection(
                    int_ids, int_coords, int_tree, start_lat + dlat, start_lng + dlng
                )

            def penalize(seg):
                for u, v in zip(seg[:-1], seg[1:]):
                    if G.has_edge(u, v):
                        for key in G[u][v]:
                            orig = G[u][v][key][weight_attr]
                            G[u][v][key][weight_attr] *= 10
                            modified.append((u, v, key, orig))

            def restore():
                for u, v, key, orig in modified:
                    if G.has_edge(u, v) and key in G[u][v]:
                        G[u][v][key][weight_attr] = orig

            wp1 = get_wp_node(outbound_dist_m, outbound_angle - offset)
            wp2 = get_wp_node(outbound_dist_m, outbound_angle + offset)

            seg1 = nx.astar_path(G, start_node, wp1, heuristic=heuristic, weight=weight_attr)
            seg1 = _remove_spurs(seg1)
            penalize(seg1)

            seg2 = nx.astar_path(G, wp1, wp2, heuristic=heuristic, weight=weight_attr)
            seg2 = _remove_spurs(seg2)
            penalize(seg2)

            seg3 = nx.astar_path(G, wp2, start_node, heuristic=heuristic, weight=weight_attr)
            seg3 = _remove_spurs(seg3)

            restore()
            modified = []

            full_path = seg1 + seg2[1:] + seg3[1:]
            full_path = _remove_spurs(full_path)

            total_dist = sum(
                G.get_edge_data(u, v, 0).get("length", 0)
                for u, v in zip(full_path[:-1], full_path[1:])
            )
            dist_miles = total_dist * 0.000621371
            low = target_miles * (1 - tolerance)
            high = target_miles * (1 + tolerance)

            if not (low <= dist_miles <= high):
                outbound_dist_m *= (target_miles / dist_miles) ** 0.5 if dist_miles > 0 else 1
                continue

            midpoint_idx = len(full_path) // 2
            mid_node = full_path[midpoint_idx]
            mid_lat = float(G.nodes[mid_node]["y"])
            mid_lng = float(G.nodes[mid_node]["x"])
            dlat = math.radians(mid_lat - start_lat)
            dlng = math.radians(mid_lng - start_lng)
            a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(start_lat)) * math.cos(math.radians(mid_lat)) * math.sin(dlng / 2) ** 2
            mid_dist_miles = 3958.8 * 2 * math.asin(math.sqrt(a))
            if mid_dist_miles < 3.0:
                continue

            all_edges = list(zip(full_path[:-1], full_path[1:]))
            unique_edges = len(set(all_edges))
            reuse_ratio = 1 - (unique_edges / len(all_edges)) if all_edges else 1
            if reuse_ratio > 0.15:
                continue

            if _has_detours(G, full_path):
                continue

            return full_path, dist_miles

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            for u, v, key, orig in modified:
                if G.has_edge(u, v) and key in G[u][v]:
                    G[u][v][key][weight_attr] = orig
            continue

    return None, None


# -- Helpers -------------------------------------------------------------------

def _make_heuristic(G):
    coords = {n: (float(d["y"]), float(d["x"])) for n, d in G.nodes(data=True)}
    def heuristic(u, v):
        u_lat, u_lng = coords[u]
        v_lat, v_lng = coords[v]
        return math.sqrt((u_lat - v_lat) ** 2 + (u_lng - v_lng) ** 2) * 111320
    return heuristic


def _remove_spurs(path):
    if len(path) < 3:
        return path
    stack = [path[0]]
    for node in path[1:]:
        if len(stack) >= 2 and stack[-2] == node:
            stack.pop()
        else:
            stack.append(node)
    return stack


def _has_detours(G, path, max_detour_ratio=2.5):
    window = 10
    for i in range(0, len(path) - window, window // 2):
        segment = path[i:i + window]
        start = G.nodes[segment[0]]
        end = G.nodes[segment[-1]]
        lat1, lng1 = math.radians(float(start["y"])), math.radians(float(start["x"]))
        lat2, lng2 = math.radians(float(end["y"])), math.radians(float(end["x"]))
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
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


def _estimate_elevation_gain(G, path):
    gain = 0
    prev_elev = None
    for node in path:
        elev = G.nodes[node].get("elevation")
        try:
            elev = float(elev)
            if elev <= 0:
                elev = None
        except (TypeError, ValueError):
            elev = None
        if elev is not None and prev_elev is not None:
            diff = elev - prev_elev
            if diff > 1.0:
                gain += diff
        prev_elev = elev
    return gain * 3.28084


def _build_elevation_profile(G, path, max_points=60):
    """
    Build a downsampled elevation profile for charting: a list of
    [cumulative_distance_mi, elevation_ft] points along the route.
    Reuses _path_to_coords (which already resolves per-point elevation in
    meters) and ox.distance.great_circle for cumulative distance. Returns
    None if no usable elevation data exists.
    """
    coords = _path_to_coords(G, path)
    if not coords:
        return None

    points = []
    cum_m = 0.0
    prev = None
    last_elev_ft = 0.0
    have_elev = False

    for lat, lng, elev in coords:
        if prev is not None:
            cum_m += ox.distance.great_circle(prev[0], prev[1], lat, lng)
        elev_ft = None
        try:
            if elev is not None and float(elev) > 0:
                elev_ft = float(elev) * 3.28084
                have_elev = True
        except (TypeError, ValueError):
            elev_ft = None
        # Carry the last known elevation across gaps so the line stays continuous.
        last_elev_ft = elev_ft if elev_ft is not None else last_elev_ft
        points.append([cum_m * 0.000621371, last_elev_ft])
        prev = (lat, lng)

    if not have_elev or len(points) < 2:
        return None

    # Downsample evenly to keep the payload small, always keeping the last point.
    if len(points) > max_points:
        step = len(points) / max_points
        sampled = [points[int(i * step)] for i in range(max_points)]
        sampled.append(points[-1])
        points = sampled

    return [[round(d, 3), round(e, 1)] for d, e in points]


def _calculate_novelty_pct(G, path):
    novel_dist = 0
    total_dist = 0
    for u, v in zip(path[:-1], path[1:]):
        edge_data = G.get_edge_data(u, v) or G.get_edge_data(v, u)
        if edge_data is None:
            continue
        data = edge_data.get(0, list(edge_data.values())[0])
        dist = data.get("length", 0)
        total_dist += dist
        if data.get("ridden_count", 0) == 0:
            novel_dist += dist
    return round(novel_dist / total_dist * 100, 1) if total_dist > 0 else 0


def _path_to_coords(G, path):
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
                u_lng = float(G.nodes[u]["x"])
                if abs(geom_coords[0][0] - u_lng) > 0.0001:
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
                d = G.nodes[node]
                elev = d.get("elevation")
                try:
                    elev = float(elev) if elev is not None else None
                except (TypeError, ValueError):
                    elev = None
                coords.append((float(d["y"]), float(d["x"]), elev))
    return coords


def _encode_polyline(G, path):
    coords = _path_to_coords(G, path)
    return pl.encode([(lat, lng) for lat, lng, _ in coords])


def _encode_segments(G, path):
    segments = []
    current_coords = []
    current_is_new = None

    for u, v in zip(path[:-1], path[1:]):
        edge_data = G.get_edge_data(u, v) or G.get_edge_data(v, u)
        if edge_data is None:
            continue
        data = edge_data.get(0, list(edge_data.values())[0])
        is_new = data.get("ridden_count", 0) == 0

        if "geometry" in data:
            geom_coords = list(data["geometry"].coords)
            if u in G.nodes:
                u_lng = float(G.nodes[u]["x"])
                if abs(geom_coords[0][0] - u_lng) > 0.0001:
                    geom_coords = list(reversed(geom_coords))
            edge_coords = [(lat, lng) for lng, lat in geom_coords]
        else:
            edge_coords = [
                (float(G.nodes[u]["y"]), float(G.nodes[u]["x"])),
                (float(G.nodes[v]["y"]), float(G.nodes[v]["x"])),
            ]

        if current_is_new is None:
            current_is_new = is_new
            current_coords.extend(edge_coords)
        elif current_is_new == is_new:
            current_coords.extend(edge_coords[1:])
        else:
            segments.append({"polyline": pl.encode(current_coords), "is_new": current_is_new})
            current_is_new = is_new
            current_coords = edge_coords

    if current_coords:
        segments.append({"polyline": pl.encode(current_coords), "is_new": current_is_new})

    return json.dumps(segments)


def _save_gpx(G, path, route_id, historic_sites=None):
    gpx_dir = os.path.join(settings.data_dir, "gpx")
    os.makedirs(gpx_dir, exist_ok=True)
    coords = _path_to_coords(G, path)

    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack(name=f"Route {route_id[:8]}")
    gpx.tracks.append(track)
    segment = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(segment)
    for lat, lng, elev in coords:
        elev_clean = elev if (elev is not None and elev > 0) else None
        segment.points.append(gpxpy.gpx.GPXTrackPoint(lat, lng, elevation=elev_clean))

    if historic_sites:
        for site in historic_sites:
            wp = gpxpy.gpx.GPXWaypoint(site["lat"], site["lng"], name=site["name"])
            gpx.waypoints.append(wp)

    filepath = os.path.join(gpx_dir, f"{route_id}.gpx")
    with open(filepath, "w") as f:
        f.write(gpx.to_xml())
    return filepath
