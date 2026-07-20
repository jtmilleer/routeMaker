# LAYER: Service — Street Coverage
# PURPOSE: Powers the "100% coverage" feature. Given a user's saved home base and
#          a radius, it measures how many of the ridable streets in that area the
#          user has already ridden (reusing the same ride->edge matching used for
#          novel routes), and generates loops that send the user down the streets
#          they HAVEN'T ridden yet, to help close the gaps.
#
#          All functions here are CPU-bound and meant to be run in a thread pool
#          (see routers/coverage.py), exactly like route generation.

import logging
import uuid

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import polyline as pl

from backend.models.schemas import RouteResult
from backend.services import graph_service, model_service, route_service

logger = logging.getLogger("routemaker.coverage")

MILES_PER_METER = 0.000621371
METERS_PER_MILE = 1.0 / MILES_PER_METER

# How far beyond the radius a ride point may sit and still count as "in area".
# Covers GPS jitter and the gap between a snapped point and the edge midpoint.
SNAP_BUFFER_M = 250.0

# Multiplier applied to already-ridden / already-covered edges when routing the
# completionist loop, so the pathfinder strongly prefers fresh streets.
COVERAGE_PENALTY = 6.0


# ── Radius subgraph ────────────────────────────────────────────────────────────

def _radius_subgraph(G: nx.MultiDiGraph, lat: float, lng: float,
                     radius_mi: float) -> nx.MultiDiGraph:
    """
    Return a *copy* of the part of G within `radius_mi` (crow-flies) of the center.
    Working on a small copy keeps matching/routing fast and — crucially — means we
    never mutate the shared, cached city graph.
    """
    radius_m = radius_mi * METERS_PER_MILE
    nodes = list(G.nodes)
    if not nodes:
        return G.subgraph([]).copy()
    ys = np.array([float(G.nodes[n]["y"]) for n in nodes])  # lat
    xs = np.array([float(G.nodes[n]["x"]) for n in nodes])  # lng
    dists = ox.distance.great_circle(lat, lng, ys, xs)
    kept = [n for n, d in zip(nodes, dists) if d <= radius_m]
    return G.subgraph(kept).copy()


# ── Ride matching (restricted to the area) ──────────────────────────────────────

def _rides_within_radius(rides_df: pd.DataFrame, lat: float, lng: float,
                         radius_mi: float) -> pd.DataFrame:
    """
    Clip the user's ride polylines to just the points within the area (plus a small
    buffer) and re-encode them. This is what makes reusing graph_service's
    point-snapping safe on a subgraph: without it, points from rides far away would
    snap to the subgraph's boundary streets and falsely mark them ridden.
    """
    if rides_df is None or len(rides_df) == 0 or "polyline" not in rides_df:
        return pd.DataFrame(columns=["polyline", "distance_mi"])

    limit_m = radius_mi * METERS_PER_MILE + SNAP_BUFFER_M
    rows = []
    for _, r in rides_df.iterrows():
        poly = r.get("polyline")
        if not poly:
            continue
        pts = pl.decode(poly)
        if not pts:
            continue
        arr = np.asarray(pts, dtype=float)  # columns: lat, lng
        d = ox.distance.great_circle(lat, lng, arr[:, 0], arr[:, 1])
        mask = d <= limit_m
        if not mask.any():
            continue
        in_pts = [(float(p[0]), float(p[1])) for p in arr[mask]]
        rows.append({"polyline": pl.encode(in_pts),
                     "distance_mi": r.get("distance_mi")})
    return pd.DataFrame(rows, columns=["polyline", "distance_mi"])


def _marked_subgraph(G: nx.MultiDiGraph, rides_df: pd.DataFrame,
                     lat: float, lng: float, radius_mi: float) -> nx.MultiDiGraph:
    """Build the radius subgraph and tag each edge with ridden_count via the
    existing matcher, using only ride points inside the area."""
    H = _radius_subgraph(G, lat, lng, radius_mi)
    local_rides = _rides_within_radius(rides_df, lat, lng, radius_mi)
    # mark_ridden_edges initializes ridden_count/routing_weight on every edge,
    # so this is safe even when the user has no rides in the area.
    graph_service.mark_ridden_edges(H, local_rides)
    return H


def _edge_polyline(G: nx.MultiDiGraph, u, v, data) -> str:
    """Encode a single edge's geometry (or its endpoints) as a polyline string."""
    if "geometry" in data:
        coords = [(lat, lng) for lng, lat in data["geometry"].coords]
    else:
        coords = [
            (float(G.nodes[u]["y"]), float(G.nodes[u]["x"])),
            (float(G.nodes[v]["y"]), float(G.nodes[v]["x"])),
        ]
    return pl.encode(coords)


def _unique_undirected_edges(G: nx.MultiDiGraph):
    """
    Yield (u, v, data) once per physical street, collapsing the MultiDiGraph's two
    directions. Parallel edges between the same nodes are kept distinct by osmid.
    """
    seen = set()
    for u, v, data in G.edges(data=True):
        key = (frozenset((u, v)), str(data.get("osmid")))
        if key in seen:
            continue
        seen.add(key)
        yield u, v, data


# ── Coverage stats + segments ───────────────────────────────────────────────────

def compute_coverage(G: nx.MultiDiGraph, rides_df: pd.DataFrame,
                     lat: float, lng: float, radius_mi: float,
                     activity: str = "ride") -> dict:
    """
    Measure street coverage within `radius_mi` of (lat, lng) and return per-street
    segments for the map. Coverage is measured by street length (more meaningful
    than edge count). `rides_df` should already be filtered to the chosen activity.
    """
    H = _marked_subgraph(G, rides_df, lat, lng, radius_mi)

    total_m = 0.0
    ridden_m = 0.0
    segments = []
    for u, v, data in _unique_undirected_edges(H):
        length = float(data.get("length", 0.0))
        ridden = data.get("ridden_count", 0) > 0
        total_m += length
        if ridden:
            ridden_m += length
        segments.append({"polyline": _edge_polyline(H, u, v, data), "ridden": ridden})

    coverage_pct = round(ridden_m / total_m * 100, 1) if total_m > 0 else 0.0
    return {
        "city_key": graph_service.city_key(lat, lng),
        "center_lat": lat,
        "center_lng": lng,
        "radius_mi": radius_mi,
        "activity": activity,
        "coverage_pct": coverage_pct,
        "total_mi": round(total_m * MILES_PER_METER, 2),
        "ridden_mi": round(ridden_m * MILES_PER_METER, 2),
        "segments": segments,
    }


# ── Completionist route generation ──────────────────────────────────────────────

def _largest_scc(H: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Largest strongly connected component as a copy — guarantees A* solvability
    (the radius cut can leave dead-end stubs that point only outside the area)."""
    if H.number_of_nodes() == 0:
        return H
    largest = max(nx.strongly_connected_components(H), key=len)
    return H.subgraph(largest).copy()


def _path_distance_m(G: nx.MultiDiGraph, path) -> float:
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        ed = G.get_edge_data(u, v)
        if ed:
            total += ed.get(0, list(ed.values())[0]).get("length", 0.0)
    return total


def _mark_path_covered(G: nx.MultiDiGraph, path):
    """Treat every edge along `path` as covered so the next A* avoids re-using it,
    biasing each leg toward unridden streets."""
    for u, v in zip(path[:-1], path[1:]):
        for a, b in ((u, v), (v, u)):
            if G.has_edge(a, b):
                for k in G[a][b]:
                    d = G[a][b][k]
                    d["ridden_count"] = d.get("ridden_count", 0) + 1
                    d["routing_weight"] = d.get("length", 0.0) * COVERAGE_PENALTY


def _build_coverage_loop(H_work: nx.MultiDiGraph, start, target_m: float, rng):
    """
    Greedily stitch together a loop that favors uncovered streets: repeatedly route
    to the nearest uncovered edge, mark what we traverse as covered, and stop once
    the distance budget is reached. Returns a node path (closed loop) or None.
    """
    coords = {n: (float(H_work.nodes[n]["y"]), float(H_work.nodes[n]["x"]))
              for n in H_work.nodes}
    heuristic = route_service._make_heuristic(H_work)

    path = [start]
    cur = start
    safety = 0
    max_steps = 200

    while _path_distance_m(H_work, path) < target_m and safety < max_steps:
        safety += 1

        # Candidate destinations: endpoints of still-uncovered streets.
        uncovered_nodes = set()
        for u, v, d in H_work.edges(data=True):
            if d.get("ridden_count", 0) == 0:
                uncovered_nodes.add(u)
                uncovered_nodes.add(v)
        uncovered_nodes.discard(cur)
        if not uncovered_nodes:
            break

        # Pick among the nearest few uncovered endpoints (a little randomness so
        # repeated calls explore different parts of the area).
        clat, clng = coords[cur]
        ranked = sorted(
            uncovered_nodes,
            key=lambda n: (coords[n][0] - clat) ** 2 + (coords[n][1] - clng) ** 2,
        )
        target = rng.choice(ranked[:5]) if len(ranked) >= 5 else ranked[0]

        try:
            leg = nx.astar_path(H_work, cur, target,
                                heuristic=heuristic, weight="routing_weight")
        except nx.NetworkXNoPath:
            _mark_path_covered(H_work, [target, target])  # ignore this target
            continue

        if len(leg) > 1:
            path.extend(leg[1:])
            _mark_path_covered(H_work, leg)
            cur = target

    # Close the loop back to the start.
    if cur != start:
        try:
            back = nx.astar_path(H_work, cur, start,
                                 heuristic=heuristic, weight="routing_weight")
            path.extend(back[1:])
        except nx.NetworkXNoPath:
            return None

    path = route_service._remove_spurs(path)
    return path if len(path) >= 3 else None


def generate_coverage_route(G: nx.MultiDiGraph, athlete_id: int,
                            rides_df: pd.DataFrame, lat: float, lng: float,
                            radius_mi: float, distance_mi: float) -> list[RouteResult]:
    """
    Generate up to 3 candidate loops that prioritize streets the user hasn't ridden
    within the area. Routes are scored/encoded with the same helpers as regular
    route generation, and use route_type="coverage".
    """
    import random

    H_marked = _largest_scc(_marked_subgraph(G, rides_df, lat, lng, radius_mi))
    if H_marked.number_of_edges() == 0:
        return []

    # Nothing left to discover?
    if all(d.get("ridden_count", 0) > 0 for _, _, d in H_marked.edges(data=True)):
        return []

    start = ox.distance.nearest_nodes(H_marked, lng, lat)
    target_m = distance_mi * METERS_PER_MILE

    candidates = []
    seen_polylines = set()
    for seed in range(6):  # a few attempts to collect up to 3 distinct loops
        if len(candidates) >= 3:
            break
        rng = random.Random(seed)
        H_work = H_marked.copy()  # fresh covered-state per attempt
        for u, v, d in H_work.edges(data=True):
            d["routing_weight"] = (d.get("length", 0.0) *
                                   (COVERAGE_PENALTY if d.get("ridden_count", 0) > 0 else 1.0))

        path = _build_coverage_loop(H_work, start, target_m, rng)
        if not path:
            continue

        polyline_str = route_service._encode_polyline(H_marked, path)
        if polyline_str in seen_polylines:
            continue
        seen_polylines.add(polyline_str)

        dist_mi = round(_path_distance_m(H_marked, path) * MILES_PER_METER, 1)
        elev_ft = round(route_service._estimate_elevation_gain(H_marked, path))
        novelty_pct = route_service._calculate_novelty_pct(H_marked, path)
        route_segments = route_service._encode_segments(H_marked, path)
        elevation_profile = route_service._build_elevation_profile(H_marked, path)
        route_id = str(uuid.uuid4())
        gpx_path = route_service._save_gpx(H_marked, path, route_id)
        score = model_service.predict_score(athlete_id, {
            "distance_mi": dist_mi, "elevation_ft": elev_ft,
            "avg_speed_mph": 15, "moving_time_min": dist_mi / 15 * 60,
            "suffer_score": 0, "avg_watts": 0, "pr_count": 0,
        })

        candidates.append(RouteResult(
            id=route_id,
            polyline=polyline_str,
            route_segments=route_segments,
            distance_mi=dist_mi,
            elevation_ft=elev_ft,
            predicted_score=score,
            novelty_pct=novelty_pct,
            elevation_profile=elevation_profile,
            city_key=graph_service.city_key(lat, lng),
            route_type="coverage",
            gpx_path=gpx_path,
        ))

    # Most "new street" first.
    candidates.sort(key=lambda r: -(r.novelty_pct or 0))
    return candidates
