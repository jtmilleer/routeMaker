# LAYER: Service — Graph Cache
# PURPOSE: Loads, caches, and manages road network graphs (NetworkX MultiDiGraphs).
#          Graphs are shared across users — two users near Iowa City share one file.
#          Graphs are keyed by a "city_key": rounded lat/lng e.g. "41.65_-91.53".
#          In-memory cache avoids re-loading the 126MB graphml on every request.
#          Background build is async-safe: build_status tracks progress for polling.

import asyncio
import math
import os
import time
from typing import Optional

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from scipy.spatial import KDTree

import srtm

from backend.core.config import settings

# ── In-memory caches ──────────────────────────────────────────────────────────

# city_key → loaded NetworkX graph
_graph_cache: dict[str, nx.MultiDiGraph] = {}

# city_key → { status, progress_pct, node_count, edge_count, error }
_build_status: dict[str, dict] = {}


# ── Pre-seeded cities ─────────────────────────────────────────────────────────

PRESET_CITIES = {
    "iowa_city": (41.6543, -91.5267),
    "madison":   (43.0731, -89.4012),
    "des_moines": (41.5868, -93.6250),
}

NETWORK_DIST = 50000  # meters radius around center point


# ── City key ──────────────────────────────────────────────────────────────────

def city_key(lat: float, lng: float) -> str:
    """
    Round lat/lng to 2 decimal places and return a stable cache key.
    Two points within ~1km of each other share the same key (and graph).
    Example: city_key(41.6543, -91.5267) → "41.65_-91.53"
    """
    return f"{round(lat, 2)}_{round(lng, 2)}"


# ── Graph path ────────────────────────────────────────────────────────────────

def graph_path(key: str) -> str:
    """Return the filesystem path for a cached graph file."""
    os.makedirs(settings.graphs_dir, exist_ok=True)
    return os.path.join(settings.graphs_dir, f"{key}.graphml")


# ── Status helpers ────────────────────────────────────────────────────────────

def get_graph_status(key: str) -> dict:
    """Return current build status for a city key."""
    path = graph_path(key)
    if key in _graph_cache:
        G = _graph_cache[key]
        return {"city_key": key, "status": "ready", "progress_pct": 100,
                "node_count": len(G.nodes), "edge_count": len(G.edges)}
    if os.path.exists(path) and key not in _build_status:
        return {"city_key": key, "status": "ready", "progress_pct": 100,
                "node_count": None, "edge_count": None}
    if key in _build_status:
        return {"city_key": key, **_build_status[key]}
    return {"city_key": key, "status": "not_found", "progress_pct": 0,
            "node_count": None, "edge_count": None}


# ── Graph loading ─────────────────────────────────────────────────────────────

def _load_graph(key: str) -> nx.MultiDiGraph:
    """Load a graph from disk into the memory cache. Blocking — run in thread pool."""
    path = graph_path(key)
    G = ox.load_graphml(path)
    _graph_cache[key] = G
    return G


def get_graph_sync(lat: float, lng: float) -> Optional[nx.MultiDiGraph]:
    """
    Synchronous graph retrieval (for use inside thread pool during route generation).
    Returns the graph if available in cache or on disk; None if not yet built.
    """
    key = city_key(lat, lng)
    if key in _graph_cache:
        return _graph_cache[key]
    path = graph_path(key)
    if os.path.exists(path):
        return _load_graph(key)
    return None


# ── Graph building ────────────────────────────────────────────────────────────

async def request_graph_build(lat: float, lng: float):
    """
    Kick off a background graph build for a custom location.
    Returns immediately; check get_graph_status() to poll progress.
    """
    key = city_key(lat, lng)
    if key in _graph_cache or os.path.exists(graph_path(key)):
        return key  # already exists

    if key in _build_status and _build_status[key]["status"] == "building":
        return key  # already in progress

    _build_status[key] = {"status": "building", "progress_pct": 0,
                          "node_count": None, "edge_count": None}

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _build_graph_blocking, lat, lng, key)
    return key


def _build_graph_blocking(lat: float, lng: float, key: str):
    """
    Download, process, and save a road network graph. Blocking — runs in a thread.
    Updates _build_status at each step so the frontend polling can show progress.
    """
    try:
        _build_status[key] = {"status": "building", "progress_pct": 5,
                               "node_count": None, "edge_count": None}

        # Step 1: Download from OpenStreetMap
        G = ox.graph_from_point((lat, lng), dist=NETWORK_DIST,
                                 network_type="bike", simplify=True)
        _build_status[key]["progress_pct"] = 20

        # Step 2: Remove highways unsuitable for cycling
        edges_to_remove = [
            (u, v, k) for u, v, k, d in G.edges(keys=True, data=True)
            if d.get("highway") in ("motorway", "trunk", "primary",
                                    "motorway_link", "trunk_link")
        ]
        G.remove_edges_from(edges_to_remove)
        _build_status[key]["progress_pct"] = 30

        _build_status[key]["progress_pct"] = 40

        # Step 4: Elevation via local SRTM tiles
        srtm_cache = os.path.join(settings.data_dir, "srtm_cache")
        os.makedirs(srtm_cache, exist_ok=True)
        elevation_svc = srtm.SrtmService(srtm_cache)
        node_ids = list(G.nodes)
        total = len(node_ids)
        coords = [(float(G.nodes[n]["y"]), float(G.nodes[n]["x"])) for n in node_ids]
        elevations = elevation_svc.get_elevations_batch(coords)
        for i, (node_id, elev) in enumerate(zip(node_ids, elevations)):
            G.nodes[node_id]["elevation"] = elev if elev != srtm.VOID_VALUE else 0.0
            if i % 5000 == 0:
                pct = 40 + int((i / total) * 40)
                _build_status[key]["progress_pct"] = pct

        _build_status[key]["progress_pct"] = 80

        # Step 5: Edge grades
        G = ox.add_edge_grades(G)
        _build_status[key]["progress_pct"] = 85

        # Step 6: Keep only largest strongly connected component
        largest_scc = max(nx.strongly_connected_components(G), key=len)
        G.remove_nodes_from(set(G.nodes) - largest_scc)
        _build_status[key]["progress_pct"] = 90

        # Step 7: Save to disk
        ox.save_graphml(G, graph_path(key))
        _graph_cache[key] = G

        _build_status[key] = {
            "status": "ready", "progress_pct": 100,
            "node_count": len(G.nodes), "edge_count": len(G.edges),
        }
        print(f"[graph_service] Build complete for {key}: {len(G.nodes)} nodes, {len(G.edges)} edges")

    except Exception as e:
        _build_status[key] = {"status": "error", "progress_pct": 0,
                               "node_count": None, "edge_count": None,
                               "error": str(e)}
        print(f"[graph_service] Build FAILED for {key}: {e}")


# ── Startup seeding ───────────────────────────────────────────────────────────

async def seed_preset_cities():
    """
    Called at app startup. For each preset city, load the graph from disk if it
    exists, or kick off a background download if it doesn't.
    The Iowa City graph already exists as data/iowa_city_network.graphml — it
    will be detected and loaded automatically once copied to data/graphs/.
    """
    for name, (lat, lng) in PRESET_CITIES.items():
        key = city_key(lat, lng)
        path = graph_path(key)

        if os.path.exists(path):
            print(f"[graph_service] Loading cached graph for {name} ({key})...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _load_graph, key)
            print(f"[graph_service] {name} ready: {len(_graph_cache[key].nodes)} nodes")
        else:
            print(f"[graph_service] Graph for {name} not found — queuing background build")
            await request_graph_build(lat, lng)


# ── Spatial index ─────────────────────────────────────────────────────────────

def build_intersection_index(G: nx.MultiDiGraph) -> tuple:
    """
    Build a KDTree spatial index over intersection nodes (degree ≥ 3).
    Returns (ids, coords_array, KDTree) — pass all three to nearest_intersection().
    """
    G_undir = G.to_undirected()
    nodes = [(n, float(G.nodes[n]["y"]), float(G.nodes[n]["x"]))
             for n in G.nodes if G_undir.degree(n) >= 3]
    ids = [n[0] for n in nodes]
    coords = np.array([(n[1], n[2]) for n in nodes])
    tree = KDTree(coords)
    return ids, coords, tree


def nearest_intersection(ids, coords, tree, lat: float, lng: float) -> int:
    """Find the nearest intersection node to (lat, lng) using the KDTree."""
    _, idx = tree.query([lat, lng])
    return ids[idx]


# ── Ridden-edge marking (for novel routes, per-user) ─────────────────────────

def mark_ridden_edges(G: nx.MultiDiGraph, rides_df: pd.DataFrame,
                      novelty_factor: float = 5.0) -> nx.MultiDiGraph:
    """
    Apply routing weights based on which edges the user has previously ridden.
    Ridden edges get a penalty of novelty_factor × length to make the algorithm
    prefer unridden roads. Called per-request with the requesting user's rides.

    Unlike the original script (which read a global CSV), this receives the
    user's rides as a DataFrame so novelty is personalized per user.
    """
    # Initialize all edges
    for u, v, k, d in G.edges(keys=True, data=True):
        d["ridden_count"] = 0
        d["routing_weight"] = d["length"]

    df = rides_df[rides_df["polyline"].notna()] if len(rides_df) > 0 else pd.DataFrame()
    if df.empty:
        return G

    import polyline as pl

    X, Y, ride_indices = [], [], []
    for ride_idx, poly in enumerate(df["polyline"]):
        for lat, lng in pl.decode(poly):
            X.append(lng)
            Y.append(lat)
            ride_indices.append(ride_idx)

    if not X:
        return G

    nearest_edges = ox.distance.nearest_edges(G, X, Y)
    ride_edge_counts: dict = {}
    for ride_idx, (u, v, _) in zip(ride_indices, nearest_edges):
        if ride_idx not in ride_edge_counts:
            ride_edge_counts[ride_idx] = {}
        pair = (u, v)
        ride_edge_counts[ride_idx][pair] = ride_edge_counts[ride_idx].get(pair, 0) + 1

    for ride_idx, counts in ride_edge_counts.items():
        for (u, v), c in counts.items():
            edge_data = G.get_edge_data(u, v) or G.get_edge_data(v, u)
            if not edge_data:
                continue
            data = edge_data.get(0, list(edge_data.values())[0])
            length = data.get("length", 0)
            # Filter GPS snapping noise on long edges
            if length > 200 and c <= (length / 300.0):
                continue
            for key in (G[u][v] if G.has_edge(u, v) else {}):
                G[u][v][key]["ridden_count"] += 1
            for key in (G[v][u] if G.has_edge(v, u) else {}):
                G[v][u][key]["ridden_count"] += 1

    # Fill tiny gaps between ridden nodes (GPS snapping skips)
    ridden_nodes = {n for u, v, k, d in G.edges(keys=True, data=True)
                    if d.get("ridden_count", 0) > 0 for n in (u, v)}
    for u, v, k, d in G.edges(keys=True, data=True):
        if d.get("ridden_count", 0) == 0 and d.get("length", 0) < 20:
            if u in ridden_nodes and v in ridden_nodes:
                d["ridden_count"] += 1

    # Apply routing weights
    for u, v, k, d in G.edges(keys=True, data=True):
        d["routing_weight"] = (d["length"] * novelty_factor
                               if d.get("ridden_count", 0) > 0
                               else d["length"])

    return G


def mark_hilly_edges(G: nx.MultiDiGraph, hilly_factor: float) -> nx.MultiDiGraph:
    """
    Adjust routing weights to favor hilly roads (for hilly route type).
    Higher hilly_factor = stronger preference for steep roads.
    """
    for u, v, k, d in G.edges(keys=True, data=True):
        grade_abs = min(d.get("grade_abs", 0.0), 0.15)
        d["routing_weight"] = d["length"] / (1 + grade_abs * hilly_factor)
    return G
