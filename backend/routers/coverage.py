# LAYER: Router — Street Coverage
# PURPOSE: Endpoints for the "100% coverage" feature. The user saves a home base,
#          views how much of the surrounding street network they've ridden, and
#          generates loops that fill in the gaps. CPU-bound work (ride matching,
#          pathfinding) runs in a thread pool so the event loop is never blocked —
#          same pattern as routers/routes.py.

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.db import get_db
from backend.core.security import get_current_user
from backend.models.db_models import GeneratedRoute, Ride, User
from backend.models.schemas import (
    CoverageResponse, CoverageRouteRequest, HomeBaseRequest, RouteResult,
)
from backend.services import coverage_service, graph_service

router = APIRouter(prefix="/api/coverage", tags=["coverage"])
_executor = ThreadPoolExecutor(max_workers=2)


async def _load_rides_df(db: AsyncSession, athlete_id: int,
                         activity: str) -> pd.DataFrame:
    """Load the user's activities (with polylines) as a DataFrame, filtered to the
    chosen activity ("walk", "ride", or "both")."""
    stmt = select(Ride).where(Ride.athlete_id == athlete_id)
    if activity in ("walk", "ride"):
        stmt = stmt.where(Ride.activity_type == activity)
    result = await db.execute(stmt)
    rides = result.scalars().all()
    return pd.DataFrame([
        {"polyline": r.polyline, "distance_mi": r.distance_mi}
        for r in rides if r.polyline
    ], columns=["polyline", "distance_mi"])


async def _require_home(db: AsyncSession, athlete_id: int) -> tuple[float, float]:
    user = await db.get(User, athlete_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    if user.home_lat is None or user.home_lng is None:
        raise HTTPException(status_code=400, detail="No home base set. Set one first.")
    return user.home_lat, user.home_lng


def _require_graph(lat: float, lng: float):
    """Return a graph that already covers this point. Each graph spans ~50km, so a
    nearby preset/built graph is reused rather than requiring an exact-key match.
    Raises 503 with the build status if nothing covers it yet."""
    found = graph_service.find_covering_graph(lat, lng)
    if found is not None:
        return found[1]
    city_k = graph_service.city_key(lat, lng)
    status = graph_service.get_graph_status(city_k)
    raise HTTPException(
        status_code=503,
        detail=f"Map for this area is not ready yet (status: {status['status']}). "
               "Please wait and try again.",
    )


@router.put("/home")
async def set_home_base(
    body: HomeBaseRequest,
    db: AsyncSession = Depends(get_db),
    athlete_id: int = Depends(get_current_user),
):
    """Save the user's coverage-area center. Kicks off a background graph build for
    the area if it isn't already available (frontend polls the existing graph
    status banner)."""
    user = await db.get(User, athlete_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    user.home_lat = body.lat
    user.home_lng = body.lng
    await db.flush()

    # Reuse a nearby graph if one already covers this point (each spans ~50km).
    # Only build a fresh network when there's genuinely nothing nearby.
    found = graph_service.find_covering_graph(body.lat, body.lng)
    if found is not None:
        city_k = found[0]
    else:
        city_k = await graph_service.request_graph_build(body.lat, body.lng)
    return {"home_lat": body.lat, "home_lng": body.lng, "city_key": city_k}


def _normalize_activity(activity: str) -> str:
    return activity if activity in ("walk", "ride", "both") else "ride"


@router.get("", response_model=CoverageResponse)
async def get_coverage(
    radius_mi: float = 1.0,
    activity: str = "ride",
    db: AsyncSession = Depends(get_db),
    athlete_id: int = Depends(get_current_user),
):
    """Return coverage stats + per-street segments around the user's home base."""
    radius_mi = max(0.25, min(radius_mi, 5.0))
    activity = _normalize_activity(activity)
    lat, lng = await _require_home(db, athlete_id)
    G = _require_graph(lat, lng)
    rides_df = await _load_rides_df(db, athlete_id, activity)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor, coverage_service.compute_coverage,
        G, rides_df, lat, lng, radius_mi, activity,
    )
    return result


@router.post("/route", response_model=list[RouteResult])
async def generate_coverage_route(
    body: CoverageRouteRequest,
    db: AsyncSession = Depends(get_db),
    athlete_id: int = Depends(get_current_user),
):
    """Generate loops that prioritize streets the user hasn't covered in the area."""
    activity = _normalize_activity(body.activity)
    lat, lng = await _require_home(db, athlete_id)
    G = _require_graph(lat, lng)
    rides_df = await _load_rides_df(db, athlete_id, activity)

    # Walks are shorter outings — default to a smaller loop when unspecified.
    distance_mi = body.distance_mi or (5.0 if activity == "walk" else 15.0)

    loop = asyncio.get_event_loop()
    results: list[RouteResult] = await loop.run_in_executor(
        _executor, coverage_service.generate_coverage_route,
        G, athlete_id, rides_df, lat, lng, body.radius_mi, distance_mi,
    )

    if not results:
        raise HTTPException(
            status_code=400,
            detail="No uncovered streets to route here — this area looks fully covered!",
        )

    for r in results:
        db.add(GeneratedRoute(
            id=r.id,
            athlete_id=athlete_id,
            route_type=r.route_type,
            distance_mi=r.distance_mi,
            elevation_ft=r.elevation_ft,
            predicted_score=r.predicted_score,
            novelty_pct=r.novelty_pct,
            polyline=r.polyline,
            route_segments=r.route_segments,
            city_key=r.city_key,
            gpx_path=getattr(r, "gpx_path", None),
        ))
    await db.flush()

    return results
