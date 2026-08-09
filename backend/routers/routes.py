# LAYER: Router — Route Generation
# PURPOSE: Handles all route generation requests and GPX file downloads.
#          Generation is run in a thread pool (run_in_executor) so the async
#          event loop is never blocked by the CPU-intensive A* computation.

import io
import os
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.db import get_db
from backend.core.security import get_current_user
from backend.models.db_models import GeneratedRoute, Ride, RouteFeedback
from backend.models.schemas import GenerateRequest, RouteResult
from backend.services import graph_service, model_service, route_service, strava_service

router = APIRouter(prefix="/api/routes", tags=["routes"])
_executor = ThreadPoolExecutor(max_workers=4)


@router.post("/generate", response_model=list[RouteResult])
async def generate(
    params: GenerateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    athlete_id: int = Depends(get_current_user),
):
    """
    Generate up to 5 candidate routes for the given parameters.
    Steps:
      1. Verify the graph for this city is ready
      2. Load the user's rides (for novel route novelty scoring)
      3. Run generation in a thread pool (CPU-bound, ~30-60s)
      4. Persist results to DB
      5. Return RouteResult list to Angular
    """
    import asyncio

    # Check graph availability
    city_k = graph_service.city_key(params.start_lat, params.start_lng)
    status = graph_service.get_graph_status(city_k)
    if status["status"] != "ready":
        raise HTTPException(
            status_code=503,
            detail=f"Map for this area is not ready yet (status: {status['status']}). "
                   "Please wait and try again."
        )

    G = graph_service.get_graph_sync(params.start_lat, params.start_lng)
    if G is None:
        raise HTTPException(status_code=503, detail="Graph unavailable")

    # Load user's rides as DataFrame (needed for novel route type). Only actual
    # bike rides drive novelty — walks/runs are coverage-only and excluded.
    rides_result = await db.execute(
        select(Ride).where(
            Ride.athlete_id == athlete_id,
            Ride.activity_type == "ride",
        )
    )
    rides = rides_result.scalars().all()
    import pandas as pd
    rides_df = pd.DataFrame([{
        "polyline": r.polyline,
        "distance_mi": r.distance_mi,
    } for r in rides if r.polyline])

    # Run generation in thread pool (blocks up to ~60s)
    loop = asyncio.get_event_loop()
    results: list[RouteResult] = await loop.run_in_executor(
        _executor,
        route_service.generate_routes,
        G, athlete_id, params, rides_df,
    )

    if not results:
        raise HTTPException(status_code=500, detail="Route generation failed — no valid routes found")

    # Persist to DB
    for r in results:
        row = GeneratedRoute(
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
        )
        db.add(row)
    await db.flush()

    return results


@router.get("/gpx/{route_id}")
async def download_gpx(
    route_id: str,
    db: AsyncSession = Depends(get_db),
    athlete_id: int = Depends(get_current_user),
):
    """
    Stream a GPX file download for a previously generated route.
    Only the athlete who generated the route can download it.
    """
    row = await db.get(GeneratedRoute, route_id)
    if row is None or row.athlete_id != athlete_id:
        raise HTTPException(status_code=404, detail="Route not found")

    if not row.gpx_path or not os.path.exists(row.gpx_path):
        raise HTTPException(status_code=404, detail="GPX file not found on server")

    def iter_file():
        with open(row.gpx_path, "rb") as f:
            yield from f

    filename = f"route_{route_id[:8]}.gpx"
    return StreamingResponse(
        iter_file(),
        media_type="application/gpx+xml",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/history", response_model=list[RouteResult])
async def get_history(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
    athlete_id: int = Depends(get_current_user),
):
    """Return the user's previously generated routes (most recent first), with
    each route's rating merged in (null if not yet rated)."""
    result = await db.execute(
        select(GeneratedRoute)
        .where(GeneratedRoute.athlete_id == athlete_id)
        .order_by(GeneratedRoute.created_at.desc())
        .limit(limit)
    )
    rows = result.scalars().all()

    feedback_result = await db.execute(
        select(RouteFeedback).where(RouteFeedback.athlete_id == athlete_id)
    )
    ratings_map = {f.route_id: f.rating for f in feedback_result.scalars().all()}

    return [
        RouteResult(
            id=r.id,
            polyline=r.polyline or "",
            route_segments=r.route_segments,
            distance_mi=r.distance_mi or 0,
            elevation_ft=r.elevation_ft or 0,
            predicted_score=r.predicted_score or 0,
            novelty_pct=r.novelty_pct,
            city_key=r.city_key or "",
            route_type=r.route_type,
            user_rating=ratings_map.get(r.id),
        )
        for r in rows
    ]
