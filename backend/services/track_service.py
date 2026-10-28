# LAYER: Service — GPS Track Backfill
# PURPOSE: Fetches each activity's full-resolution GPS track from Strava's streams
#          API into Ride.track_polyline, for accurate street-coverage matching.
#          Strava's summary polyline (stored at sync) is simplified enough to cut
#          corners and skip whole blocks. One request per activity against a read
#          limit of ~100/15min and ~1000/day, so this runs as a long-lived
#          background task per athlete that paces itself and resumes on restart.

import asyncio
import logging
from typing import Optional

import httpx
import numpy as np
import osmnx as ox
import polyline as pl
from sqlalchemy import func, select

from backend.core.db import AsyncSessionFactory
from backend.models.db_models import Ride
from backend.services import strava_service

logger = logging.getLogger("routemaker.tracks")

BATCH_SIZE = 20
# Drop points closer than this to the previous kept point — 1Hz GPS while stopped
# or slow adds nothing to matching and bloats storage.
MIN_POINT_SPACING_M = 5.0
# Pause after an unexpected Strava/network error before trying again.
ERROR_BACKOFF_S = 60

_running: dict[int, asyncio.Task] = {}


def _thin(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Keep points at least MIN_POINT_SPACING_M from the last kept one."""
    if len(points) < 3:
        return points
    arr = np.asarray(points, dtype=float)
    kept = [0]
    last = arr[0]
    for i in range(1, len(arr) - 1):
        d = ox.distance.great_circle(last[0], last[1], arr[i, 0], arr[i, 1])
        if d >= MIN_POINT_SPACING_M:
            kept.append(i)
            last = arr[i]
    kept.append(len(arr) - 1)
    return [points[i] for i in kept]


async def pending_count(db, athlete_id: int) -> int:
    """How many of this athlete's activities still lack a fetched track."""
    result = await db.execute(
        select(func.count()).select_from(Ride)
        .where(Ride.athlete_id == athlete_id, Ride.track_polyline.is_(None))
    )
    return int(result.scalar_one())


def start_backfill(athlete_id: int) -> None:
    """Start the backfill task for this athlete unless one is already running."""
    task = _running.get(athlete_id)
    if task is not None and not task.done():
        return
    _running[athlete_id] = asyncio.create_task(_backfill(athlete_id))


async def resume_all_backfills() -> None:
    """Called on startup: restart backfills for anyone with unfetched tracks."""
    async with AsyncSessionFactory() as db:
        result = await db.execute(
            select(Ride.athlete_id).where(Ride.track_polyline.is_(None)).distinct()
        )
        athlete_ids = list(result.scalars().all())
    for athlete_id in athlete_ids:
        start_backfill(athlete_id)


async def _next_batch(athlete_id: int) -> tuple[list[int], Optional[str]]:
    """Most recent activities without a track, plus a fresh access token."""
    async with AsyncSessionFactory() as db:
        result = await db.execute(
            select(Ride.id)
            .where(Ride.athlete_id == athlete_id, Ride.track_polyline.is_(None))
            .order_by(Ride.date.desc())
            .limit(BATCH_SIZE)
        )
        ids = list(result.scalars().all())
        if not ids:
            return [], None
        token = await strava_service.refresh_token_if_needed(athlete_id, db)
        await db.commit()
    return ids, token


async def _save_track(ride_id: int, points: list[tuple[float, float]]) -> None:
    async with AsyncSessionFactory() as db:
        ride = await db.get(Ride, ride_id)
        if ride is None:
            return
        ride.track_polyline = pl.encode(_thin(points)) if points else ""
        await db.commit()


async def _backfill(athlete_id: int) -> None:
    fetched = 0
    try:
        while True:
            try:
                ids, token = await _next_batch(athlete_id)
            except (httpx.HTTPError, ValueError) as e:
                logger.warning("Track backfill for %s: token refresh failed (%s); retrying in %ds",
                               athlete_id, e, ERROR_BACKOFF_S)
                await asyncio.sleep(ERROR_BACKOFF_S)
                continue
            if not ids:
                break
            for ride_id in ids:
                try:
                    points, wait_s = await strava_service.fetch_track(token, ride_id)
                except strava_service.StravaRateLimited as e:
                    logger.info("Track backfill for %s: rate limited, pausing %.0fs",
                                athlete_id, e.retry_after)
                    await asyncio.sleep(e.retry_after)
                    break  # re-fetch the batch (and token) after the pause
                except httpx.HTTPError as e:
                    logger.warning("Track backfill for %s: activity %s failed (%s); retrying in %ds",
                                   athlete_id, ride_id, e, ERROR_BACKOFF_S)
                    await asyncio.sleep(ERROR_BACKOFF_S)
                    break
                await _save_track(ride_id, points)
                fetched += 1
                if fetched % 25 == 0:
                    logger.info("Track backfill for %s: %d tracks fetched", athlete_id, fetched)
                if wait_s:
                    logger.info("Track backfill for %s: near rate limit, pausing %.0fs",
                                athlete_id, wait_s)
                    await asyncio.sleep(wait_s)
        logger.info("Track backfill for %s complete (%d fetched)", athlete_id, fetched)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Track backfill for %s crashed", athlete_id)
    finally:
        _running.pop(athlete_id, None)
