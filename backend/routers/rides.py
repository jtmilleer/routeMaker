# LAYER: Router — Ride Management
# PURPOSE: Syncing rides from Strava and listing them with user ratings merged in.

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.db import get_db
from backend.core.security import get_current_user
from backend.models.db_models import Ride, RideRating
from backend.models.schemas import RideOut, SyncSummary
from backend.services import strava_service

router = APIRouter(prefix="/api/rides", tags=["rides"])


@router.post("/sync", response_model=SyncSummary)
async def sync_rides(
    db: AsyncSession = Depends(get_db),
    athlete_id: int = Depends(get_current_user),
):
    """
    Fetch the latest cycling activities from Strava and upsert into the local DB.
    Returns a count of new rides added and the total ride count.
    Refreshes the access token if expired before making any Strava calls.
    """
    access_token = await strava_service.refresh_token_if_needed(athlete_id, db)
    activities = await strava_service.fetch_activities(access_token)
    df = strava_service.activities_to_df(activities)

    new_count = 0
    for _, row in df.iterrows():
        existing = await db.get(Ride, int(row["id"]))
        if existing is None:
            ride = Ride(
                id=int(row["id"]),
                athlete_id=athlete_id,
                name=row["name"],
                date=row.get("date"),
                distance_mi=row.get("distance_mi"),
                elevation_ft=row.get("elevation_ft"),
                moving_time_min=row.get("moving_time_min"),
                avg_speed_mph=row.get("avg_speed_mph"),
                avg_watts=row.get("avg_watts") if row.get("avg_watts") == row.get("avg_watts") else None,
                suffer_score=row.get("suffer_score") if row.get("suffer_score") == row.get("suffer_score") else None,
                pr_count=int(row["pr_count"]) if row.get("pr_count") == row.get("pr_count") else 0,
                polyline=row.get("polyline"),
                start_lat=row.get("start_lat"),
                start_lng=row.get("start_lng"),
            )
            db.add(ride)
            new_count += 1

    await db.flush()

    total_result = await db.execute(
        select(Ride).where(Ride.athlete_id == athlete_id)
    )
    total = len(total_result.scalars().all())

    return SyncSummary(new_rides=new_count, total_rides=total)


@router.get("", response_model=list[RideOut])
async def list_rides(
    limit: int = 200,
    db: AsyncSession = Depends(get_db),
    athlete_id: int = Depends(get_current_user),
):
    """
    Return all cached rides for this athlete, with user rating merged in.
    Sorted by date descending (most recent first).
    """
    rides_result = await db.execute(
        select(Ride).where(Ride.athlete_id == athlete_id)
        .order_by(Ride.date.desc())
        .limit(limit)
    )
    rides = rides_result.scalars().all()

    # Load all ratings for this athlete in one query
    ratings_result = await db.execute(
        select(RideRating).where(RideRating.athlete_id == athlete_id)
    )
    ratings_map = {r.ride_id: r.rating for r in ratings_result.scalars().all()}

    output = []
    for r in rides:
        ride_out = RideOut.model_validate(r)
        ride_out.user_rating = ratings_map.get(r.id)
        output.append(ride_out)

    return output
