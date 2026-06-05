# LAYER: Router — Ratings and Feedback
# PURPOSE: Stores user ratings for real rides and generated routes.
#          After each save, increments total_ratings and triggers model
#          retraining as a background task when the threshold is hit.

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.db import get_db
from backend.core.security import get_current_user
from backend.models.db_models import GeneratedRoute, Ride, RideRating, RouteFeedback, User
from backend.models.schemas import RideRatingRequest, RouteRatingRequest, RatingStats
from backend.services import model_service

router = APIRouter(prefix="/api/ratings", tags=["ratings"])

RETRAIN_THRESHOLD = model_service.RETRAIN_THRESHOLD


async def _increment_and_maybe_retrain(
    athlete_id: int,
    db: AsyncSession,
    background_tasks: BackgroundTasks,
):
    """Helper: increment total_ratings and queue a retrain if threshold is hit."""
    user = await db.get(User, athlete_id)
    if user:
        user.total_ratings += 1
        db.add(user)
        await db.flush()
        if user.total_ratings % RETRAIN_THRESHOLD == 0:
            # Run retraining in the background — doesn't block the response
            background_tasks.add_task(model_service.retrain_model, athlete_id, db)

    return user


async def _get_stats(athlete_id: int, db: AsyncSession) -> RatingStats:
    user = await db.get(User, athlete_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    until_retrain = RETRAIN_THRESHOLD - (user.total_ratings % RETRAIN_THRESHOLD)
    return RatingStats(
        total_ratings=user.total_ratings,
        model_version=user.model_version,
        ratings_until_next_retrain=until_retrain if until_retrain != RETRAIN_THRESHOLD else 0,
        model_trained_at=user.model_trained_at,
    )


@router.post("/ride", response_model=RatingStats)
async def rate_ride(
    body: RideRatingRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    athlete_id: int = Depends(get_current_user),
):
    """
    Save or update a 1-10 rating for a real Strava ride.
    If this rating pushes total_ratings to a multiple of RETRAIN_THRESHOLD,
    model retraining is queued as a background task.
    """
    # Verify the ride belongs to this athlete
    ride = await db.get(Ride, body.ride_id)
    if ride is None or ride.athlete_id != athlete_id:
        raise HTTPException(status_code=404, detail="Ride not found")

    # Upsert rating
    existing = await db.get(RideRating, (body.ride_id, athlete_id))
    is_new = existing is None
    if is_new:
        rating_row = RideRating(ride_id=body.ride_id, athlete_id=athlete_id, rating=body.rating)
        db.add(rating_row)
    else:
        existing.rating = body.rating

    await db.flush()

    # Only increment counter for new ratings (not updates)
    if is_new:
        await _increment_and_maybe_retrain(athlete_id, db, background_tasks)

    return await _get_stats(athlete_id, db)


@router.post("/route", response_model=RatingStats)
async def rate_generated_route(
    body: RouteRatingRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    athlete_id: int = Depends(get_current_user),
):
    """
    Save or update a 1-10 rating for a generated route (after actually riding it).
    Triggers model retraining on threshold — same logic as ride ratings.
    """
    route = await db.get(GeneratedRoute, body.route_id)
    if route is None or route.athlete_id != athlete_id:
        raise HTTPException(status_code=404, detail="Generated route not found")

    existing = await db.get(RouteFeedback, (body.route_id, athlete_id))
    is_new = existing is None
    if is_new:
        fb = RouteFeedback(route_id=body.route_id, athlete_id=athlete_id, rating=body.rating)
        db.add(fb)
    else:
        existing.rating = body.rating

    await db.flush()

    if is_new:
        await _increment_and_maybe_retrain(athlete_id, db, background_tasks)

    return await _get_stats(athlete_id, db)


@router.get("/stats", response_model=RatingStats)
async def get_rating_stats(
    db: AsyncSession = Depends(get_db),
    athlete_id: int = Depends(get_current_user),
):
    """Return model training progress stats for the dashboard."""
    return await _get_stats(athlete_id, db)
