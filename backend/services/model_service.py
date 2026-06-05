# LAYER: Service — Per-User ML Model
# PURPOSE: Manages loading, caching, and retraining of per-user
#          GradientBoostingRegressor models. Each user has their own .pkl file.
#          New users start with a copy of the global baseline. Retraining is
#          triggered automatically after every RETRAIN_THRESHOLD new ratings.
#          The model architecture (GradientBoostingRegressor + StandardScaler)
#          is unchanged from the original train_model.py.

import os
import pickle
import shutil
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import settings
from backend.models.db_models import GeneratedRoute, Ride, RideRating, RouteFeedback, User

# Minimum new ratings before auto-retraining a user's model
RETRAIN_THRESHOLD = 10

# In-memory model cache: athlete_id → { model, scaler, feature_names }
_model_cache: dict[int, dict] = {}


# ── Paths ─────────────────────────────────────────────────────────────────────

def user_model_path(athlete_id: int) -> str:
    os.makedirs(settings.models_dir, exist_ok=True)
    return os.path.join(settings.models_dir, f"{athlete_id}.pkl")


# ── Load ──────────────────────────────────────────────────────────────────────

def get_model(athlete_id: int) -> dict:
    """
    Return the model bundle for this athlete. Load order:
      1. In-memory cache (fastest)
      2. User's .pkl on disk
      3. Global baseline (copied to user's path for future loads)

    Returns a dict with keys: model, scaler, feature_names
    """
    if athlete_id in _model_cache:
        return _model_cache[athlete_id]

    user_path = user_model_path(athlete_id)

    if not os.path.exists(user_path):
        # First time this user generates a route — seed from global baseline
        if not os.path.exists(settings.global_baseline_path):
            raise FileNotFoundError(
                f"Global baseline model not found at {settings.global_baseline_path}. "
                "Copy data/route_model.pkl to backend/data/models/global_baseline.pkl"
            )
        shutil.copy2(settings.global_baseline_path, user_path)

    with open(user_path, "rb") as f:
        bundle = pickle.load(f)

    _model_cache[athlete_id] = bundle
    return bundle


def predict_score(athlete_id: int, route_features: dict) -> float:
    """
    Predict a 1-10 enjoyment score for a candidate route using the user's model.
    Feature engineering mirrors train_model.py exactly so the model receives
    the same input shape it was trained on.
    """
    bundle = get_model(athlete_id)
    model: GradientBoostingRegressor = bundle["model"]
    scaler: StandardScaler = bundle["scaler"]
    feature_names: list[str] = bundle["feature_names"]

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

    features = features[feature_names]
    X_scaled = scaler.transform(features)
    score = model.predict(X_scaled)[0]
    return round(float(np.clip(score, 1, 10)), 2)


# ── Retrain ───────────────────────────────────────────────────────────────────

async def retrain_model(athlete_id: int, db: AsyncSession) -> None:
    """
    Retrain the user's personal GradientBoostingRegressor on all their rated
    rides and rated generated routes combined. Saves the new model to disk
    and evicts the cached version so the next predict_score() picks it up.

    Called as a FastAPI BackgroundTask when total_ratings % RETRAIN_THRESHOLD == 0.
    """
    # ── Gather ride ratings ────────────────────────────────────────────────────
    ride_rows = await db.execute(
        select(Ride, RideRating.rating)
        .join(RideRating, (RideRating.ride_id == Ride.id) & (RideRating.athlete_id == athlete_id))
        .where(Ride.athlete_id == athlete_id)
    )
    ride_data = [
        {
            "distance_mi":     r.Ride.distance_mi or 0,
            "elevation_ft":    r.Ride.elevation_ft or 0,
            "avg_speed_mph":   r.Ride.avg_speed_mph or 0,
            "moving_time_min": r.Ride.moving_time_min or 0,
            "avg_watts":       r.Ride.avg_watts or 0,
            "suffer_score":    r.Ride.suffer_score or 0,
            "pr_count":        r.Ride.pr_count or 0,
            "rating":          r.rating,
        }
        for r in ride_rows.all()
    ]

    # ── Gather generated route feedback ───────────────────────────────────────
    fb_rows = await db.execute(
        select(GeneratedRoute, RouteFeedback.rating)
        .join(RouteFeedback,
              (RouteFeedback.route_id == GeneratedRoute.id) &
              (RouteFeedback.athlete_id == athlete_id))
        .where(GeneratedRoute.athlete_id == athlete_id)
    )
    fb_data = [
        {
            "distance_mi":     r.GeneratedRoute.distance_mi or 0,
            "elevation_ft":    r.GeneratedRoute.elevation_ft or 0,
            "avg_speed_mph":   15,  # estimated
            "moving_time_min": (r.GeneratedRoute.distance_mi or 0) / 15 * 60,
            "avg_watts":       0,
            "suffer_score":    0,
            "pr_count":        0,
            "rating":          r.rating,
        }
        for r in fb_rows.all()
    ]

    all_data = ride_data + fb_data
    if len(all_data) < 5:
        print(f"[model_service] Not enough data to retrain athlete {athlete_id} ({len(all_data)} samples)")
        return

    df = pd.DataFrame(all_data)

    # ── Feature engineering (mirrors train_model.py) ──────────────────────────
    X = pd.DataFrame()
    X["distance_mi"]     = df["distance_mi"]
    X["elevation_ft"]    = df["elevation_ft"].fillna(0)
    X["avg_speed_mph"]   = df["avg_speed_mph"].fillna(0)
    X["moving_time_min"] = df["moving_time_min"].fillna(0)
    X["elev_per_mile"]   = X["elevation_ft"] / X["distance_mi"].replace(0, 1)
    X["distance_sq"]     = X["distance_mi"] ** 2
    X["elevation_sq"]    = X["elevation_ft"] ** 2
    X["suffer_score"]    = df["suffer_score"].fillna(0)
    X["avg_watts"]       = df["avg_watts"].fillna(0)
    X["pr_count"]        = df["pr_count"].fillna(0)
    y = df["rating"].astype(float)

    # ── Train ─────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
    )
    model.fit(X_scaled, y)

    # ── Save ──────────────────────────────────────────────────────────────────
    bundle = {"model": model, "scaler": scaler, "feature_names": list(X.columns)}
    with open(user_model_path(athlete_id), "wb") as f:
        pickle.dump(bundle, f)

    # Evict from cache so the next call reloads the new model
    _model_cache.pop(athlete_id, None)

    # Update the user's model metadata in the DB
    user_row = await db.get(User, athlete_id)
    if user_row:
        user_row.model_version += 1
        user_row.model_trained_at = datetime.now(timezone.utc)
        db.add(user_row)
        await db.flush()

    print(f"[model_service] Retrained model for athlete {athlete_id} on {len(all_data)} samples")


async def should_retrain(athlete_id: int, db: AsyncSession) -> bool:
    """Check if the user has hit the retrain threshold since last model save."""
    user_row = await db.get(User, athlete_id)
    if user_row is None:
        return False
    return user_row.total_ratings > 0 and user_row.total_ratings % RETRAIN_THRESHOLD == 0
