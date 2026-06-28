# LAYER: Service — Strava API
# PURPOSE: All HTTP communication with the Strava API lives here. No other
#          file makes direct Strava requests. Handles token refresh transparently
#          before any API call that needs a valid access token.

import secrets
import time
from typing import Optional
from urllib.parse import urlencode

import httpx
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import settings
from backend.models.db_models import StravaToken

STRAVA_AUTH_URL = "https://www.strava.com/oauth/authorize"
STRAVA_TOKEN_URL = "https://www.strava.com/oauth/token"
STRAVA_API_BASE = "https://www.strava.com/api/v3"

# Minimum bike ride distance to import (mirrors the old MIN_MILES constant)
MIN_MILES = 10.0

# In-memory CSRF state store for the OAuth flow: state -> created_at (epoch).
# Single-process deployment assumed (same constraint as the graph cache).
_oauth_states: dict[str, float] = {}
_STATE_TTL = 600  # seconds a pending `state` stays valid (10 minutes)


# ── OAuth ─────────────────────────────────────────────────────────────────────

def get_authorization_url() -> str:
    """
    Build the Strava OAuth authorization URL.
    Angular redirects the user here to start the login flow.
    A random ``state`` is generated and stored so the callback can verify it
    (CSRF protection); all query params are URL-encoded.
    """
    now = time.time()
    # Prune expired pending states so the dict can't grow unbounded.
    for s, ts in list(_oauth_states.items()):
        if now - ts > _STATE_TTL:
            _oauth_states.pop(s, None)

    state = secrets.token_urlsafe(24)
    _oauth_states[state] = now

    params = urlencode({
        "client_id": settings.strava_client_id,
        "response_type": "code",
        "redirect_uri": settings.strava_redirect_uri,
        "approval_prompt": "auto",
        "scope": "read,activity:read_all",
        "state": state,
    })
    return f"{STRAVA_AUTH_URL}?{params}"


def verify_oauth_state(state: Optional[str]) -> bool:
    """
    Validate and consume a ``state`` returned by Strava on the callback.
    Returns True only if it was issued by us and has not expired or been used.
    """
    if not state:
        return False
    created_at = _oauth_states.pop(state, None)
    if created_at is None:
        return False
    return (time.time() - created_at) <= _STATE_TTL


async def exchange_code(code: str) -> dict:
    """
    Exchange the OAuth authorization code for access + refresh tokens.
    Returns the full token payload from Strava (includes athlete profile).
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(STRAVA_TOKEN_URL, data={
            "client_id": settings.strava_client_id,
            "client_secret": settings.strava_client_secret,
            "code": code,
            "grant_type": "authorization_code",
        })
        resp.raise_for_status()
        return resp.json()


async def refresh_token_if_needed(athlete_id: int, db: AsyncSession) -> str:
    """
    Load the stored token for this athlete. If it's expired (or within
    60 seconds of expiry), refresh it via Strava and update the DB row.
    Returns a valid access_token string ready to use.
    """
    result = await db.execute(select(StravaToken).where(StravaToken.athlete_id == athlete_id))
    token_row = result.scalar_one_or_none()

    if token_row is None:
        raise ValueError(f"No Strava token found for athlete {athlete_id}")

    # Refresh if expired or expiring in the next 60 seconds
    if token_row.expires_at - time.time() < 60:
        async with httpx.AsyncClient() as client:
            resp = await client.post(STRAVA_TOKEN_URL, data={
                "client_id": settings.strava_client_id,
                "client_secret": settings.strava_client_secret,
                "grant_type": "refresh_token",
                "refresh_token": token_row.refresh_token,
            })
            resp.raise_for_status()
            data = resp.json()

        token_row.access_token = data["access_token"]
        token_row.refresh_token = data["refresh_token"]
        token_row.expires_at = data["expires_at"]
        db.add(token_row)
        await db.flush()

    return token_row.access_token


# ── Activity Fetching ─────────────────────────────────────────────────────────

async def fetch_activities(access_token: str, max_rides: int = 200) -> list[dict]:
    """
    Page through /athlete/activities and collect all cycling activities.
    Filters to Ride / MountainBikeRide / GravelRide only.
    Stops when an empty page is returned or max_rides is reached.
    """
    headers = {"Authorization": f"Bearer {access_token}"}
    rides = []
    page = 1

    async with httpx.AsyncClient() as client:
        while len(rides) < max_rides:
            resp = await client.get(
                f"{STRAVA_API_BASE}/athlete/activities",
                headers=headers,
                params={"per_page": 50, "page": page},
            )
            resp.raise_for_status()
            batch = resp.json()

            if not batch:
                break

            bike_rides = [
                a for a in batch
                if a.get("sport_type") in ("Ride", "MountainBikeRide", "GravelRide")
            ]
            rides.extend(bike_rides)
            page += 1

    return rides[:max_rides]


def activities_to_df(activities: list[dict]) -> pd.DataFrame:
    """
    Convert raw Strava activity JSON into a clean DataFrame.
    Column names match the original my_rides.csv schema so downstream
    ML code (feature engineering) requires no changes.
    """
    rows = []
    for a in activities:
        distance_mi = round(a.get("distance", 0) * 0.000621371, 1)
        if distance_mi < MIN_MILES:
            continue
        rows.append({
            "id": a["id"],
            "name": a["name"],
            "date": a.get("start_date_local"),
            "distance_mi": distance_mi,
            "elevation_ft": round(a.get("total_elevation_gain", 0) * 3.28084),
            "moving_time_min": round(a.get("moving_time", 0) / 60, 1),
            "avg_speed_mph": round(a.get("average_speed", 0) * 2.23694, 1),
            "avg_watts": a.get("average_watts"),
            "suffer_score": a.get("suffer_score"),
            "pr_count": a.get("pr_count", 0),
            "polyline": a.get("map", {}).get("summary_polyline"),
            "start_lat": (a.get("start_latlng") or [None, None])[0],
            "start_lng": (a.get("start_latlng") or [None, None])[1],
        })
    return pd.DataFrame(rows)
