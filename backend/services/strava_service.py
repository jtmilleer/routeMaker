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

# Every activity above a tiny floor is imported so coverage sees short trips too.
MIN_ACTIVITY_MILES = 0.3
# Rides shorter than this are coverage-only: hidden from rating so they never
# reach the rating model (mirrors the old MIN_MILES constant in src/).
MIN_RATED_MILES = 10.0

# Strava sport_type -> our internal activity_type. Anything not listed is skipped.
RIDE_SPORT_TYPES = {"Ride", "MountainBikeRide", "GravelRide", "EBikeRide", "VirtualRide"}
WALK_SPORT_TYPES = {"Walk", "Run", "Hike", "TrailRun", "Snowshoe"}


def _activity_category(sport_type: Optional[str]) -> Optional[str]:
    """Map a Strava sport_type to "ride", "walk", or None (skip)."""
    if sport_type in RIDE_SPORT_TYPES:
        return "ride"
    if sport_type in WALK_SPORT_TYPES:
        return "walk"
    return None

# In-memory CSRF state store for the OAuth flow: state -> created_at (epoch).
# Single-process deployment assumed (same constraint as the graph cache).
_oauth_states: dict[str, float] = {}
_STATE_TTL = 600  # seconds a pending `state` stays valid (10 minutes)


# OAuth

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


# Activity Fetching

async def fetch_activities(access_token: str) -> list[dict]:
    """
    Page through all of /athlete/activities and collect cycling AND foot activities
    (walk/run/hike). Other sport types are skipped. Stops at the first empty page.
    """
    headers = {"Authorization": f"Bearer {access_token}"}
    rides = []
    page = 1

    async with httpx.AsyncClient() as client:
        while True:
            resp = await client.get(
                f"{STRAVA_API_BASE}/athlete/activities",
                headers=headers,
                params={"per_page": 200, "page": page},
            )
            resp.raise_for_status()
            batch = resp.json()

            if not batch:
                break

            kept = [a for a in batch if _activity_category(a.get("sport_type"))]
            rides.extend(kept)
            page += 1

    return rides


# Full GPS tracks (streams API) + rate limiting

# Headroom left under Strava's read limits so an interactive sync still works
# while a track backfill is running.
SHORT_LIMIT_HEADROOM = 10
DAILY_LIMIT_HEADROOM = 50


class StravaRateLimited(Exception):
    """Strava returned 429. `retry_after` is how long to wait, in seconds."""

    def __init__(self, retry_after: float):
        super().__init__(f"Strava rate limit hit; retry in {retry_after:.0f}s")
        self.retry_after = retry_after


def _seconds_until_window_reset(daily: bool) -> float:
    """Strava's short window resets on the quarter hour, the daily one at UTC
    midnight. Adds a small buffer past the boundary."""
    now = time.time()
    period = 86400 if daily else 900
    return period - (now % period) + 5


def _rate_limit_wait(resp: httpx.Response) -> float:
    """Seconds to pause before the next request, based on the usage headers, so
    we back off before hitting the limit rather than after."""
    limit = resp.headers.get("X-ReadRateLimit-Limit") or resp.headers.get("X-RateLimit-Limit")
    usage = resp.headers.get("X-ReadRateLimit-Usage") or resp.headers.get("X-RateLimit-Usage")
    if not limit or not usage:
        return 0.0
    try:
        short_lim, daily_lim = (int(x) for x in limit.split(",")[:2])
        short_use, daily_use = (int(x) for x in usage.split(",")[:2])
    except ValueError:
        return 0.0
    if daily_use >= daily_lim - DAILY_LIMIT_HEADROOM:
        return _seconds_until_window_reset(daily=True)
    if short_use >= short_lim - SHORT_LIMIT_HEADROOM:
        return _seconds_until_window_reset(daily=False)
    return 0.0


async def fetch_track(access_token: str, activity_id: int) -> tuple[list[tuple[float, float]], float]:
    """
    Fetch an activity's full-resolution GPS track via the streams API.
    Returns (points, wait_s): points is [(lat, lng), ...] — empty when the activity
    has no GPS (indoor/manual) or no longer exists — and wait_s is how long the
    caller should pause before the next request to stay under the rate limit.
    Raises StravaRateLimited on 429.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{STRAVA_API_BASE}/activities/{activity_id}/streams",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"keys": "latlng", "key_by_type": "true"},
        )
    if resp.status_code == 429:
        daily = _rate_limit_wait(resp) > 900
        raise StravaRateLimited(_seconds_until_window_reset(daily=daily))
    wait_s = _rate_limit_wait(resp)
    if resp.status_code == 404:
        return [], wait_s
    resp.raise_for_status()
    latlng = (resp.json().get("latlng") or {}).get("data") or []
    return [(float(p[0]), float(p[1])) for p in latlng], wait_s


def activities_to_df(activities: list[dict]) -> pd.DataFrame:
    """
    Convert raw Strava activity JSON into a clean DataFrame.
    Column names match the original my_rides.csv schema so downstream
    ML code (feature engineering) requires no changes.
    """
    rows = []
    for a in activities:
        category = _activity_category(a.get("sport_type"))
        if category is None:
            continue
        distance_mi = round(a.get("distance", 0) * 0.000621371, 1)
        if distance_mi < MIN_ACTIVITY_MILES:
            continue
        rows.append({
            "id": a["id"],
            "name": a["name"],
            "activity_type": category,
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
