# LAYER: Schema — Pydantic Models
# PURPOSE: Request validation and response serialization for all API endpoints.
#          Keeps FastAPI route functions clean — they receive typed objects,
#          not raw dicts. "from_attributes=True" allows building from ORM rows.

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ── Auth ──────────────────────────────────────────────────────────────────────

class TokenResponse(BaseModel):
    """Returned to Angular after successful OAuth. Angular stores the token."""
    access_token: str
    token_type: str = "bearer"
    athlete_id: int
    name: str
    profile_pic_url: Optional[str] = None


# ── Users ─────────────────────────────────────────────────────────────────────

class UserProfile(BaseModel):
    """Public profile info + model training stats for the dashboard."""
    model_config = {"from_attributes": True}

    athlete_id: int
    name: str
    profile_pic_url: Optional[str] = None
    model_version: int
    total_ratings: int
    model_trained_at: Optional[datetime] = None


# ── Rides ─────────────────────────────────────────────────────────────────────

class RideOut(BaseModel):
    """A single Strava ride returned by GET /api/rides."""
    model_config = {"from_attributes": True}

    id: int
    name: str
    date: Optional[datetime] = None
    distance_mi: Optional[float] = None
    elevation_ft: Optional[float] = None
    moving_time_min: Optional[float] = None
    avg_speed_mph: Optional[float] = None
    avg_watts: Optional[float] = None
    suffer_score: Optional[float] = None
    pr_count: Optional[int] = None
    polyline: Optional[str] = None
    start_lat: Optional[float] = None
    start_lng: Optional[float] = None
    user_rating: Optional[int] = None  # Joined from ride_ratings


class SyncSummary(BaseModel):
    """Response for POST /api/rides/sync."""
    new_rides: int
    total_rides: int


# ── Route Generation ──────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    """
    Parameters sent by Angular when the user clicks Generate.
    All optional fields default to the same values as the original CLI scripts.
    """
    route_type: str = Field(default="regular", pattern="^(regular|hilly|historic|novel)$")
    distance_mi: float = Field(default=35.0, ge=5, le=150)
    tolerance: float = Field(default=0.25, ge=0.05, le=0.5)
    start_lat: float = Field(..., ge=-90, le=90)
    start_lng: float = Field(..., ge=-180, le=180)

    # Type-specific options
    hilly_factor: float = Field(default=100.0, ge=0, le=500)
    downtown_radius: float = Field(default=1000.0, ge=100, le=10000)
    novelty_factor: float = Field(default=5.0, ge=0.5, le=20.0)


class HistoricSite(BaseModel):
    name: str
    lat: float
    lng: float


class RouteResult(BaseModel):
    """A single generated route returned to Angular."""
    id: str                          # UUID
    polyline: str                    # Google encoded polyline
    route_segments: Optional[str] = None  # JSON string; only for novel routes
    distance_mi: float
    elevation_ft: float
    predicted_score: float
    novelty_pct: Optional[float] = None  # only for novel routes
    historic_sites: Optional[list[HistoricSite]] = None
    # Downsampled [distance_mi, elevation_ft] points for the elevation chart.
    # Populated on generation; omitted on history (not persisted to the DB).
    elevation_profile: Optional[list[list[float]]] = None
    city_key: str
    route_type: str
    gpx_path: Optional[str] = Field(default=None, exclude=True)


# ── Ratings ───────────────────────────────────────────────────────────────────

class RideRatingRequest(BaseModel):
    """POST /api/ratings/ride body."""
    ride_id: int
    rating: int = Field(..., ge=1, le=10)


class RouteRatingRequest(BaseModel):
    """POST /api/ratings/route body."""
    route_id: str
    rating: int = Field(..., ge=1, le=10)


class RatingStats(BaseModel):
    """
    Returned after any rating action. Angular uses this to update the
    model training progress indicator.
    """
    total_ratings: int
    model_version: int
    ratings_until_next_retrain: int
    model_trained_at: Optional[datetime] = None


# ── Graph ─────────────────────────────────────────────────────────────────────

class GraphStatusResponse(BaseModel):
    """Polled by Angular's map-building-banner while a graph is being built."""
    city_key: str
    status: str        # "ready" | "building" | "not_found"
    progress_pct: int  # 0-100
    node_count: Optional[int] = None
    edge_count: Optional[int] = None


class GraphRequestBody(BaseModel):
    """POST /api/graph/request — user drops a custom pin."""
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)
