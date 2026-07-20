# LAYER: Database Models
# PURPOSE: SQLAlchemy ORM definitions for all tables. These map directly to the
#          schema described in the implementation plan. One class = one table.
#          All FKs use athlete_id (Strava's integer ID) as the user key.

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger, Boolean, DateTime, Float, ForeignKey,
    Integer, SmallInteger, String, Text, func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.core.db import Base


class User(Base):
    """
    One row per Strava athlete. The athlete_id IS the primary key —
    no separate internal UUID needed since Strava IDs are stable integers.
    """
    __tablename__ = "users"

    athlete_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    profile_pic_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Coverage feature: user's saved "home base" — center of the coverage area.
    home_lat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    home_lng: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # ML model tracking
    model_version: Mapped[int] = mapped_column(Integer, default=0)
    total_ratings: Mapped[int] = mapped_column(Integer, default=0)
    model_trained_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    strava_token: Mapped[Optional["StravaToken"]] = relationship(back_populates="user", uselist=False)
    rides: Mapped[list["Ride"]] = relationship(back_populates="user")
    ride_ratings: Mapped[list["RideRating"]] = relationship(back_populates="user")
    generated_routes: Mapped[list["GeneratedRoute"]] = relationship(back_populates="user")
    route_feedback: Mapped[list["RouteFeedback"]] = relationship(back_populates="user")


class StravaToken(Base):
    """
    Stores OAuth tokens for each user. One row per user.
    access_token expires (checked before each Strava API call);
    refresh_token is used to get a new one.
    """
    __tablename__ = "strava_tokens"

    athlete_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.athlete_id", ondelete="CASCADE"), primary_key=True
    )
    access_token: Mapped[str] = mapped_column(Text)
    refresh_token: Mapped[str] = mapped_column(Text)
    expires_at: Mapped[int] = mapped_column(BigInteger)  # Unix epoch seconds

    user: Mapped["User"] = relationship(back_populates="strava_token")


class Ride(Base):
    """
    Strava activities cached locally. Synced from Strava on demand.
    The id is Strava's own activity ID (stable, globally unique integer).
    """
    __tablename__ = "rides"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)  # Strava activity ID
    athlete_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.athlete_id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(512))
    # "ride" (cycling) or "walk" (foot: walk/run/hike). Drives coverage filtering;
    # only "ride" rows feed the cycling ML model.
    activity_type: Mapped[str] = mapped_column(String(16), default="ride", server_default="ride")
    date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    distance_mi: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    elevation_ft: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    moving_time_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_speed_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_watts: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    suffer_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pr_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    polyline: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    start_lat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    start_lng: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    synced_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="rides")
    rating: Mapped[Optional["RideRating"]] = relationship(back_populates="ride", uselist=False)


class RideRating(Base):
    """
    User's 1-10 rating for a real Strava ride. One rating per (ride, athlete) pair.
    These ratings are the primary training signal for the per-user ML model.
    """
    __tablename__ = "ride_ratings"

    ride_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("rides.id", ondelete="CASCADE"), primary_key=True)
    athlete_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.athlete_id", ondelete="CASCADE"), primary_key=True)
    rating: Mapped[int] = mapped_column(SmallInteger)  # 1-10
    rated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    ride: Mapped["Ride"] = relationship(back_populates="rating")
    user: Mapped["User"] = relationship(back_populates="ride_ratings")


class GeneratedRoute(Base):
    """
    Routes produced by the system's generation algorithm. Stored so users can
    download GPX files later, view history, and rate them after riding.
    id is a UUID string (generated in Python before insert).
    """
    __tablename__ = "generated_routes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # UUID
    athlete_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.athlete_id", ondelete="CASCADE"))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    route_type: Mapped[str] = mapped_column(String(32))  # regular/hilly/historic/novel
    distance_mi: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    elevation_ft: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    predicted_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    novelty_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    polyline: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    route_segments: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON string
    city_key: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)  # "41.65_-91.53"
    gpx_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # server-side file path

    user: Mapped["User"] = relationship(back_populates="generated_routes")
    feedback: Mapped[Optional["RouteFeedback"]] = relationship(back_populates="route", uselist=False)


class RouteFeedback(Base):
    """
    User's 1-10 rating for a generated route, submitted after actually riding it.
    Combined with RideRatings to train the per-user ML model.
    """
    __tablename__ = "route_feedback"

    route_id: Mapped[str] = mapped_column(String(36), ForeignKey("generated_routes.id", ondelete="CASCADE"), primary_key=True)
    athlete_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.athlete_id", ondelete="CASCADE"), primary_key=True)
    rating: Mapped[int] = mapped_column(SmallInteger)  # 1-10
    rated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    route: Mapped["GeneratedRoute"] = relationship(back_populates="feedback")
    user: Mapped["User"] = relationship(back_populates="route_feedback")
