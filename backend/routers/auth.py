# LAYER: Router — Authentication
# PURPOSE: Implements the Strava OAuth2 flow. Angular calls /auth/strava/init
#          to get a redirect URL, Strava calls back to /auth/strava/callback,
#          we issue a JWT and redirect Angular to /auth/callback with the token.

import logging
import os
import shutil
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import settings
from backend.core.db import get_db
from backend.core.security import create_access_token, get_current_user
from backend.models.db_models import StravaToken, User
from backend.services import strava_service
from backend.services.model_service import user_model_path

logger = logging.getLogger("routemaker.auth")

router = APIRouter(prefix="/auth", tags=["auth"])


def _auth_error_redirect(reason: str) -> RedirectResponse:
    """Send the browser back to the landing page with a friendly error code."""
    return RedirectResponse(url=f"{settings.frontend_url}/?auth_error={reason}")


@router.get("/strava/init")
async def strava_init():
    """
    Return the Strava authorization URL.
    Angular redirects the browser here to start the login flow.
    """
    return {"url": strava_service.get_authorization_url()}


@router.get("/strava/callback")
async def strava_callback(
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Strava redirects here after the user approves (or declines) the app.
    Steps:
      0. Handle declined auth / missing code gracefully (friendly redirect)
      1. Verify CSRF state
      2. Exchange code for tokens
      3. Enforce whitelist if configured
      4. Upsert user + tokens in DB
      5. Seed user's model from global baseline if first login
      6. Issue JWT and redirect Angular to /auth/callback#token=...
    """
    # User clicked "Cancel" on Strava, or Strava returned an error / no code
    if error or not code:
        logger.warning("Strava OAuth not completed (error=%s, code_present=%s)", error, bool(code))
        return _auth_error_redirect(error or "denied")

    # CSRF protection: state must match one we issued in /strava/init
    if not strava_service.verify_oauth_state(state):
        logger.warning("Strava OAuth state verification failed")
        return _auth_error_redirect("invalid_state")

    # Exchange code for Strava tokens + athlete profile
    token_data = await strava_service.exchange_code(code)
    athlete = token_data.get("athlete", {})
    athlete_id = int(athlete.get("id", 0))

    if athlete_id == 0:
        logger.error("Strava token exchange returned no athlete id")
        return _auth_error_redirect("no_athlete")

    # Whitelist check
    allowed = settings.allowed_ids_list
    if allowed and athlete_id not in allowed:
        logger.warning("Strava login blocked (not whitelisted) athlete_id=%s", athlete_id)
        return _auth_error_redirect("not_authorized")

    # Upsert user
    user = await db.get(User, athlete_id)
    if user is None:
        user = User(
            athlete_id=athlete_id,
            name=f"{athlete.get('firstname', '')} {athlete.get('lastname', '')}".strip(),
            profile_pic_url=athlete.get("profile"),
            created_at=datetime.now(timezone.utc),
        )
        db.add(user)
    else:
        user.name = f"{athlete.get('firstname', '')} {athlete.get('lastname', '')}".strip()
        user.profile_pic_url = athlete.get("profile")

    user.last_login = datetime.now(timezone.utc)

    # Upsert Strava tokens
    token_row = await db.get(StravaToken, athlete_id)
    if token_row is None:
        token_row = StravaToken(athlete_id=athlete_id)
        db.add(token_row)

    token_row.access_token = token_data["access_token"]
    token_row.refresh_token = token_data["refresh_token"]
    token_row.expires_at = token_data["expires_at"]

    await db.flush()

    # Seed user's personal model from global baseline on first login
    model_path = user_model_path(athlete_id)
    if not os.path.exists(model_path):
        if os.path.exists(settings.global_baseline_path):
            shutil.copy2(settings.global_baseline_path, model_path)

    # Issue JWT and redirect to Angular
    jwt = create_access_token(athlete_id)
    logger.info("Strava login success athlete_id=%s", athlete_id)
    redirect_url = f"{settings.frontend_url}/auth/callback#token={jwt}"
    return RedirectResponse(url=redirect_url)


@router.post("/refresh")
async def refresh_session(athlete_id: int = Depends(get_current_user)):
    """
    Re-issue a JWT for an already-authenticated user. Angular calls this on
    app load (after a successful /auth/me) to slide the session window forward,
    so users with the app open aren't logged out every jwt_expire_minutes.
    """
    return {"access_token": create_access_token(athlete_id), "token_type": "bearer"}


@router.get("/me")
async def get_me(
    db: AsyncSession = Depends(get_db),
    athlete_id: int = Depends(get_current_user),
):
    """Return the current user's profile. Angular calls this on app load to verify the JWT."""
    user = await db.get(User, athlete_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "athlete_id": user.athlete_id,
        "name": user.name,
        "profile_pic_url": user.profile_pic_url,
        "model_version": user.model_version,
        "total_ratings": user.total_ratings,
        "model_trained_at": user.model_trained_at,
        "home_lat": user.home_lat,
        "home_lng": user.home_lng,
    }
