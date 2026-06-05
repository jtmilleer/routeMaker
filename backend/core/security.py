# LAYER: Security
# PURPOSE: JWT creation and validation. Provides get_current_user() as a FastAPI
#          dependency that any protected route can inject to identify the caller.
#          On invalid/expired tokens, raises HTTP 401 automatically.

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

from backend.core.config import settings

bearer_scheme = HTTPBearer()


def create_access_token(athlete_id: int) -> str:
    """Encode a JWT containing the athlete's Strava ID and an expiry timestamp."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": str(athlete_id), "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> Optional[int]:
    """Decode and validate a JWT. Returns athlete_id on success, None on failure."""
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        athlete_id = payload.get("sub")
        return int(athlete_id) if athlete_id else None
    except JWTError:
        return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> int:
    """
    FastAPI dependency. Extracts and validates the Bearer JWT from the
    Authorization header. Returns the athlete_id or raises HTTP 401.

    Usage in a route:
        @router.get("/protected")
        async def my_route(athlete_id: int = Depends(get_current_user)):
            ...
    """
    athlete_id = decode_token(credentials.credentials)
    if athlete_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return athlete_id
