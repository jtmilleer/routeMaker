# LAYER: Configuration
# PURPOSE: Single source of truth for all environment-driven settings.
#          Pydantic Settings reads from environment variables or a .env file.
#          Import `settings` anywhere in the app — never os.getenv() directly.

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Database
    database_url: str = "sqlite+aiosqlite:///./routemaker.db"

    # JWT
    jwt_secret: str = "change_me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # Strava OAuth
    strava_client_id: str = ""
    strava_client_secret: str = ""
    strava_redirect_uri: str = "http://localhost:8000/auth/strava/callback"

    # Frontend URL (for post-OAuth redirect)
    frontend_url: str = "http://localhost:4200"

    # Whitelist (empty string = allow all)
    allowed_athlete_ids: str = ""

    # Data paths
    data_dir: str = "./data"

    @property
    def graphs_dir(self) -> str:
        return f"{self.data_dir}/graphs"

    @property
    def models_dir(self) -> str:
        return f"{self.data_dir}/models"

    @property
    def global_baseline_path(self) -> str:
        return f"{self.models_dir}/global_baseline.pkl"

    @property
    def allowed_ids_list(self) -> list[int]:
        """Returns parsed list of allowed athlete IDs, or empty list (= allow all)."""
        if not self.allowed_athlete_ids.strip():
            return []
        return [int(x.strip()) for x in self.allowed_athlete_ids.split(",") if x.strip()]


settings = Settings()
