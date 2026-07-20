# LAYER: Application Entry Point
# PURPOSE: Creates the FastAPI app, registers all routers, configures CORS
#          (Angular dev server on :4200 is allowed), and runs startup tasks:
#          creating DB tables and seeding/loading preset city graphs.

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import settings
from backend.core.db import create_tables
from backend.models import db_models  # noqa
from backend.routers import auth, routes, rides, ratings, graph, coverage
from backend.services.graph_service import seed_preset_cities

# Configure root logging once for the whole app. Modules use
# logging.getLogger("routemaker.<area>") instead of print().
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("routemaker.startup")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Run once on startup (before yield) and shutdown (after yield):
    1. Ensure data directories exist
    2. Migrate legacy graph/model files to the shared data location
    3. Create all DB tables (idempotent)
    4. Load/seed preset city graphs (Iowa City, Madison, Des Moines)
    """
    # Ensure data directories exist
    os.makedirs(settings.graphs_dir, exist_ok=True)
    os.makedirs(settings.models_dir, exist_ok=True)
    os.makedirs(os.path.join(settings.data_dir, "gpx"), exist_ok=True)

    # Migrate the existing Iowa City graph to the new shared location if needed
    old_graph = os.path.join(os.path.dirname(__file__), "..", "data", "iowa_city_network.graphml")
    iowa_key = "41.65_-91.53"
    new_graph = os.path.join(settings.graphs_dir, f"{iowa_key}.graphml")
    if os.path.exists(old_graph) and not os.path.exists(new_graph):
        import shutil
        logger.info("Migrating existing Iowa City graph to new location...")
        shutil.copy2(old_graph, new_graph)

    # Migrate existing global baseline model
    old_model = os.path.join(os.path.dirname(__file__), "..", "data", "route_model.pkl")
    if os.path.exists(old_model) and not os.path.exists(settings.global_baseline_path):
        import shutil
        logger.info("Migrating existing route_model.pkl to global baseline...")
        shutil.copy2(old_model, settings.global_baseline_path)

    await create_tables()
    logger.info("Database tables ready")

    await seed_preset_cities()
    logger.info("Preset city graphs loaded")

    yield


app = FastAPI(
    title="RouteMaker API",
    description="Personalized AI cycling route generation backend.",
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allow the Angular dev server and, in production, the deployed frontend URL.
allowed_origins = [
    "http://localhost:4200",   # Angular dev server
    settings.frontend_url,     # Configured frontend (same in dev, different in prod)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(set(allowed_origins)),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(routes.router)
app.include_router(rides.router)
app.include_router(ratings.router)
app.include_router(graph.router)
app.include_router(coverage.router)


@app.get("/health")
async def health():
    """Health check endpoint. Returns 200 if the server is running."""
    return {"status": "ok"}
