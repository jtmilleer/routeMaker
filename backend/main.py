# LAYER: Application Entry Point
# PURPOSE: Creates the FastAPI app, registers all routers, configures CORS
#          (Angular dev server on :4200 is allowed), and runs startup tasks:
#          creating DB tables and seeding/loading preset city graphs.

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import settings
from backend.core.db import engine, Base
from backend.models import db_models  # noqa
from backend.routers import auth, routes, rides, ratings, graph
from backend.services.graph_service import graph_service
from backend.services.model_service import model_service

app = FastAPI(
    title="RouteMaker API",
    description="Personalized AI cycling route generation backend.",
    version="2.0.0",
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


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    """
    Run once when the server starts:
    1. Create all DB tables (idempotent — safe to run on every start)
    2. Ensure data directories exist
    3. Load/seed preset city graphs (Iowa City, Madison, Des Moines)
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
        print("[startup] Migrating existing Iowa City graph to new location...")
        shutil.copy2(old_graph, new_graph)

    # Migrate existing global baseline model
    old_model = os.path.join(os.path.dirname(__file__), "..", "data", "route_model.pkl")
    if os.path.exists(old_model) and not os.path.exists(settings.global_baseline_path):
        import shutil
        print("[startup] Migrating existing route_model.pkl to global baseline...")
        shutil.copy2(old_model, settings.global_baseline_path)

    await create_tables()
    print("[startup] Database tables ready")

    await seed_preset_cities()
    print("[startup] Preset city graphs loaded")


@app.get("/health")
async def health():
    """Health check endpoint. Returns 200 if the server is running."""
    return {"status": "ok"}
