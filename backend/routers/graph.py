# LAYER: Router — Graph Management
# PURPOSE: Exposes graph build status for the Angular "building map" polling flow,
#          and accepts custom location requests to trigger background graph builds.

from fastapi import APIRouter, Depends

from backend.core.security import get_current_user
from backend.models.schemas import GraphRequestBody, GraphStatusResponse
from backend.services import graph_service

router = APIRouter(prefix="/api/graph", tags=["graph"])


@router.get("/status/{city_key}", response_model=GraphStatusResponse)
async def graph_status(city_key: str):
    """
    Poll the build status for a city graph.
    Angular's map-building-banner calls this every 3 seconds until status == "ready".
    """
    status = graph_service.get_graph_status(city_key)
    return GraphStatusResponse(**status)


@router.post("/request")
async def request_graph(
    body: GraphRequestBody,
    _: int = Depends(get_current_user),
):
    """
    Queue a background download for a custom start location.
    Returns the city_key immediately; Angular polls /status/{city_key} to track progress.
    """
    key = await graph_service.request_graph_build(body.lat, body.lng)
    return {"city_key": key}


@router.get("/presets")
async def get_presets(_: int = Depends(get_current_user)):
    """Return the list of pre-seeded cities with their coordinates and status."""
    result = []
    for name, (lat, lng) in graph_service.PRESET_CITIES.items():
        key = graph_service.city_key(lat, lng)
        status = graph_service.get_graph_status(key)
        result.append({"name": name, "lat": lat, "lng": lng,
                       "city_key": key, "status": status["status"]})
    return result
