import requests
import webbrowser
import json
import os
import pandas as pd
import gpxpy
import gpxpy.gpx
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from dotenv import load_dotenv
from paths import DATA_DIR, CONFIG_DIR

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv(os.path.join(CONFIG_DIR, ".env"))

CLIENT_ID     = os.getenv("STRAVA_CLIENT_ID")
CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")
REDIRECT_URI  = "http://localhost:8080/callback"
TOKEN_FILE    = os.path.join(CONFIG_DIR, "strava_token.json")

MIN_MILES = 10

# ── OAuth Flow ────────────────────────────────────────────────────────────────

auth_code = None

class CallbackHandler(BaseHTTPRequestHandler):
    """Tiny local server to catch the OAuth redirect."""
    def do_GET(self):
        global auth_code
        params = parse_qs(urlparse(self.path).query)
        auth_code = params.get("code", [None])[0]
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"<h2>Auth complete! You can close this tab.</h2>")
    def log_message(self, *args):
        pass


def get_tokens_via_browser():
    """Open browser for user to authorize, capture code via local server."""
    global auth_code

    server = HTTPServer(("localhost", 8080), CallbackHandler)
    thread = threading.Thread(target=server.handle_request)
    thread.start()

    auth_url = (
        f"https://www.strava.com/oauth/authorize"
        f"?client_id={CLIENT_ID}"
        f"&response_type=code"
        f"&redirect_uri={REDIRECT_URI}"
        f"&approval_prompt=force"
        f"&scope=read,activity:read_all"
    )
    print("Opening browser for Strava authorization...")
    webbrowser.open(auth_url)
    thread.join()

    if not auth_code:
        raise RuntimeError("No auth code received — did you approve the app?")

    resp = requests.post("https://www.strava.com/oauth/token", data={
        "client_id":     CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "code":          auth_code,
        "grant_type":    "authorization_code",
    })
    resp.raise_for_status()
    tokens = resp.json()

    with open(TOKEN_FILE, "w") as f:
        json.dump(tokens, f, indent=2)
    print(f"Tokens saved to {TOKEN_FILE}")
    return tokens


def load_or_refresh_tokens():
    """Load saved tokens, refreshing if expired."""
    if not os.path.exists(TOKEN_FILE):
        return get_tokens_via_browser()

    with open(TOKEN_FILE) as f:
        tokens = json.load(f)

    resp = requests.post("https://www.strava.com/oauth/token", data={
        "client_id":     CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type":    "refresh_token",
        "refresh_token": tokens["refresh_token"],
    })
    resp.raise_for_status()
    tokens = resp.json()

    with open(TOKEN_FILE, "w") as f:
        json.dump(tokens, f, indent=2)
    return tokens


# ── Data Fetching ─────────────────────────────────────────────────────────────

def get_activities(access_token, max_rides=100):
    """Fetch cycling activities from Strava."""
    headers = {"Authorization": f"Bearer {access_token}"}
    rides = []
    page = 1

    while len(rides) < max_rides:
        resp = requests.get(
            "https://www.strava.com/api/v3/athlete/activities",
            headers=headers,
            params={
                "per_page": 50,
                "page": page,
            },
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break

        rides_only = [a for a in batch if a.get("sport_type") in (
            "Ride", "MountainBikeRide", "GravelRide"
        )]
        rides.extend(rides_only)
        page += 1

    return rides[:max_rides]


def get_activity_streams(activity_id, access_token):
    """Fetch the GPS stream for a single activity."""
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = requests.get(
        f"https://www.strava.com/api/v3/activities/{activity_id}/streams",
        headers=headers,
        params={"keys": "latlng,altitude,distance", "key_by_type": True},
    )
    resp.raise_for_status()
    return resp.json()


def activities_to_dataframe(activities):
    """Extract useful features from raw activity JSON."""
    rows = []
    for a in activities:
        rows.append({
            "id":              a["id"],
            "name":            a["name"],
            "date":            a["start_date_local"],
            "distance_mi":     round(a["distance"] * 0.000621371, 1),
            "elevation_ft":    round(a.get("total_elevation_gain", 0) * 3.28084),
            "moving_time_min": round(a["moving_time"] / 60),
            "avg_speed_mph":   round(a.get("average_speed", 0) * 2.23694, 1),
            "avg_watts":       a.get("average_watts"),
            "suffer_score":    a.get("suffer_score"),
            "kudos":           a.get("kudos_count", 0),
            "pr_count":        a.get("pr_count", 0),
            "has_route":       bool(a.get("map", {}).get("summary_polyline")),
            "polyline":        a.get("map", {}).get("summary_polyline"),
            "start_lat": a.get("start_latlng", [None, None])[0] if a.get("start_latlng") else None,
            "start_lng": a.get("start_latlng", [None, None])[1] if a.get("start_latlng") else None,
        })
    return pd.DataFrame(rows)


def filter_rides(df):
    """Filter to rides over MIN_MILES."""
    return df[df["distance_mi"] > MIN_MILES]


def save_activity_as_gpx(activity_id, access_token, filename=None):
    """Download a ride's GPS track and save as a .gpx file."""
    streams = get_activity_streams(activity_id, access_token)

    if "latlng" not in streams:
        print(f"No GPS data for activity {activity_id}")
        return None

    latlngs   = streams["latlng"]["data"]
    altitudes = streams.get("altitude", {}).get("data", [None] * len(latlngs))

    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(track)
    segment = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(segment)

    for i, (lat, lng) in enumerate(latlngs):
        ele = altitudes[i] if i < len(altitudes) else None
        segment.points.append(gpxpy.gpx.GPXTrackPoint(lat, lng, elevation=ele))

    fname = filename or f"ride_{activity_id}.gpx"
    with open(fname, "w") as f:
        f.write(gpx.to_xml())
    print(f"Saved: {fname}")
    return fname


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not CLIENT_ID or not CLIENT_SECRET:
        raise ValueError("Missing STRAVA_CLIENT_ID or STRAVA_CLIENT_SECRET in .env file")

    tokens = load_or_refresh_tokens()
    access_token = tokens["access_token"]

    print("\nFetching your rides...")
    activities = get_activities(access_token, max_rides=100)
    df = activities_to_dataframe(activities)
    print(f"Total rides fetched: {len(df)}")

    df = filter_rides(df)
    print(f"Rides after filtering (>{MIN_MILES}mi): {len(df)}")

    print(df[["name", "date", "distance_mi", "elevation_ft", "moving_time_min"]].to_string())

    df.to_csv(os.path.join(DATA_DIR, "my_rides.csv"), index=False)
    print(f"\nSaved to {os.path.join(DATA_DIR, 'my_rides.csv')}")

    # Optional: save a specific ride as GPX
    # save_activity_as_gpx(df.iloc[0]["id"], access_token)