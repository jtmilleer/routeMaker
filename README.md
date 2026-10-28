# RouteMaker

Cycling route generator that connects to your Strava account and builds personalized routes using a road network graph and a per-user ML model trained on your own ride ratings.

Started as a collection of Python scripts for generating routes around Iowa City. Now has a full web UI (Angular + FastAPI) so you can pick a starting point on a map, choose a route type, and get back GPX files ranked by how much the model thinks you'll enjoy them.

## How it works

1. You log in with Strava. The app imports your ride history.
2. You rate rides on a 1-10 scale. These ratings train a personal GradientBoostingRegressor model (scikit-learn).
3. When you generate routes, the backend loads an OpenStreetMap road network graph (via OSMnx), runs A* pathfinding to build lollipop loops near your target distance, scores each candidate with your model, and returns the top 5.
4. After you ride a generated route, you rate it too. Every 10 ratings, the model retrains automatically in the background.

**Route types:**
- **Regular** -- standard loops, scored by your model
- **Hilly** -- routing weights favor roads with steeper grades
- **Historic** -- biases toward roads you've ridden frequently (your comfort roads)
- **Novel** -- penalizes roads you've already ridden so you explore new ones

Road network graphs are cached per-city and shared across users. Building a new city graph from scratch takes 5-15 minutes (downloads OSM data, strips highways, adds SRTM elevation to every node, computes edge grades, extracts the largest connected component). Iowa City, Madison, and Des Moines are pre-seeded.

## Stack

- **Frontend:** Angular 22, Leaflet, TypeScript
- **Backend:** Python, FastAPI, SQLAlchemy (async), SQLite
- **ML:** scikit-learn (GradientBoostingRegressor + StandardScaler per user)
- **Geo:** OSMnx, NetworkX, SRTM elevation data, KDTree spatial indexing
- **Auth:** Strava OAuth2 + JWT

## Setup

**Requirements:** Python 3.12 or 3.13, Node.js v20+, a [Strava API app](https://www.strava.com/settings/api) with callback domain set to `localhost`.

```
git clone <this repo>
cd routeMaker
```

Create `backend/.env` from the example and fill in your Strava credentials:

```
cp backend/.env.example backend/.env
```

```env
STRAVA_CLIENT_ID=your_client_id
STRAVA_CLIENT_SECRET=your_client_secret
JWT_SECRET=any_long_random_string
```

Install backend dependencies (Python must be 3.12 or 3.13 -- not 3.14):

**Windows (PowerShell)**

```powershell
cd backend
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd ..
```

**Linux / macOS**

```bash
cd backend
python3.13 -m venv .venv     # or python3.12
source .venv/bin/activate
pip install -r requirements.txt
cd ..
```

On Debian/Ubuntu, if `venv` fails with `No module named 'ensurepip'`, install the venv package first: `sudo apt install python3.12-venv` (or `python3.13-venv`).

Install frontend dependencies:

```
cd frontend
npm install
cd ..
```

Add a CARTO basemaps key for the map tiles. Get one free at https://carto.com/basemaps/apikey, then copy the template and paste the key into `CARTO_API_KEY`:

```
cp frontend/src/environments/carto-key.example.ts frontend/src/environments/carto-key.ts
```

`carto-key.ts` is gitignored. The frontend won't build without it, and the maps show an "API KEY REQUIRED" watermark until the key is set.

## Running

In two terminals from the project root:

```
# Terminal 1 - backend
.\backend\.venv\Scripts\Activate.ps1   # Windows
source backend/.venv/bin/activate      # Linux / macOS
python -m backend

# Terminal 2 - frontend
cd frontend
npm start
```

Open `http://localhost:4200`. API docs at `http://localhost:8000/docs`.

## Project layout

```
backend/
  main.py              # FastAPI app, CORS, startup (table creation, graph seeding)
  core/                # config (pydantic-settings), db engine, JWT security
  models/              # SQLAlchemy ORM models, Pydantic request/response schemas
  routers/             # auth (Strava OAuth), routes, rides, ratings, graph
  services/
    graph_service.py   # OSMnx graph download/cache/spatial index, ridden-edge marking
    model_service.py   # per-user .pkl model loading, prediction, retraining
    strava_service.py  # all Strava API calls, token refresh
frontend/
  src/app/
    features/          # landing, route-builder, rate-rides, rate-generated
    core/              # auth service, route API/state services, guards, interceptors
    shared/            # city selector, map building banner, loading overlay
src/                   # original CLI scripts (generate_routes.py, train_model.py, etc.)
```

## Notes

- Per-user models live in `data/models/`. New users start with a copy of the global baseline model.
- Road network graphs are in `data/graphs/` (~100MB+ per city as graphml). They're shared across users.
- Switch to Postgres by changing `DATABASE_URL` in `.env` to `postgresql+asyncpg://...` -- no code changes needed.
- The `src/` directory has the original standalone Python scripts that this project grew out of.
