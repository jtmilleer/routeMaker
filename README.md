# RouteMaker

RouteMaker is a personalized AI-powered cycling route generator. It creates customized routes (lollipop loops, hilly climbs, historic tours, and novel roads) tailored to your specific riding preferences. It learns from your Strava activity history and ratings using a personalized Machine Learning model.

## Architecture

RouteMaker is a full-stack application built with:
- **Frontend:** Angular 17, Leaflet (Map UI)
- **Backend:** Python FastAPI, SQLAlchemy (SQLite)
- **Data/ML:** Scikit-learn, OSMnx, NetworkX

## Prerequisites

To run this project locally, you will need:
1. **Python 3.10+** (Tested with Python 3.13)
2. **Node.js** (v20+ recommended)
3. A **Strava API Application**. You can create one at [https://www.strava.com/settings/api](https://www.strava.com/settings/api).
   - Set the **Authorization Callback Domain** to `localhost`

## 1. Environment Setup

### Configure the Backend
Navigate to the `backend` directory and copy the environment template:
```bash
cd backend
cp .env.example .env
```
*(On Windows PowerShell, use `Copy-Item .env.example .env`)*

Open the new `.env` file and fill in your Strava credentials:
```env
# Strava OAuth
STRAVA_CLIENT_ID=your_client_id_here
STRAVA_CLIENT_SECRET=your_client_secret_here

# JWT Security
# Generate a random string for this (e.g., using `openssl rand -hex 32`)
JWT_SECRET=your_secure_random_string

# For testing, you must put your Strava Athlete ID here to allow login.
# Leave blank to allow anyone (not recommended for public exposure).
ALLOWED_ATHLETE_IDS=your_athlete_id
```

## 2. Running the Backend

The backend runs on FastAPI and handles route generation, ML training, and database management.

Open a terminal in the root project directory:

```powershell
# 1. Set the PYTHONPATH so module imports work correctly
$env:PYTHONPATH = (Get-Location).Path

# 2. Navigate to the backend folder
cd backend

# 3. Create a virtual environment
python -m venv .venv

# 4. Activate the virtual environment
# On Windows:
.\.venv\Scripts\activate
# On Mac/Linux:
# source .venv/bin/activate

# 5. Install dependencies
pip install -r requirements.txt

# 6. Start the development server
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

The backend should now be running at `http://127.0.0.1:8000`. It will automatically create the local SQLite database and necessary data directories on its first run.

## 3. Running the Frontend

The frontend is an Angular application. Open a **second terminal window** in the root project directory.

```powershell
# 1. Navigate to the frontend folder
cd frontend

# 2. Install dependencies
npm install

# 3. Start the Angular development server
npm start
```

The frontend will start and automatically open in your browser at `http://localhost:4200`. 
*Note: The Angular app is configured to automatically proxy all `/api` and `/auth` requests to the FastAPI backend running on port 8000.*

## Troubleshooting

- **ModuleNotFoundError: No module named 'backend'**: Ensure you set the `PYTHONPATH` variable in your terminal before running `uvicorn`, or ensure you are running `uvicorn backend.main:app` from the root project directory (not inside the `backend` directory itself).
- **Graph Building takes a long time**: The first time you request a custom location, the backend downloads the road network from OpenStreetMap. This can take 30-60 seconds depending on the radius. Subsequent requests in the same city are cached.
