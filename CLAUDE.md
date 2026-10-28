# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

RouteMaker: a cycling route generator. Users log in with Strava, rate rides, and the
app trains a per-user ML model on those ratings. Route generation loads an OSM road
network graph, runs A* pathfinding to build candidate loops near a target distance,
scores each with the user's model, and returns the top 5 as GPX downloads.

Stack: Angular 22 + Leaflet frontend, FastAPI + SQLAlchemy (async) + SQLite backend,
scikit-learn (GradientBoostingRegressor + StandardScaler per user), OSMnx/NetworkX/SRTM
for geospatial routing.

## Commands

Run from the project root in two terminals — the backend needs its venv active, the
frontend runs via npm from `frontend/`.

```
# Backend (from repo root, venv active)
.\backend\.venv\Scripts\Activate.ps1
python -m backend                    # serves on :8000, docs at /docs

# Frontend
cd frontend
npm start                            # ng serve, :4200
npm run build                        # ng build -> dist/
npm test                             # ng test (Vitest runner)
```

First-time frontend setup: copy `frontend/src/environments/carto-key.example.ts` to
`carto-key.ts` (gitignored) and add a free CARTO basemaps key — the build fails without
the file, and map tiles are watermarked without the key.

There is no backend test suite (no `test*` files under `backend/`). Frontend has only
the Angular CLI boilerplate spec (`app.spec.ts`) — no meaningful test coverage exists
on either side yet. Don't assume `npm test` or a pytest command validates behavior.

No lint command is configured on the backend. Frontend has no separate lint script either
(no ESLint config present) — Prettier is a devDependency but not wired to a script.

## Architecture

### Backend layering (`backend/`)

Every file opens with a `# LAYER: ... / # PURPOSE: ...` header — read it before editing,
it states the file's contract more precisely than inferring from code.

- `main.py` — FastAPI app, CORS, lifespan startup (creates DB tables, seeds/loads preset
  city graphs, migrates legacy data files into `data/`).
- `core/config.py` — single `Settings` object (pydantic-settings) read from `backend/.env`.
  **Always import `settings` from here — never `os.getenv()` directly.**
- `core/db.py` — async SQLAlchemy engine/session. `get_db()` is the FastAPI dependency
  every route uses. **No migration framework (Alembic is a dependency but unused for
  this)** — new columns on existing tables must be added to `_ADDED_COLUMNS` in this
  file so `create_tables()` ALTERs them idempotently on startup. `create_all` never
  alters existing tables, only creates missing ones.
- `core/security.py` — JWT issue/verify. `get_current_user()` is the auth dependency
  (returns `athlete_id: int`, raises 401).
- `models/db_models.py` — SQLAlchemy ORM. `athlete_id` (Strava's own integer ID) is the
  user key everywhere — no separate internal UUID for users.
- `models/schemas.py` — Pydantic request/response models.
- `routers/` — one file per resource (`auth`, `routes`, `rides`, `ratings`, `graph`,
  `coverage`). Routers stay thin: validate, call a service, persist/return. Ownership is
  checked manually in the router (e.g. `row.athlete_id != athlete_id` → 404), there's no
  shared ownership-check helper.
- `services/` — where the real logic lives:
  - `graph_service.py` — downloads/caches OSMnx road network graphs per city, keyed by
    rounded lat/lng (`city_key`, e.g. `"41.65_-91.53"` — two points within ~1km share a
    graph). In-memory `_graph_cache` avoids reloading the 100MB+ graphml per request.
    `_build_status` tracks async build progress for polling. Graphs are shared across
    all users, not per-user.
  - `model_service.py` — per-user model load/predict/retrain. Cache: memory → user's
    `.pkl` on disk → copy of `global_baseline.pkl` for first-time users. Feature
    engineering in `predict_score()` must mirror `src/train_model.py` exactly (same
    input shape the model was trained on) — a mismatch silently produces bad scores,
    not an error.
  - `route_service.py` — A* candidate generation and scoring.
  - `coverage_service.py` — the "100% coverage" feature (ride matching against the
    graph to compute street coverage %, generate gap-filling loops).
  - `strava_service.py` — all Strava API calls + token refresh.
- CPU-bound work (route generation, coverage computation) runs via
  `loop.run_in_executor(ThreadPoolExecutor, ...)` from the router — never blocks the
  async event loop. Follow this pattern for any new CPU-heavy endpoint.
- Logging: `logging.getLogger("routemaker.<area>")`, configured once in `main.py`. Don't
  use `print()`.

### Data & DB gotchas

- `DATABASE_URL` default (`sqlite+aiosqlite:///./routemaker.db`) is relative to the
  process's cwd. Since the backend is launched with `python -m backend` **from the repo
  root**, the live DB is `<repo-root>/routemaker.db`. Running the server from inside
  `backend/` would silently create a fresh empty DB (and `data/` dir) there instead.
- Switching to Postgres is a one-line `DATABASE_URL` change in `.env`
  (`postgresql+asyncpg://...`) — no code changes needed elsewhere.
- Per-user models: `data/models/<athlete_id>.pkl` (repo root, via `DATA_DIR=./data`). Road graphs:
  `data/graphs/<city_key>.graphml`. Both are gitignored data, not source.

### Frontend (`frontend/src/app/`)

- Angular 22, standalone components only (no NgModules). Routes are lazy-loaded via
  `loadComponent` in `app.routes.ts`, guarded by `authGuard` for everything under `/app`.
- **Components are single-file**: template and styles are inline in the `.ts` file via
  `@Component({ template: ..., styles: ... })` — there are no sibling `.html`/`.css`
  files per component. Follow this convention for new components.
- `core/services/route-api.service.ts` is the **single source of truth for all HTTP
  calls** to the backend — components never call `HttpClient` directly. Add new backend
  endpoints here first, typed with request/response interfaces, before wiring up a
  component.
- `core/services/auth.service.ts` — JWT lives in `localStorage`, exposed as a signal
  (`currentUser`). `auth.interceptor.ts` attaches the token to outgoing requests.
  OAuth flow: backend redirects to Strava, Strava redirects back to
  `/auth/callback#token=...`, `handleCallback()` reads the fragment.
  `refreshSession()` silently re-issues a JWT on each app load so an open tab doesn't
  expire mid-session.
- `core/services/route-state.service.ts` — shared UI/route state across route-builder
  subcomponents.
- `features/` — one folder per page (`landing`, `route-builder`, `rate-rides`,
  `rate-generated`, `coverage`, `map-view`). `route-builder/` further splits into
  `route-form/` and `route-results/` subcomponents.
- `shared/components/` — cross-feature UI (`city-selector`, `map-building-banner`,
  `loading-overlay`, `route-card`).

### Design system — "Field Guide"

The whole app (not just landing) follows a warm paper/trail-map visual identity, defined
in `frontend/src/styles.css` `:root` and documented in
`docs/superpowers/specs/2026-07-20-frontend-redesign-design.md`. Key tokens: `--bg-primary`
(paper cream `#e0d5b8`), `--accent` (rust/terracotta `#a8471f`), `--font-display`
(Fraunces serif for headings), `--font-primary` (Inter). Leaflet basemap uses CARTO's
warm "Voyager" tiles, not a dark tile set. **New UI work should reuse these tokens
rather than introducing new colors/fonts** — this palette was deliberately chosen over
several alternatives after user review (see the spec doc for the rejected options and why).

### `src/` — legacy CLI scripts

`src/` (repo root, not `backend/`) holds the original standalone Python scripts
(`generate_routes.py`, `train_model.py`, etc.) that the web app grew out of. The model
architecture and feature engineering in `backend/services/model_service.py` must stay
consistent with `src/train_model.py`. Not part of the running app.

## Working conventions

- **Parallelize independent work via subagents.** When a task splits into pieces that
  are well-defined (clear scope, clear done-condition) and independent (no shared state,
  no sequential dependency), dispatch each piece to a subagent and run them in parallel
  rather than working through them one at a time in the main thread. Reserve sequential,
  inline work for pieces that share state, depend on each other's output, or need
  ongoing judgment calls you can't hand off.
- **No backwards-compatibility shims.** This project has no external consumers/API
  contract to preserve. If a better design exists, prefer changing the code over adding
  a compatibility path, deprecated alias, or feature flag — but confirm with the user
  before making a breaking change to shared structure (DB schema, API request/response
  shape, config keys), since it may affect data already on disk (models, graphs, the
  live DB) or an in-progress frontend/backend contract.
- Route types (`regular`, `hilly`, `historic`, `novel`, `coverage`) are threaded through
  request params, generation logic, and scoring — check `route_service.py` and
  `coverage_service.py` together when changing how a type behaves.
