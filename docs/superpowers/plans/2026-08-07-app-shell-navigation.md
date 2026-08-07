# App Shell & Navigation Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the four independently-drifting `/app/*` nav bars with one shared app shell, merge Rate Rides/Rate Generated into a single "Rate" section, and add a pending-ratings badge to the nav.

**Architecture:** `/app` becomes a real parent route owning a new `AppShellComponent` (persistent nav + `<router-outlet>`). `RouteBuilderComponent` and `CoverageComponent` become its children directly; a new `RateShellComponent` becomes a third child that itself owns a Rides/Generated sub-nav and two grandchild routes. A small `PendingRatingsService` holds the nav badge count, fed by a new backend endpoint. Two small backend additions (a pending-counts endpoint, and a `user_rating` field on generated-route history) are prerequisites for the rating pages to distinguish rated from unrated items at all.

**Tech Stack:** Angular 22 standalone components, Angular Router (nested routes), RxJS/signals, FastAPI + SQLAlchemy async, Pydantic.

## Global Constraints

- No backend test suite and no meaningful frontend test coverage exists in this repo (confirmed in `CLAUDE.md`) — this plan does not introduce one. Backend tasks verify with `python -m py_compile` (syntax) plus a manual server smoke check; frontend tasks verify with `npm run build` (catches TypeScript/template errors) plus a final manual click-through in the browser.
- URL prefix stays `/app/*` (confirmed with user — the `/app` boundary between public and authenticated pages is intentional and only one redirect target, `auth.service.ts:98`, depends on it).
- Follow existing per-component inline template/styles convention — no separate `.html`/`.css` files.
- `RouteApiService` (`frontend/src/app/core/services/route-api.service.ts`) stays the single source of truth for HTTP calls — no component calls `HttpClient` directly.
- Always import `settings` in backend code, never `os.getenv()` directly (n/a for this plan's backend changes, noted per project convention).

---

## Task 1: Backend — `PendingCounts` schema + `user_rating` on `RouteResult`

**Files:**
- Modify: `backend/models/schemas.py`

**Interfaces:**
- Produces: `PendingCounts(unrated_rides: int, unrated_routes: int)` — used by Task 2.
- Produces: `RouteResult.user_rating: Optional[int] = None` — new field, used by Task 3 and by frontend Task 4.

- [ ] **Step 1: Add `user_rating` to `RouteResult`**

In `backend/models/schemas.py`, find the `RouteResult` class:

```python
class RouteResult(BaseModel):
    """A single generated route returned to Angular."""
    id: str                          # UUID
    polyline: str                    # Google encoded polyline
    route_segments: Optional[str] = None  # JSON string; only for novel routes
    distance_mi: float
    elevation_ft: float
    predicted_score: float
    novelty_pct: Optional[float] = None  # only for novel routes
    historic_sites: Optional[list[HistoricSite]] = None
    # Downsampled [distance_mi, elevation_ft] points for the elevation chart.
    # Populated on generation; omitted on history (not persisted to the DB).
    elevation_profile: Optional[list[list[float]]] = None
    city_key: str
    route_type: str
    gpx_path: Optional[str] = Field(default=None, exclude=True)
```

Replace with (adds `user_rating` before `gpx_path`):

```python
class RouteResult(BaseModel):
    """A single generated route returned to Angular."""
    id: str                          # UUID
    polyline: str                    # Google encoded polyline
    route_segments: Optional[str] = None  # JSON string; only for novel routes
    distance_mi: float
    elevation_ft: float
    predicted_score: float
    novelty_pct: Optional[float] = None  # only for novel routes
    historic_sites: Optional[list[HistoricSite]] = None
    # Downsampled [distance_mi, elevation_ft] points for the elevation chart.
    # Populated on generation; omitted on history (not persisted to the DB).
    elevation_profile: Optional[list[list[float]]] = None
    city_key: str
    route_type: str
    user_rating: Optional[int] = None  # Joined from route_feedback; None if not yet rated
    gpx_path: Optional[str] = Field(default=None, exclude=True)
```

- [ ] **Step 2: Add the `PendingCounts` schema**

In the same file, find the `RatingStats` class (in the `# ── Ratings ──` section):

```python
class RatingStats(BaseModel):
    """
    Returned after any rating action. Angular uses this to update the
    model training progress indicator.
    """
    total_ratings: int
    model_version: int
    ratings_until_next_retrain: int
    model_trained_at: Optional[datetime] = None
```

Add directly below it:

```python


class PendingCounts(BaseModel):
    """GET /api/ratings/pending-counts — powers the Rate nav badge."""
    unrated_rides: int
    unrated_routes: int
```

- [ ] **Step 3: Verify it compiles**

Run: `python -m py_compile backend/models/schemas.py`
Expected: no output, exit code 0.

- [ ] **Step 4: Commit**

```bash
git add backend/models/schemas.py
git commit -m "feat: add PendingCounts schema and RouteResult.user_rating"
```

---

## Task 2: Backend — `GET /api/ratings/pending-counts`

**Files:**
- Modify: `backend/routers/ratings.py`

**Interfaces:**
- Consumes: `PendingCounts` from Task 1 (`backend/models/schemas.py`).
- Produces: `GET /api/ratings/pending-counts` → `{"unrated_rides": int, "unrated_routes": int}`. Consumed by frontend Task 4/5.

- [ ] **Step 1: Add imports**

In `backend/routers/ratings.py`, find:

```python
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.db import AsyncSessionFactory, get_db
from backend.core.security import get_current_user
from backend.models.db_models import GeneratedRoute, Ride, RideRating, RouteFeedback, User
from backend.models.schemas import RideRatingRequest, RouteRatingRequest, RatingStats
from backend.services import model_service
```

Replace with:

```python
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.db import AsyncSessionFactory, get_db
from backend.core.security import get_current_user
from backend.models.db_models import GeneratedRoute, Ride, RideRating, RouteFeedback, User
from backend.models.schemas import PendingCounts, RideRatingRequest, RouteRatingRequest, RatingStats
from backend.services import model_service
```

- [ ] **Step 2: Add the endpoint**

At the end of the file, after `get_rating_stats`, add:

```python


@router.get("/pending-counts", response_model=PendingCounts)
async def get_pending_counts(
    db: AsyncSession = Depends(get_db),
    athlete_id: int = Depends(get_current_user),
):
    """Count unrated rides and unrated generated routes, for the nav badge."""
    total_rides = (await db.execute(
        select(func.count()).select_from(Ride).where(
            Ride.athlete_id == athlete_id,
            Ride.activity_type == "ride",
        )
    )).scalar_one()

    rated_rides = (await db.execute(
        select(func.count()).select_from(RideRating).where(RideRating.athlete_id == athlete_id)
    )).scalar_one()

    total_routes = (await db.execute(
        select(func.count()).select_from(GeneratedRoute).where(GeneratedRoute.athlete_id == athlete_id)
    )).scalar_one()

    rated_routes = (await db.execute(
        select(func.count()).select_from(RouteFeedback).where(RouteFeedback.athlete_id == athlete_id)
    )).scalar_one()

    return PendingCounts(
        unrated_rides=max(total_rides - rated_rides, 0),
        unrated_routes=max(total_routes - rated_routes, 0),
    )
```

- [ ] **Step 3: Verify it compiles**

Run: `python -m py_compile backend/routers/ratings.py`
Expected: no output, exit code 0.

- [ ] **Step 4: Manual smoke check**

With the backend venv active, from the repo root:

Run: `python -m backend` (leave running), then in another terminal: `curl http://localhost:8000/openapi.json | grep pending-counts`
Expected: the string `/api/ratings/pending-counts` appears in the output. Stop the server after checking (Ctrl+C).

- [ ] **Step 5: Commit**

```bash
git add backend/routers/ratings.py
git commit -m "feat: add GET /api/ratings/pending-counts endpoint"
```

---

## Task 3: Backend — populate `user_rating` on `GET /api/routes/history`

**Files:**
- Modify: `backend/routers/routes.py`

**Interfaces:**
- Consumes: `RouteResult.user_rating` from Task 1.
- Produces: `GET /api/routes/history` now returns each route's `user_rating` (null if unrated). Consumed by frontend Tasks 8/9 to filter unrated vs. rated.

- [ ] **Step 1: Import `RouteFeedback`**

In `backend/routers/routes.py`, find:

```python
from backend.models.db_models import GeneratedRoute, Ride
```

Replace with:

```python
from backend.models.db_models import GeneratedRoute, Ride, RouteFeedback
```

- [ ] **Step 2: Join feedback into `get_history`**

Find:

```python
@router.get("/history", response_model=list[RouteResult])
async def get_history(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
    athlete_id: int = Depends(get_current_user),
):
    """Return the user's previously generated routes (most recent first)."""
    result = await db.execute(
        select(GeneratedRoute)
        .where(GeneratedRoute.athlete_id == athlete_id)
        .order_by(GeneratedRoute.created_at.desc())
        .limit(limit)
    )
    rows = result.scalars().all()
    return [
        RouteResult(
            id=r.id,
            polyline=r.polyline or "",
            route_segments=r.route_segments,
            distance_mi=r.distance_mi or 0,
            elevation_ft=r.elevation_ft or 0,
            predicted_score=r.predicted_score or 0,
            novelty_pct=r.novelty_pct,
            city_key=r.city_key or "",
            route_type=r.route_type,
        )
        for r in rows
    ]
```

Replace with:

```python
@router.get("/history", response_model=list[RouteResult])
async def get_history(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
    athlete_id: int = Depends(get_current_user),
):
    """Return the user's previously generated routes (most recent first), with
    each route's rating merged in (null if not yet rated)."""
    result = await db.execute(
        select(GeneratedRoute)
        .where(GeneratedRoute.athlete_id == athlete_id)
        .order_by(GeneratedRoute.created_at.desc())
        .limit(limit)
    )
    rows = result.scalars().all()

    feedback_result = await db.execute(
        select(RouteFeedback).where(RouteFeedback.athlete_id == athlete_id)
    )
    ratings_map = {f.route_id: f.rating for f in feedback_result.scalars().all()}

    return [
        RouteResult(
            id=r.id,
            polyline=r.polyline or "",
            route_segments=r.route_segments,
            distance_mi=r.distance_mi or 0,
            elevation_ft=r.elevation_ft or 0,
            predicted_score=r.predicted_score or 0,
            novelty_pct=r.novelty_pct,
            city_key=r.city_key or "",
            route_type=r.route_type,
            user_rating=ratings_map.get(r.id),
        )
        for r in rows
    ]
```

- [ ] **Step 3: Verify it compiles**

Run: `python -m py_compile backend/routers/routes.py`
Expected: no output, exit code 0.

- [ ] **Step 4: Commit**

```bash
git add backend/routers/routes.py
git commit -m "feat: merge user_rating into GET /api/routes/history"
```

---

## Task 4: Frontend — `route-api.service.ts` types + `getPendingCounts()`

**Files:**
- Modify: `frontend/src/app/core/services/route-api.service.ts`

**Interfaces:**
- Consumes: backend `GET /api/ratings/pending-counts` (Task 2), backend `RouteResult.user_rating` (Task 3).
- Produces: `PendingCounts` interface, `RouteApiService.getPendingCounts(): Observable<PendingCounts>`, `RouteResult.user_rating?: number | null`. Used by Task 5 (`PendingRatingsService`) and Tasks 8/9 (rate components).

- [ ] **Step 1: Add `user_rating` to the `RouteResult` interface**

Find:

```typescript
export interface RouteResult {
  id: string;
  polyline: string;
  route_segments?: string;   // JSON string, only for novel routes
  distance_mi: number;
  elevation_ft: number;
  predicted_score: number;
  novelty_pct?: number;
  historic_sites?: HistoricSite[];
  elevation_profile?: number[][];   // [distance_mi, elevation_ft] points
  city_key: string;
  route_type: string;
}
```

Replace with:

```typescript
export interface RouteResult {
  id: string;
  polyline: string;
  route_segments?: string;   // JSON string, only for novel routes
  distance_mi: number;
  elevation_ft: number;
  predicted_score: number;
  novelty_pct?: number;
  historic_sites?: HistoricSite[];
  elevation_profile?: number[][];   // [distance_mi, elevation_ft] points
  city_key: string;
  route_type: string;
  user_rating?: number | null;   // Joined from route_feedback; null if not yet rated
}
```

- [ ] **Step 2: Add the `PendingCounts` interface**

Find:

```typescript
export interface RatingStats {
  total_ratings: number;
  model_version: number;
  ratings_until_next_retrain: number;
  model_trained_at: string | null;
}
```

Add directly below it:

```typescript

export interface PendingCounts {
  unrated_rides: number;
  unrated_routes: number;
}
```

- [ ] **Step 3: Add the `getPendingCounts()` method**

Find:

```typescript
  getRatingStats(): Observable<RatingStats> {
    return this.http.get<RatingStats>(`${this.base}/api/ratings/stats`);
  }
```

Replace with:

```typescript
  getRatingStats(): Observable<RatingStats> {
    return this.http.get<RatingStats>(`${this.base}/api/ratings/stats`);
  }

  getPendingCounts(): Observable<PendingCounts> {
    return this.http.get<PendingCounts>(`${this.base}/api/ratings/pending-counts`);
  }
```

- [ ] **Step 4: Verify it builds**

Run (from `frontend/`): `npm run build`
Expected: build succeeds with no TypeScript errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/app/core/services/route-api.service.ts
git commit -m "feat: add PendingCounts type and getPendingCounts() to RouteApiService"
```

---

## Task 5: Frontend — `PendingRatingsService`

**Files:**
- Create: `frontend/src/app/core/services/pending-ratings.service.ts`

**Interfaces:**
- Consumes: `RouteApiService.getPendingCounts()` from Task 4.
- Produces: `PendingRatingsService.pendingCount: Signal<number | null>`, `.refresh(): void`, `.decrement(): void`. Used by Task 6 (`AppShellComponent`, reads `pendingCount`, calls `refresh()`) and Tasks 8/9 (rate components, call `decrement()`).

- [ ] **Step 1: Create the service**

```typescript
// LAYER: Service — Pending Ratings Count
// PURPOSE: Tracks how many rides + generated routes are still unrated, for the
//          "Rate" nav badge. Fetched once by AppShellComponent; decremented
//          locally by the rate sub-components on a successful (not skipped)
//          rating so the badge updates without a refetch.

import { Injectable, signal } from '@angular/core';
import { RouteApiService } from './route-api.service';

@Injectable({ providedIn: 'root' })
export class PendingRatingsService {
  readonly pendingCount = signal<number | null>(null);

  constructor(private api: RouteApiService) {}

  refresh(): void {
    this.api.getPendingCounts().subscribe({
      next: counts => this.pendingCount.set(counts.unrated_rides + counts.unrated_routes),
      error: () => { /* non-critical — badge just stays hidden */ },
    });
  }

  decrement(): void {
    const current = this.pendingCount();
    if (current != null && current > 0) {
      this.pendingCount.set(current - 1);
    }
  }
}
```

- [ ] **Step 2: Verify it builds**

Run (from `frontend/`): `npm run build`
Expected: build succeeds with no TypeScript errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/app/core/services/pending-ratings.service.ts
git commit -m "feat: add PendingRatingsService for the Rate nav badge"
```

---

## Task 6: Frontend — `AppShellComponent`

**Files:**
- Create: `frontend/src/app/shared/components/app-shell/app-shell.component.ts`

**Interfaces:**
- Consumes: `AuthService` (`core/services/auth.service.ts` — `currentUser` signal, `logout()`), `PendingRatingsService` from Task 5, `MapBuildingBannerComponent` (`shared/components/map-building-banner/map-building-banner.component.ts`, selector `app-map-building-banner`, no inputs).
- Produces: `AppShellComponent`, selector `app-shell`. Wired as the parent `/app` route component in Task 12. Its `:host` and `.app-shell` are `display:flex/column; height:100vh` — child route components must be flex items (`flex:1; min-height:0`) to fill the remaining height under the nav (see Tasks 7, 10, 11).

- [ ] **Step 1: Create the component**

```typescript
// LAYER: Component — App Shell
// PURPOSE: Persistent top-level chrome for every authenticated /app page —
//          nav (Builder/Rate/Coverage), user info, sign out, and the
//          graph-build banner. Hosts the router-outlet for the active section.

import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { AuthService } from '../../../core/services/auth.service';
import { PendingRatingsService } from '../../../core/services/pending-ratings.service';
import { MapBuildingBannerComponent } from '../map-building-banner/map-building-banner.component';

@Component({
  selector: 'app-shell',
  standalone: true,
  imports: [CommonModule, RouterLink, RouterLinkActive, RouterOutlet, MapBuildingBannerComponent],
  template: `
    <div class="app-shell">
      <nav class="topnav">
        <div class="nav-brand">
          <span class="brand-name">RouteMaker</span>
        </div>
        <div class="nav-links">
          <a routerLink="/app" routerLinkActive="active" [routerLinkActiveOptions]="{ exact: true }" class="nav-link" id="nav-builder">Builder</a>
          <a routerLink="/app/rate" routerLinkActive="active" class="nav-link" id="nav-rate">
            Rate
            <span class="nav-badge" *ngIf="pending.pendingCount()">{{ pending.pendingCount() }}</span>
          </a>
          <a routerLink="/app/coverage" routerLinkActive="active" class="nav-link" id="nav-coverage">Coverage</a>
          <div class="nav-user" *ngIf="auth.currentUser() as user">
            <img *ngIf="user.profile_pic_url" [src]="user.profile_pic_url" class="avatar" alt="Profile">
            <span class="user-name">{{ user.name }}</span>
            <span class="model-badge" title="Preference model version">v{{ user.model_version }}</span>
          </div>
          <button class="logout-btn" (click)="auth.logout()" id="logout-btn">Sign Out</button>
        </div>
      </nav>

      <app-map-building-banner></app-map-building-banner>

      <router-outlet></router-outlet>
    </div>
  `,
  styles: [`
    :host { display: block; height: 100vh; overflow: hidden; }
    .app-shell { display: flex; flex-direction: column; height: 100vh; background: var(--bg-primary); color: var(--text-primary); font-family: var(--font-primary); }

    .topnav {
      display: flex; align-items: center; justify-content: space-between;
      padding: 0 1.5rem; height: 56px; min-height: 56px;
      background: rgba(224, 213, 184, 0.95);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(12px);
      z-index: 100;
    }
    .nav-brand { display: flex; align-items: center; gap: 0.5rem; }
    .brand-name { font-size: 1.25rem; font-weight: 700; color: var(--text-primary); font-family: var(--font-display); }

    .nav-links { display: flex; align-items: center; gap: 1rem; }
    .nav-link { color: var(--text-muted); text-decoration: none; font-size: 0.875rem; font-weight: 500; transition: color 0.15s; display: flex; align-items: center; }
    .nav-link:hover { color: var(--text-primary); }
    .nav-link.active { color: var(--accent); font-weight: 700; }
    .nav-badge {
      display: inline-block; margin-left: 0.4rem; background: var(--accent); color: var(--on-accent);
      font-size: 0.7rem; font-weight: 700; border-radius: 999px; padding: 0.05rem 0.4rem;
      font-family: var(--font-mono); line-height: 1.4;
    }
    .nav-user { display: flex; align-items: center; gap: 0.5rem; }
    .avatar { width: 28px; height: 28px; border-radius: 50%; }
    .user-name { font-size: 0.875rem; color: var(--text-secondary); }
    .model-badge { font-size: 0.7rem; font-family: var(--font-mono); background: var(--surface-active); color: var(--accent); border-radius: 3px; padding: 2px 6px; font-weight: 600; }
    .logout-btn { background: var(--bg-surface); border: 1px solid var(--border); border-radius: 4px; color: var(--text-muted); padding: 0.375rem 0.75rem; font-size: 0.8rem; font-family: var(--font-primary); cursor: pointer; transition: all 0.15s; }
    .logout-btn:hover { background: var(--surface-hover); color: var(--text-primary); }
  `]
})
export class AppShellComponent implements OnInit {
  constructor(public auth: AuthService, public pending: PendingRatingsService) {}

  ngOnInit(): void {
    this.pending.refresh();
  }
}
```

- [ ] **Step 2: Verify it builds**

Run (from `frontend/`): `npm run build`
Expected: build succeeds with no TypeScript errors. (It isn't wired into any route yet, so no visual check is possible until Task 12 — that's expected.)

- [ ] **Step 3: Commit**

```bash
git add frontend/src/app/shared/components/app-shell/app-shell.component.ts
git commit -m "feat: add AppShellComponent (shared nav for all /app pages)"
```

---

## Task 7: Frontend — `RateShellComponent`

**Files:**
- Create: `frontend/src/app/features/rate/rate-shell.component.ts`

**Interfaces:**
- Produces: `RateShellComponent`, selector `app-rate-shell`. Wired as the `/app/rate` route component in Task 12, with `RateRidesComponent` (Task 8) and `RateGeneratedComponent` (Task 9) as its children via `<router-outlet>`. Its `:host` is `flex:1; min-height:0` so it fills the space `AppShellComponent` gives it.

- [ ] **Step 1: Create the component**

```typescript
// LAYER: Component — Rate Section Shell
// PURPOSE: Owns the Rides/Generated sub-nav for the merged "Rate" section and
//          hosts the router-outlet for its two child views.

import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-rate-shell',
  standalone: true,
  imports: [CommonModule, RouterLink, RouterLinkActive, RouterOutlet],
  template: `
    <div class="rate-shell">
      <header class="rate-header">
        <h1 class="rate-title">Rate</h1>
        <div class="sub-nav">
          <a routerLink="rides" routerLinkActive="active" class="sub-tab" id="rate-tab-rides">Rides</a>
          <a routerLink="generated" routerLinkActive="active" class="sub-tab" id="rate-tab-generated">Generated Routes</a>
        </div>
      </header>
      <div class="rate-body">
        <router-outlet></router-outlet>
      </div>
    </div>
  `,
  styles: [`
    :host { display: flex; flex-direction: column; flex: 1; min-height: 0; overflow: hidden; }
    .rate-shell { display: flex; flex-direction: column; height: 100%; }
    .rate-header {
      display: flex; align-items: center; gap: 1.5rem;
      padding: 1.25rem 2rem 0;
    }
    .rate-title { font-family: var(--font-display); font-size: 1.5rem; font-weight: 800; margin: 0; color: var(--text-primary); }
    .sub-nav { display: flex; gap: 0.5rem; border-bottom: 1px solid var(--border); flex: 1; }
    .sub-tab {
      padding: 0.6rem 1rem; text-decoration: none; color: var(--text-muted);
      font-size: 0.9rem; font-weight: 600; border-bottom: 2px solid transparent;
      margin-bottom: -1px;
    }
    .sub-tab:hover { color: var(--text-primary); }
    .sub-tab.active { color: var(--accent); border-bottom-color: var(--accent); }
    .rate-body { flex: 1; min-height: 0; overflow-y: auto; }
  `]
})
export class RateShellComponent {}
```

- [ ] **Step 2: Verify it builds**

Run (from `frontend/`): `npm run build`
Expected: build succeeds with no TypeScript errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/app/features/rate/rate-shell.component.ts
git commit -m "feat: add RateShellComponent (Rides/Generated sub-nav)"
```

---

## Task 8: Frontend — move + rewrite `RateRidesComponent`

**Files:**
- Move: `frontend/src/app/features/rate-rides/rate-rides.component.ts` → `frontend/src/app/features/rate/rate-rides/rate-rides.component.ts`

**Interfaces:**
- Consumes: `RouteApiService` (`../../../core/services/route-api.service` from the new path), `PendingRatingsService.decrement()` from Task 5, `Ride.user_rating` (already existed).
- Produces: `RateRidesComponent`, selector `app-rate-rides`, now with `viewMode: 'pending' | 'rated'` and a "View rated" toggle. Wired as the `/app/rate/rides` route component in Task 12.

- [ ] **Step 1: Move the file**

```bash
git mv frontend/src/app/features/rate-rides/rate-rides.component.ts frontend/src/app/features/rate/rate-rides/rate-rides.component.ts
```

(This also removes the now-empty `frontend/src/app/features/rate-rides/` directory.)

- [ ] **Step 2: Replace the file's contents**

Overwrite `frontend/src/app/features/rate/rate-rides/rate-rides.component.ts` with:

```typescript
// LAYER: Component — Rate Rides (queue + rated history)
// PURPOSE: Rate un-rated Strava rides one at a time. A "View rated" toggle
//          switches to a read-only list of rides already rated. Nav/back-link
//          now lives in AppShellComponent + RateShellComponent's sub-nav.

import { Component, OnInit, AfterViewChecked, ElementRef, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import * as L from 'leaflet';
import { RouteApiService, Ride, RatingStats } from '../../../core/services/route-api.service';
import { PendingRatingsService } from '../../../core/services/pending-ratings.service';

function decodePolyline(encoded: string): [number, number][] {
  const points: [number, number][] = [];
  let idx = 0, lat = 0, lng = 0;
  while (idx < encoded.length) {
    let b, shift = 0, result = 0;
    do { b = encoded.charCodeAt(idx++) - 63; result |= (b & 0x1f) << shift; shift += 5; } while (b >= 0x20);
    lat += (result & 1) ? ~(result >> 1) : (result >> 1);
    shift = 0; result = 0;
    do { b = encoded.charCodeAt(idx++) - 63; result |= (b & 0x1f) << shift; shift += 5; } while (b >= 0x20);
    lng += (result & 1) ? ~(result >> 1) : (result >> 1);
    points.push([lat / 1e5, lng / 1e5]);
  }
  return points;
}

@Component({
  selector: 'app-rate-rides',
  standalone: true,
  imports: [CommonModule, RouterLink],
  template: `
    <div class="page">
      <div class="toolbar">
        <button class="toggle-view-btn" (click)="toggleView()" id="toggle-rated-view">
          {{ viewMode === 'pending' ? 'View rated (' + ratedRides.length + ')' : '← Back to rating' }}
        </button>
      </div>

      <div class="model-bar" *ngIf="stats">
        <span>Preference model v{{ stats.model_version }}</span>
        <span>{{ stats.total_ratings }} total ratings</span>
        <span *ngIf="stats.ratings_until_next_retrain > 0">
          {{ stats.ratings_until_next_retrain }} ratings until next retrain
        </span>
        <span *ngIf="stats.ratings_until_next_retrain === 0" class="retrain-soon">
          Retraining triggered!
        </span>
      </div>

      <ng-container *ngIf="viewMode === 'pending'">
        <div class="sync-row">
          <button class="sync-btn" (click)="syncRides()" [disabled]="syncing" id="sync-rides-btn">
            {{ syncing ? 'Syncing...' : 'Sync from Strava' }}
          </button>
          <span class="hint" *ngIf="syncMsg">{{ syncMsg }}</span>
        </div>

        <div class="ride-card" *ngIf="currentRide && !done">
          <div class="ride-header">
            <h2 class="ride-name">{{ currentRide.name }}</h2>
            <span class="ride-count">{{ currentIndex + 1 }} / {{ rides.length }}</span>
          </div>
          <div #rideMap class="ride-map" *ngIf="currentRide.polyline"></div>
          <div class="ride-stats">
            <div class="stat-box">
              <span class="stat-val">{{ currentRide.distance_mi | number:'1.1-1' }}</span>
              <span class="stat-label">miles</span>
            </div>
            <div class="stat-box">
              <span class="stat-val">{{ currentRide.elevation_ft | number:'1.0-0' }}</span>
              <span class="stat-label">ft gain</span>
            </div>
            <div class="stat-box">
              <span class="stat-val">{{ currentRide.moving_time_min | number:'1.0-0' }}</span>
              <span class="stat-label">min</span>
            </div>
            <div class="stat-box">
              <span class="stat-val">{{ currentRide.avg_speed_mph | number:'1.1-1' }}</span>
              <span class="stat-label">mph avg</span>
            </div>
          </div>
          <div class="rating-row">
            <span class="rating-label">How enjoyable was this ride?</span>
            <div class="rating-btns">
              <button *ngFor="let r of ratingValues"
                class="rating-btn"
                [class.selected]="selectedRating === r"
                [id]="'rating-' + r"
                (click)="rate(r)">
                {{ r }}
              </button>
            </div>
          </div>
          <div class="skip-row">
            <button class="skip-btn" (click)="skip()" id="skip-ride-btn">Skip (no rating)</button>
          </div>
        </div>

        <div class="done-state" *ngIf="!loading && rides.length === 0 && !done">
          <h2>No rides yet</h2>
          <p>Hit "Sync from Strava" to import your ride history.</p>
        </div>

        <div class="done-state" *ngIf="done && rides.length === 0 && totalRides > 0">
          <h2>All rides rated!</h2>
          <p>Your routes will personalize as you add more data.</p>
          <a routerLink="/app" class="back-btn">Back to Route Builder</a>
        </div>

        <div class="done-state" *ngIf="done && rides.length > 0">
          <h2>All done!</h2>
          <p>Your routes will personalize as you add more data.</p>
          <a routerLink="/app" class="back-btn">Back to Route Builder</a>
        </div>

        <div class="loading-state" *ngIf="loading">
          <div class="spinner"></div>
          <span>Loading rides...</span>
        </div>
      </ng-container>

      <div class="rated-list" *ngIf="viewMode === 'rated'">
        <div class="rated-empty" *ngIf="ratedRides.length === 0">No rated rides yet.</div>
        <div class="rated-row" *ngFor="let r of ratedRides">
          <span class="rated-name">{{ r.name }}</span>
          <span class="rated-date">{{ r.date | date:'mediumDate' }}</span>
          <span class="rated-score">{{ r.user_rating }}/10</span>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .page { min-height: 100%; background: var(--bg-primary); color: var(--text-primary); padding: 2rem; font-family: var(--font-primary); }
    .ride-name, .done-state h2 { font-family: var(--font-display); }

    .toolbar { display: flex; justify-content: flex-end; margin-bottom: 1rem; }
    .toggle-view-btn { background: none; border: 1px solid var(--border); border-radius: 6px; color: var(--text-muted); padding: 0.35rem 0.75rem; font-size: 0.8rem; font-family: var(--font-primary); cursor: pointer; }
    .toggle-view-btn:hover { color: var(--text-primary); border-color: var(--surface-active-border); }

    .model-bar {
      display: flex; gap: 1.5rem; align-items: center;
      background: var(--bg-surface); border: 1px solid var(--border);
      border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 1rem; font-size: 0.82rem; color: var(--text-muted);
    }
    .retrain-soon { color: var(--accent); font-weight: 600; }

    .sync-row { display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem; }
    .sync-btn {
      padding: 0.5rem 1rem; background: var(--bg-surface); border: 1px solid var(--border);
      border-radius: 8px; color: var(--text-secondary); font-size: 0.85rem; font-family: var(--font-primary); cursor: pointer;
    }
    .hint { font-size: 0.8rem; color: var(--text-dim); }

    .ride-card {
      max-width: 580px; margin: 0 auto;
      background: var(--bg-surface); border: 1px solid var(--border);
      border-radius: 16px; padding: 1.75rem; display: flex; flex-direction: column; gap: 1.25rem;
    }
    .ride-header { display: flex; justify-content: space-between; align-items: flex-start; }
    .ride-name { margin: 0; font-size: 1.15rem; font-weight: 700; color: var(--text-primary); }
    .ride-count { font-size: 0.8rem; color: var(--text-dim); flex-shrink: 0; }

    .ride-map {
      width: 100%; height: 250px; border-radius: 10px; overflow: hidden;
      border: 1px solid var(--border);
    }

    .ride-stats { display: flex; gap: 1.25rem; }
    .stat-box { display: flex; flex-direction: column; }
    .stat-val { font-size: 1.25rem; font-weight: 800; color: var(--text-primary); }
    .stat-label { font-size: 0.65rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; }

    .rating-label { font-size: 0.875rem; color: var(--text-muted); margin-bottom: 0.5rem; display: block; }
    .rating-btns { display: flex; gap: 0.4rem; flex-wrap: wrap; }
    .rating-btn {
      width: 42px; height: 42px; border-radius: 10px; border: 1px solid var(--border);
      background: var(--bg-surface); color: var(--text-secondary); font-size: 0.9rem; font-weight: 600;
      font-family: var(--font-primary); cursor: pointer; transition: all 0.12s;
    }
    .rating-btn:hover { background: var(--surface-active); border-color: var(--surface-active-border); color: var(--accent); }
    .rating-btn.selected { background: rgba(168, 71, 31, 0.16); border-color: var(--accent); color: var(--accent); }

    .skip-row { text-align: center; }
    .skip-btn { background: none; border: none; color: var(--text-dim); font-size: 0.8rem; font-family: var(--font-primary); cursor: pointer; }
    .skip-btn:hover { color: var(--text-muted); }

    .done-state, .loading-state {
      max-width: 400px; margin: 4rem auto; text-align: center;
      display: flex; flex-direction: column; align-items: center; gap: 1rem;
    }
    .done-state h2 { margin: 0; color: var(--text-primary); }
    .done-state p { color: var(--text-muted); margin: 0; }
    .back-btn {
      padding: 0.625rem 1.5rem; background: var(--accent); color: var(--on-accent);
      border-radius: 4px; text-decoration: none; font-weight: 600; font-size: 0.9rem;
      display: inline-block; box-shadow: 3px 3px 0 rgba(60, 46, 30, 0.25); transition: all 0.15s;
    }
    .back-btn:hover { background: var(--accent-hover); transform: translate(-1px, -1px); box-shadow: 4px 4px 0 rgba(60, 46, 30, 0.3); }
    .spinner { width: 32px; height: 32px; border: 3px solid rgba(168, 71, 31, 0.2); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }

    .rated-list { max-width: 580px; margin: 0 auto; display: flex; flex-direction: column; gap: 0.5rem; }
    .rated-empty { color: var(--text-muted); text-align: center; padding: 2rem; }
    .rated-row {
      display: flex; justify-content: space-between; align-items: center;
      background: var(--bg-surface); border: 1px solid var(--border); border-radius: 10px;
      padding: 0.75rem 1rem; font-size: 0.85rem;
    }
    .rated-name { color: var(--text-primary); font-weight: 600; }
    .rated-date { color: var(--text-dim); }
    .rated-score { color: var(--accent); font-weight: 700; }
  `]
})
export class RateRidesComponent implements OnInit, AfterViewChecked {
  @ViewChild('rideMap') mapEl!: ElementRef;

  allRides: Ride[] = [];
  rides: Ride[] = [];
  ratedRides: Ride[] = [];
  totalRides = 0;
  currentIndex = 0;
  selectedRating: number | null = null;
  stats: RatingStats | null = null;
  loading = true;
  syncing = false;
  syncMsg = '';
  done = false;
  viewMode: 'pending' | 'rated' = 'pending';
  ratingValues = [1,2,3,4,5,6,7,8,9,10];

  private map: L.Map | null = null;
  private renderedRideId: number | null = null;
  private submitting = false;

  get currentRide(): Ride | null {
    return this.rides[this.currentIndex] ?? null;
  }

  constructor(private api: RouteApiService, private pending: PendingRatingsService) {}

  ngOnInit(): void {
    this.loading = true;
    this.done = false;
    this.currentIndex = 0;
    this.viewMode = 'pending';
    this.api.getRatingStats().subscribe(s => this.stats = s);
    this.api.getRides().subscribe({
      next: rides => {
        this.allRides = rides;
        this.totalRides = rides.length;
        this.rides = rides.filter(r => r.user_rating == null);
        this.ratedRides = rides.filter(r => r.user_rating != null);
        this.loading = false;
        if (this.rides.length === 0 && this.totalRides > 0) {
          this.done = true;
        }
      },
      error: () => { this.loading = false; }
    });
  }

  ngAfterViewChecked(): void {
    const ride = this.currentRide;
    if (!ride || !ride.polyline || !this.mapEl) return;
    if (this.renderedRideId === ride.id) return;
    this.renderedRideId = ride.id;
    this.renderMap(ride.polyline);
  }

  private renderMap(polyline: string): void {
    if (this.map) {
      this.map.remove();
      this.map = null;
    }
    const points = decodePolyline(polyline);
    if (points.length === 0) return;

    this.map = L.map(this.mapEl.nativeElement, {
      zoomControl: false,
      attributionControl: false,
      dragging: false,
      scrollWheelZoom: false,
      doubleClickZoom: false,
      touchZoom: false,
    });

    L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
      maxZoom: 18,
    }).addTo(this.map);

    const line = L.polyline(points, { color: '#a8471f', weight: 3, opacity: 0.9 }).addTo(this.map);
    this.map.fitBounds(line.getBounds(), { padding: [20, 20] });
  }

  syncRides(): void {
    this.syncing = true;
    this.api.syncRides().subscribe({
      next: s => {
        this.syncMsg = `${s.new_rides} new rides synced (${s.total_rides} total)`;
        this.syncing = false;
        this.renderedRideId = null;
        this.ngOnInit();
      },
      error: () => { this.syncing = false; }
    });
  }

  rate(r: number): void {
    if (this.submitting) return;
    this.submitting = true;
    this.selectedRating = r;
    const ride = this.currentRide;
    if (!ride) return;

    this.api.rateRide(ride.id, r).subscribe({
      next: stats => {
        this.stats = stats;
        this.pending.decrement();
        this.ratedRides = [{ ...ride, user_rating: r }, ...this.ratedRides];
        setTimeout(() => {
          this.advance();
          this.submitting = false;
        }, 300);
      },
      error: () => { this.submitting = false; }
    });
  }

  skip(): void {
    this.advance();
  }

  toggleView(): void {
    this.viewMode = this.viewMode === 'pending' ? 'rated' : 'pending';
  }

  private advance(): void {
    this.selectedRating = null;
    this.renderedRideId = null;
    if (this.currentIndex + 1 >= this.rides.length) {
      this.done = true;
    } else {
      this.currentIndex++;
    }
  }
}
```

- [ ] **Step 3: Verify it builds**

Run (from `frontend/`): `npm run build`
Expected: build succeeds. (It's not wired into a route yet — that happens in Task 12 — so no visual check yet.)

- [ ] **Step 4: Commit**

```bash
git add frontend/src/app/features/rate/rate-rides/rate-rides.component.ts
git commit -m "refactor: move RateRidesComponent under features/rate, add view-rated toggle"
```

---

## Task 9: Frontend — move + rewrite `RateGeneratedComponent`

**Files:**
- Move: `frontend/src/app/features/rate-generated/rate-generated.component.ts` → `frontend/src/app/features/rate/rate-generated/rate-generated.component.ts`

**Interfaces:**
- Consumes: `RouteApiService` (`../../../core/services/route-api.service`), `PendingRatingsService.decrement()` from Task 5, `RouteResult.user_rating` from Task 4.
- Produces: `RateGeneratedComponent`, selector `app-rate-generated`, now filters to unrated-only by default and has a "View rated" toggle showing predicted-vs-actual. Wired as the `/app/rate/generated` route component in Task 12.

- [ ] **Step 1: Move the file**

```bash
git mv frontend/src/app/features/rate-generated/rate-generated.component.ts frontend/src/app/features/rate/rate-generated/rate-generated.component.ts
```

- [ ] **Step 2: Replace the file's contents**

Overwrite `frontend/src/app/features/rate/rate-generated/rate-generated.component.ts` with:

```typescript
// LAYER: Component — Rate Generated Routes (queue + rated history)
// PURPOSE: Rate un-rated generated routes one at a time, after riding them.
//          Shows predicted score vs actual rating after submission. A
//          "View rated" toggle switches to a read-only predicted-vs-actual
//          list of routes already rated. Nav now lives in AppShellComponent +
//          RateShellComponent's sub-nav.

import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { RouteApiService, RouteResult, RatingStats } from '../../../core/services/route-api.service';
import { PendingRatingsService } from '../../../core/services/pending-ratings.service';

@Component({
  selector: 'app-rate-generated',
  standalone: true,
  imports: [CommonModule, RouterLink],
  template: `
    <div class="page">
      <div class="toolbar">
        <button class="toggle-view-btn" (click)="toggleView()" id="toggle-rated-view-gen">
          {{ viewMode === 'pending' ? 'View rated (' + ratedRoutes.length + ')' : '← Back to rating' }}
        </button>
      </div>

      <div class="model-bar" *ngIf="stats">
        <span>Preference model v{{ stats.model_version }}</span>
        <span>{{ stats.total_ratings }} total ratings</span>
        <span *ngIf="stats.ratings_until_next_retrain > 0">
          {{ stats.ratings_until_next_retrain }} until next retrain
        </span>
      </div>

      <ng-container *ngIf="viewMode === 'pending'">
        <div class="route-card" *ngIf="current && !done && !submitted">
          <div class="card-header">
            <div>
              <h2 class="route-label">{{ current.route_type | titlecase }} Route</h2>
              <span class="predicted">Predicted score: <strong>{{ current.predicted_score }}/10</strong></span>
            </div>
            <span class="route-count">{{ currentIndex + 1 }} / {{ routes.length }}</span>
          </div>
          <div class="route-stats">
            <div class="stat-box">
              <span class="stat-val">{{ current.distance_mi | number:'1.1-1' }}</span>
              <span class="stat-label">miles</span>
            </div>
            <div class="stat-box">
              <span class="stat-val">{{ current.elevation_ft | number:'1.0-0' }}</span>
              <span class="stat-label">ft gain</span>
            </div>
            <div class="stat-box" *ngIf="current.novelty_pct != null">
              <span class="stat-val">{{ current.novelty_pct | number:'1.0-0' }}%</span>
              <span class="stat-label">new roads</span>
            </div>
          </div>
          <div class="rating-row">
            <span class="rating-label">How was this route after actually riding it?</span>
            <div class="rating-btns">
              <button *ngFor="let r of ratings"
                class="rating-btn"
                [class.selected]="selectedRating === r"
                [id]="'gen-rating-' + r"
                (click)="rate(r)">{{ r }}</button>
            </div>
          </div>
          <button class="skip-btn" (click)="skip()" id="skip-gen-btn">Skip</button>
        </div>

        <div class="comparison-card" *ngIf="submitted && lastRating != null">
          <h3>Rating Saved</h3>
          <div class="compare-row">
            <div class="compare-item">
              <span class="compare-val">{{ lastPredicted }}</span>
              <span class="compare-label">Predicted</span>
            </div>
            <div class="compare-arrow">&rarr;</div>
            <div class="compare-item">
              <span class="compare-val actual">{{ lastRating }}</span>
              <span class="compare-label">Your Rating</span>
            </div>
          </div>
          <p class="compare-note" *ngIf="diff > 1.5">This difference will improve future routes.</p>
          <button class="next-btn" (click)="nextAfterSubmit()" id="next-route-btn">Next Route &rarr;</button>
        </div>

        <div class="done-state" *ngIf="done && !submitted">
          <h2>All routes reviewed</h2>
          <a routerLink="/app" class="back-btn">Back to Route Builder</a>
        </div>

        <div class="loading-state" *ngIf="loading">
          <div class="spinner"></div>
          <span>Loading route history...</span>
        </div>
      </ng-container>

      <div class="rated-list" *ngIf="viewMode === 'rated'">
        <div class="rated-empty" *ngIf="ratedRoutes.length === 0">No rated routes yet.</div>
        <div class="rated-row" *ngFor="let r of ratedRoutes">
          <span class="rated-type">{{ r.route_type | titlecase }} · {{ r.distance_mi | number:'1.1-1' }} mi</span>
          <span class="rated-compare">
            <span class="rated-predicted">{{ r.predicted_score }}</span>
            <span class="rated-arrow">&rarr;</span>
            <span class="rated-actual">{{ r.user_rating }}</span>
          </span>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .page { min-height: 100%; background: var(--bg-primary); color: var(--text-primary); padding: 2rem; font-family: var(--font-primary); }
    .route-label, .comparison-card h3, .done-state h2 { font-family: var(--font-display); }

    .toolbar { display: flex; justify-content: flex-end; margin-bottom: 1rem; }
    .toggle-view-btn { background: none; border: 1px solid var(--border); border-radius: 6px; color: var(--text-muted); padding: 0.35rem 0.75rem; font-size: 0.8rem; font-family: var(--font-primary); cursor: pointer; }
    .toggle-view-btn:hover { color: var(--text-primary); border-color: var(--surface-active-border); }

    .model-bar {
      display: flex; gap: 1.5rem;
      background: var(--bg-surface); border: 1px solid var(--border);
      border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 1.5rem;
      font-size: 0.82rem; color: var(--text-muted);
    }

    .route-card {
      max-width: 580px; margin: 0 auto;
      background: var(--bg-surface); border: 1px solid var(--border);
      border-radius: 16px; padding: 1.75rem; display: flex; flex-direction: column; gap: 1.25rem;
    }
    .card-header { display: flex; justify-content: space-between; align-items: flex-start; }
    .route-label { margin: 0 0 0.25rem; font-size: 1.15rem; font-weight: 700; color: var(--text-primary); }
    .predicted { font-size: 0.82rem; color: var(--text-muted); }
    .predicted strong { color: var(--accent); }
    .route-count { font-size: 0.8rem; color: var(--text-dim); }

    .route-stats { display: flex; gap: 1.25rem; }
    .stat-box { display: flex; flex-direction: column; }
    .stat-val { font-size: 1.25rem; font-weight: 800; color: var(--text-primary); }
    .stat-label { font-size: 0.65rem; color: var(--text-dim); text-transform: uppercase; }

    .rating-label { font-size: 0.875rem; color: var(--text-muted); display: block; margin-bottom: 0.5rem; }
    .rating-btns { display: flex; gap: 0.4rem; flex-wrap: wrap; }
    .rating-btn {
      width: 42px; height: 42px; border-radius: 10px; border: 1px solid var(--border);
      background: var(--bg-surface); color: var(--text-secondary); font-size: 0.9rem; font-weight: 600;
      font-family: var(--font-primary); cursor: pointer; transition: all 0.12s;
    }
    .rating-btn:hover { background: var(--surface-active); border-color: var(--surface-active-border); color: var(--accent); }
    .rating-btn.selected { background: rgba(168, 71, 31, 0.16); border-color: var(--accent); color: var(--accent); }

    .skip-btn { background: none; border: none; color: var(--text-dim); font-size: 0.8rem; font-family: var(--font-primary); cursor: pointer; align-self: center; }

    .comparison-card {
      max-width: 400px; margin: 2rem auto; text-align: center;
      background: var(--bg-surface); border: 1px solid var(--border);
      border-radius: 16px; padding: 2rem; display: flex; flex-direction: column; gap: 1.25rem; align-items: center;
    }
    .comparison-card h3 { margin: 0; color: var(--text-primary); }
    .compare-row { display: flex; align-items: center; gap: 1.5rem; }
    .compare-item { display: flex; flex-direction: column; align-items: center; }
    .compare-val { font-size: 2.5rem; font-weight: 800; color: var(--text-muted); }
    .compare-val.actual { color: var(--accent); }
    .compare-arrow { font-size: 1.5rem; color: var(--text-dim); }
    .compare-label { font-size: 0.7rem; color: var(--text-dim); text-transform: uppercase; }
    .compare-note { color: var(--text-muted); font-size: 0.85rem; margin: 0; }
    .next-btn { padding: 0.625rem 1.5rem; background: var(--accent); color: var(--on-accent); border: none; border-radius: 4px; font-weight: 600; font-family: var(--font-primary); cursor: pointer; box-shadow: 3px 3px 0 rgba(60, 46, 30, 0.25); transition: all 0.15s; }
    .next-btn:hover { background: var(--accent-hover); transform: translate(-1px, -1px); box-shadow: 4px 4px 0 rgba(60, 46, 30, 0.3); }

    .done-state, .loading-state {
      max-width: 400px; margin: 4rem auto; text-align: center;
      display: flex; flex-direction: column; align-items: center; gap: 1rem;
    }
    .done-state h2 { margin: 0; color: var(--text-primary); }
    .back-btn { padding: 0.625rem 1.5rem; background: var(--accent); color: var(--on-accent); border-radius: 4px; text-decoration: none; font-weight: 600; display: inline-block; box-shadow: 3px 3px 0 rgba(60, 46, 30, 0.25); transition: all 0.15s; }
    .back-btn:hover { background: var(--accent-hover); transform: translate(-1px, -1px); box-shadow: 4px 4px 0 rgba(60, 46, 30, 0.3); }
    .spinner { width: 32px; height: 32px; border: 3px solid rgba(168, 71, 31, 0.2); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }

    .rated-list { max-width: 580px; margin: 0 auto; display: flex; flex-direction: column; gap: 0.5rem; }
    .rated-empty { color: var(--text-muted); text-align: center; padding: 2rem; }
    .rated-row {
      display: flex; justify-content: space-between; align-items: center;
      background: var(--bg-surface); border: 1px solid var(--border); border-radius: 10px;
      padding: 0.75rem 1rem; font-size: 0.85rem;
    }
    .rated-type { color: var(--text-primary); font-weight: 600; }
    .rated-compare { display: flex; align-items: center; gap: 0.5rem; font-family: var(--font-mono); }
    .rated-predicted { color: var(--text-muted); }
    .rated-arrow { color: var(--text-dim); }
    .rated-actual { color: var(--accent); font-weight: 700; }
  `]
})
export class RateGeneratedComponent implements OnInit {
  allRoutes: RouteResult[] = [];
  routes: RouteResult[] = [];
  ratedRoutes: RouteResult[] = [];
  currentIndex = 0;
  selectedRating: number | null = null;
  stats: RatingStats | null = null;
  loading = true;
  done = false;
  submitted = false;
  lastRating: number | null = null;
  lastPredicted = 0;
  viewMode: 'pending' | 'rated' = 'pending';
  ratings = [1,2,3,4,5,6,7,8,9,10];

  get current(): RouteResult | null { return this.routes[this.currentIndex] ?? null; }
  get diff(): number { return Math.abs((this.lastRating ?? 0) - this.lastPredicted); }

  constructor(private api: RouteApiService, private pending: PendingRatingsService) {}

  ngOnInit(): void {
    this.viewMode = 'pending';
    this.api.getRatingStats().subscribe(s => this.stats = s);
    this.api.getRouteHistory().subscribe({
      next: routes => {
        this.allRoutes = routes;
        this.routes = routes.filter(r => r.user_rating == null);
        this.ratedRoutes = routes.filter(r => r.user_rating != null);
        this.loading = false;
        this.done = this.routes.length === 0;
      },
      error: () => { this.loading = false; }
    });
  }

  rate(r: number): void {
    const route = this.current;
    if (!route) return;
    this.lastPredicted = route.predicted_score;
    this.lastRating = r;
    this.api.rateGeneratedRoute(route.id, r).subscribe({
      next: stats => {
        this.stats = stats;
        this.submitted = true;
        this.pending.decrement();
        this.ratedRoutes = [{ ...route, user_rating: r }, ...this.ratedRoutes];
      }
    });
  }

  skip(): void { this.advance(); }

  nextAfterSubmit(): void {
    this.submitted = false;
    this.lastRating = null;
    this.advance();
  }

  toggleView(): void {
    this.viewMode = this.viewMode === 'pending' ? 'rated' : 'pending';
  }

  private advance(): void {
    this.selectedRating = null;
    if (this.currentIndex + 1 >= this.routes.length) {
      this.done = true;
    } else {
      this.currentIndex++;
    }
  }
}
```

- [ ] **Step 3: Verify it builds**

Run (from `frontend/`): `npm run build`
Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/app/features/rate/rate-generated/rate-generated.component.ts
git commit -m "refactor: move RateGeneratedComponent under features/rate, filter unrated, add view-rated toggle"
```

---

## Task 10: Frontend — simplify `RouteBuilderComponent`

**Files:**
- Modify: `frontend/src/app/features/route-builder/route-builder.component.ts`

**Interfaces:**
- Produces: `RouteBuilderComponent` with its own nav and the map-building banner removed (both now live in `AppShellComponent`, Task 6). `:host` becomes `flex:1; min-height:0` to fill the space `AppShellComponent` gives it. Wired as the `/app` (index) route component in Task 12.

- [ ] **Step 1: Replace the file's contents**

Overwrite `frontend/src/app/features/route-builder/route-builder.component.ts` with:

```typescript
// LAYER: Component — Route Builder (Main App Page)
// PURPOSE: Assembles the city-selector, route-form sidebar, map-view, and
//          route-results panel for the main route generation experience.
//          Nav/chrome now lives in AppShellComponent — this component only
//          owns the builder's own layout.

import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouteStateService } from '../../core/services/route-state.service';
import { CitySelectorComponent } from '../../shared/components/city-selector/city-selector.component';
import { RouteFormComponent } from './route-form/route-form.component';
import { RouteResultsComponent } from './route-results/route-results.component';
import { MapViewComponent } from '../map-view/map-view.component';
import { LoadingOverlayComponent } from '../../shared/components/loading-overlay/loading-overlay.component';

@Component({
  selector: 'app-route-builder',
  standalone: true,
  imports: [
    CommonModule,
    CitySelectorComponent,
    RouteFormComponent,
    RouteResultsComponent,
    MapViewComponent,
    LoadingOverlayComponent,
  ],
  template: `
    <div class="main-layout">
      <aside class="sidebar">
        <app-city-selector></app-city-selector>
        <app-route-form></app-route-form>
        <app-route-results></app-route-results>
      </aside>

      <main class="map-area">
        <app-loading-overlay *ngIf="state.loading$ | async"></app-loading-overlay>
        <app-map-view></app-map-view>
      </main>
    </div>
  `,
  styles: [`
    :host { display: flex; flex: 1; min-height: 0; overflow: hidden; }
    .main-layout { display: flex; flex: 1; overflow: hidden; }

    .sidebar {
      width: 320px; min-width: 320px;
      background: rgba(60, 46, 30, 0.03);
      border-right: 1px solid var(--border);
      display: flex; flex-direction: column; overflow-y: auto;
      padding: 1rem;
      gap: 1rem;
    }
    .map-area { flex: 1; position: relative; }
  `]
})
export class RouteBuilderComponent {
  constructor(public state: RouteStateService) {}
}
```

- [ ] **Step 2: Verify it builds**

Run (from `frontend/`): `npm run build`
Expected: build succeeds.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/app/features/route-builder/route-builder.component.ts
git commit -m "refactor: remove RouteBuilderComponent's own nav (now in AppShellComponent)"
```

---

## Task 11: Frontend — remove `CoverageComponent`'s own topnav

**Files:**
- Modify: `frontend/src/app/features/coverage/coverage.component.ts`

**Interfaces:**
- Produces: `CoverageComponent` with its own topnav removed (nav now lives in `AppShellComponent`, Task 6). `:host` becomes `flex:1; min-height:0`. Wired as the `/app/coverage` route component in Task 12. `AuthService` stays injected (still used for `auth.currentUser()` to read `home_lat`/`home_lng`) — only the `logout()` call and its button are removed.

- [ ] **Step 1: Remove the `RouterLink` import**

Find:

```typescript
import { RouterLink } from '@angular/router';
```

Delete this line entirely.

- [ ] **Step 2: Remove `RouterLink` from the imports array**

Find:

```typescript
  imports: [CommonModule, FormsModule, RouterLink],
```

Replace with:

```typescript
  imports: [CommonModule, FormsModule],
```

- [ ] **Step 3: Remove the topnav from the template**

Find:

```typescript
  template: `
    <div class="shell">
      <nav class="topnav">
        <span class="brand">RouteMaker</span>
        <div class="nav-links">
          <a routerLink="/app" class="nav-link">← Builder</a>
          <button class="logout-btn" (click)="auth.logout()">Sign Out</button>
        </div>
      </nav>

      <div class="body">
```

Replace with:

```typescript
  template: `
    <div class="shell">
      <div class="body">
```

- [ ] **Step 4: Simplify the nav-related styles**

Find:

```typescript
    :host { display: block; height: 100vh; overflow: hidden; }
    .shell { display: flex; flex-direction: column; height: 100vh; background: var(--bg-primary); color: var(--text-primary); font-family: var(--font-primary); }
    .topnav { display: flex; align-items: center; justify-content: space-between; padding: 0 1.5rem; height: 56px; min-height: 56px; background: rgba(224,213,184,0.95); border-bottom: 1px solid var(--border); }
    .brand { font-family: var(--font-display); font-size: 1.25rem; font-weight: 700; }
    .nav-links { display: flex; align-items: center; gap: 1rem; }
    .nav-link { color: var(--text-muted); text-decoration: none; font-size: 0.875rem; }
    .nav-link:hover { color: var(--text-primary); }
    .logout-btn { background: var(--bg-surface); border: 1px solid var(--border); border-radius: 8px; color: var(--text-muted); padding: 0.375rem 0.75rem; font-size: 0.8rem; cursor: pointer; }
    .body { display: flex; flex: 1; overflow: hidden; }
```

Replace with:

```typescript
    :host { display: flex; flex: 1; min-height: 0; overflow: hidden; }
    .shell { display: flex; flex-direction: column; height: 100%; width: 100%; background: var(--bg-primary); color: var(--text-primary); font-family: var(--font-primary); }
    .body { display: flex; flex: 1; overflow: hidden; }
```

- [ ] **Step 5: Verify it builds**

Run (from `frontend/`): `npm run build`
Expected: build succeeds.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/app/features/coverage/coverage.component.ts
git commit -m "refactor: remove CoverageComponent's own topnav (now in AppShellComponent)"
```

---

## Task 12: Frontend — wire up the new route tree

**Files:**
- Modify: `frontend/src/app/app.routes.ts`

**Interfaces:**
- Consumes: `AppShellComponent` (Task 6), `RateShellComponent` (Task 7), `RateRidesComponent` (Task 8), `RateGeneratedComponent` (Task 9), `RouteBuilderComponent` (Task 10), `CoverageComponent` (Task 11), existing `LandingComponent` and `authGuard` (unchanged).
- Produces: the final route tree described in the spec — `/app` (shell) → `''` (builder), `'rate'` (shell) → `''` (redirect to `rides`), `'rides'`, `'generated'`; `'coverage'`.

- [ ] **Step 1: Replace the file's contents**

Overwrite `frontend/src/app/app.routes.ts` with:

```typescript
import { Routes } from '@angular/router';
import { authGuard } from './core/guards/auth.guard';

export const routes: Routes = [
  // Public: landing / login page
  {
    path: '',
    loadComponent: () =>
      import('./features/landing/landing.component').then(m => m.LandingComponent),
  },
  // OAuth callback — reads JWT from URL fragment and stores it
  {
    path: 'auth/callback',
    loadComponent: () =>
      import('./features/landing/landing.component').then(m => m.LandingComponent),
  },
  // Protected: app shell (persistent nav) + its three sections
  {
    path: 'app',
    canActivate: [authGuard],
    loadComponent: () =>
      import('./shared/components/app-shell/app-shell.component').then(m => m.AppShellComponent),
    children: [
      // Main route builder (default landing page under /app)
      {
        path: '',
        loadComponent: () =>
          import('./features/route-builder/route-builder.component').then(m => m.RouteBuilderComponent),
      },
      // Rate section: sub-nav shell + Rides / Generated Routes children
      {
        path: 'rate',
        loadComponent: () =>
          import('./features/rate/rate-shell.component').then(m => m.RateShellComponent),
        children: [
          { path: '', redirectTo: 'rides', pathMatch: 'full' },
          {
            path: 'rides',
            loadComponent: () =>
              import('./features/rate/rate-rides/rate-rides.component').then(m => m.RateRidesComponent),
          },
          {
            path: 'generated',
            loadComponent: () =>
              import('./features/rate/rate-generated/rate-generated.component').then(m => m.RateGeneratedComponent),
          },
        ],
      },
      // Street coverage map
      {
        path: 'coverage',
        loadComponent: () =>
          import('./features/coverage/coverage.component').then(m => m.CoverageComponent),
      },
    ],
  },
  // Fallback
  { path: '**', redirectTo: '' },
];
```

- [ ] **Step 2: Verify it builds**

Run (from `frontend/`): `npm run build`
Expected: build succeeds with no errors — this is the first point where all the new components are actually reachable, so this is also the first meaningful compile-time check that everything wires together correctly.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/app/app.routes.ts
git commit -m "feat: restructure /app routes around AppShellComponent and RateShellComponent"
```

---

## Task 13: Manual end-to-end verification

**Files:** none (verification only).

- [ ] **Step 1: Start the backend**

From the repo root, with the backend venv active:

```bash
.\backend\.venv\Scripts\Activate.ps1
python -m backend
```

Expected: serves on `:8000` with no startup errors.

- [ ] **Step 2: Start the frontend**

In a second terminal:

```bash
cd frontend
npm start
```

Expected: serves on `:4200` with no compile errors.

- [ ] **Step 3: Walk through the nav**

In a browser, log in via Strava and confirm:
- The shell nav shows **Builder**, **Rate**, **Coverage** as peer links, plus user info and Sign Out.
- Landing on `/app` shows the route builder with no page-level nav duplicated inside it.
- Clicking **Rate** navigates to `/app/rate/rides` (via the redirect) and shows a **Rides | Generated Routes** sub-nav.
- Clicking **Generated Routes** navigates to `/app/rate/generated`.
- Clicking **Coverage** navigates to `/app/coverage` and shows no leftover "← Builder" link.
- The active nav link (Builder/Rate/Coverage) is visually highlighted on each section, and only one at a time.

- [ ] **Step 4: Walk through the pending badge**

- If there are unrated rides or generated routes, confirm the **Rate** nav link shows a numeric badge equal to the combined count.
- Rate one ride (or skip through to one if none are unrated — sync from Strava first if needed) and confirm the badge count decrements immediately without a page reload.
- Rate all remaining unrated items and confirm the badge disappears entirely (not "Rate 0").

- [ ] **Step 5: Walk through "View rated"**

- On `/app/rate/rides`, click **View rated** and confirm it shows a read-only list of previously-rated rides with name/date/rating. Click **← Back to rating** to return to the queue.
- On `/app/rate/generated`, click **View rated** and confirm it shows a read-only list with predicted-vs-actual per route. Click **← Back to rating** to return.
- Rate a new ride or route and confirm it immediately appears in that page's "View rated" list without needing a refresh.

- [ ] **Step 6: Confirm no regressions**

- Generate a route from the builder and confirm it still works end-to-end (map renders, GPX download works).
- On Coverage, confirm the map, home-base flow, and Sign Out (via the shell nav) still work.

No commit for this task — it's verification only.
