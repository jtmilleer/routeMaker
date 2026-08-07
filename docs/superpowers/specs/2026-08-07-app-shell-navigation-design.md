# App Shell & Navigation Redesign

**Date:** 2026-08-07
**Status:** Approved, ready for implementation planning

## Problem

The four authenticated pages (`/app`, `/app/rate-rides`, `/app/rate-generated`,
`/app/coverage`) are flat sibling routes that each build their own top nav from
scratch. They've drifted out of sync:

- `route-builder` has a full nav with links to all three other pages.
- `rate-rides` and `rate-generated` only have a "back to builder" link — no way to
  jump between the two rating pages or to coverage.
- `coverage` only has "← Builder" and "Sign Out" — no link to either rating page.

There's also no signal anywhere that rating is a first-class, important part of the
app (it drives model personalization) rather than a buried afterthought, and no way
for a user to see at a glance whether they have rides or generated routes waiting to
be rated.

Separately, `RateGeneratedComponent` has a latent gap: `GET /api/routes/history`
returns the same routes on every visit regardless of rating status (no
rated/unrated distinction exists on that response at all), so the page never
actually narrows down to a queue the way `RateRidesComponent` does.

## Goals

- One consistent, persistent nav shell across all four sections instead of four
  independently drifting ones.
- Route Builder, Rate, and Coverage presented as three equal peer sections (Builder
  is just the default landing page — not weighted "more important" in the nav
  itself).
- Rate Rides and Rate Generated merge into one "Rate" section with two sub-views,
  since they're conceptually one activity (give feedback so the model improves).
- Surface pending work: a badge on the Rate tab showing how many rides + generated
  routes are waiting to be rated.
- Fix the underlying data gap so "unrated" is actually meaningful for generated
  routes, and let users go back and see what they've already rated (both rides and
  generated routes).

## Non-goals

- No changes to route generation UX, onboarding copy, or coverage's internal page
  layout — this pass is scoped to navigation, routing structure, and rating-list
  consistency.
- URL prefix stays `/app/*` — not renaming to top-level paths. The `/app` boundary
  (authenticated app vs. public landing) is real; only one place
  (`auth.service.ts:98`) hardcodes the redirect target, so this was cheap either way,
  but there's no functional reason to change it.

## Routing architecture

`/app` changes from four flat sibling routes into a real parent/child structure:

```
/app  (AppShellComponent, authGuard)       — persistent top nav + <router-outlet>
  ''            → RouteBuilderComponent      (default landing)
  'rate'        (RateShellComponent — Rides/Generated sub-nav + <router-outlet>)
      ''           → redirectTo 'rides', pathMatch: 'full'
      'rides'      → RateRidesComponent
      'generated'  → RateGeneratedComponent
  'coverage'    → CoverageComponent
```

All leaf components stay lazy-loaded via `loadComponent`, same as today. `authGuard`
moves to the single parent `/app` route instead of being repeated on every route.

## Component changes

### New: `AppShellComponent`

Owns the one nav that all sections share:

- Brand mark.
- Three peer links: Builder (`/app`), Rate (`/app/rate`, with the pending-count
  badge), Coverage (`/app/coverage`).
- `routerLinkActive` highlighting on all three; the Builder link uses
  `routerLinkActiveOptions: { exact: true }` so it doesn't stay highlighted while on
  `/app/rate/*` or `/app/coverage`.
- User avatar/name/model-version badge, Sign Out button (ported as-is from the
  current `route-builder` nav).
- `<app-map-building-banner>`, moved here from `RouteBuilderComponent`. Its backing
  state (`RouteStateService.graphBuilding$`) is an app-root singleton
  (`providedIn: 'root'`), so this is a safe move — graph-build progress stays visible
  no matter which section the user is on, instead of disappearing when they navigate
  away from the builder.
- `<router-outlet>` for the active section.

Exact file location to be decided during planning (likely
`shared/components/app-shell/`, since it's cross-section chrome rather than a
single-page feature).

### New: `RateShellComponent`

Owns the section-local sub-nav: a Rides / Generated pill toggle
(`routerLink="rides"` / `routerLink="generated"`, relative to `/app/rate`), plus a
`<router-outlet>` for the two sub-pages. This is the "shared top-level shell, but
each section still gets its own distinct nav" split — the pill toggle here is scoped
to Rate only and looks different from the top-level shell nav.

### `RouteBuilderComponent`

Drops its inline `<nav class="topnav">` and the `<app-map-building-banner>` (moved
to the shell). Keeps the sidebar + map-area layout; `:host` height styling adjusts
from `100vh` to fill the remaining space under the shell's nav instead of the full
viewport.

### `RateRidesComponent`

- Drops its own page nav (`← Back to Route Builder` link + title) — context now
  comes from the shell nav + Rate sub-nav.
- Default view unchanged: queue of unrated rides (`rides.filter(r => r.user_rating
  == null)`), rate/skip one at a time.
- New "View rated" toggle: switches to a read-only list of already-rated rides
  (name, date, the rating given). No editing from this view — just a way to look
  back.

### `RateGeneratedComponent`

- Drops its own page nav, same as above.
- **Behavior change:** now filters to unrated-only by default
  (`routes.filter(r => r.user_rating == null)`), matching `RateRidesComponent`,
  instead of re-showing all 20 routes from history every visit regardless of rating
  status.
- New "View rated" toggle: read-only list of already-rated generated routes, showing
  predicted-vs-actual per route (reuses the existing comparison-row styling from the
  post-rating confirmation card).

### `CoverageComponent`

Drops its own topnav (`← Builder` + Sign Out) entirely — the shell provides both
navigation and sign-out now.

## Backend changes

### New endpoint: `GET /api/ratings/pending-counts`

```
GET /api/ratings/pending-counts
→ { "unrated_rides": 4, "unrated_routes": 2 }
```

Two `COUNT` queries scoped to `athlete_id` — rides with no matching `RideRating` row,
generated routes with no matching `RouteFeedback` row. No full-row fetch; this is
purely for the nav badge, kept cheap and separate from the list endpoints. New
`PendingCounts` Pydantic schema in `schemas.py`, new handler in `routers/ratings.py`.

### `GET /api/routes/history` — add `user_rating`

`RouteResult` gets a new optional field: `user_rating: Optional[int]`. The handler in
`routers/routes.py` joins `RouteFeedback` for the athlete's routes and populates it,
mirroring the existing pattern in `routers/rides.py` (`ratings_map` built from
`RideRating`, applied via `ride_out.user_rating = ratings_map.get(r.id)`).

This is the field that makes "unrated" meaningful for generated routes at all — it
doesn't exist today.

## Data flow: pending-count badge

- `AppShellComponent` calls `getPendingCounts()` once on init, renders
  `unrated_rides + unrated_routes` as a single combined badge on the Rate link.
- Badge is hidden entirely when the combined count is 0 (never shows "Rate 0").
- When a rating is successfully submitted in either `RateRidesComponent` or
  `RateGeneratedComponent`, the shared count is decremented locally (client-side, no
  refetch) so the badge updates immediately. Skipping a ride/route does not
  decrement it.
- If the initial fetch fails, the badge is just omitted — never blocks the shell nav
  from rendering.
- State lives in a small shared service (exact shape TBD in planning — likely a
  signal holding `number | null`, exposing a `decrement()` method used by both rate
  sub-components).

## Edge cases

- New user, zero rides and zero generated routes: no badge; Rate section still fully
  usable, shows its existing "nothing to rate yet" empty state in both sub-views.
- Navigating directly to `/app/rate` lands on `/app/rate/rides` via redirect.
- Graph-build banner: unaffected by which section is active since its state is a
  root singleton — moving it to the shell is a pure visibility improvement, not a
  behavior change.

## Testing

No existing backend test suite and no meaningful frontend test coverage currently
exists (per `CLAUDE.md`) — this redesign doesn't change that baseline. Verification
is manual: exercise all three nav sections, the Rate sub-nav toggle, the pending
badge appearing/decrementing/disappearing, and the "view rated" toggles on both rate
sub-pages, in the browser.
