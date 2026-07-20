# RouteMaker Frontend Redesign — "Field Guide" Visual Identity

## Context

The Angular frontend currently uses a dark "topo/cartographic" theme (earth
tones, amber accent `#c8915a`, Inter font) but the actual components — cards,
grid buttons, sliders, nav bar — are generic dark-SaaS patterns. Goal: give
RouteMaker a distinctive, non-boilerplate visual identity across the whole
app (landing page + all in-app pages), chosen and validated interactively
with the user via mockups.

## Direction: Field Guide / Trail Poster, Full Paper Shell

Warm, paper/print-poster aesthetic — like a printed cycling zine or national
park trail map — applied to the entire app shell (nav, sidebar, cards), not
just the landing page. Confirmed with the user across four rounds of visual
comparison:

1. Chose "Field Guide" over three alternatives (Topo Contour+, GPS Cycling
   Computer, Brutalist Blueprint).
2. Chose "Full Paper Shell" (cream nav/sidebar, map as part of the page) over
   "Dark Shell with Paper Accents".
3. Chose the deeper/heavier paper palette over a lighter/airier variant, but
   flagged the dark map inset + neon-green route line as the one thing that
   felt wrong.
4. Resolved that by swapping the Leaflet basemap from CARTO's dark tile set
   to CARTO's warm "Voyager" tile set (same OSM data, free, drop-in URL
   change) — the map becomes paper-toned and sits directly in the layout
   with a hairline divider instead of a dark framed window. Route line
   becomes solid rust/terracotta instead of neon green.

## Color System

Replace the `:root` variables in `src/styles.css`:

| Token | Value | Use |
|---|---|---|
| `--paper` | `#e0d5b8` | App shell background (nav, page background) |
| `--paper-card` | `#efe7d8` | Sidebar, cards, control surfaces |
| `--paper-card-light` | `#f4ead4` | Text-on-accent (button labels) |
| `--ink` | `#3c2e1e` | Headings, primary text |
| `--ink-muted` | `#6b5637` | Body/secondary text |
| `--ink-label` | `#8a6a3a` | Labels, mono data readouts |
| `--accent` | `#a8471f` | Primary accent — buttons, active states, route lines, progress bars |
| `--accent-hover` | `#c05a2c` | Accent hover state |
| `--border` | `rgba(60,46,30,.15)` | Standard hairline border |
| `--border-subtle` | `rgba(60,46,30,.1)` | Lighter divider |
| `--strava-orange` | `#fc4c02` | Unchanged — Strava brand button only |

Dark surfaces are removed entirely except the Leaflet map tiles themselves
(which become warm-toned, not dark, per the resolution above). No page keeps
the old near-black background.

## Typography

- **Display / headings** (`h1`–`h4`, nav brand, card titles): `Fraunces`
  (serif, variable weight 500–900) — replaces Inter for headings only.
- **Body / UI text** (buttons, labels, nav links, body copy): `Inter` —
  unchanged, already loaded.
- **Data readouts** (mileage, elevation, percentages, mono badges/labels):
  `JetBrains Mono` — new addition, used the way the current app uses
  `--font-primary` for stat values.

Add `Fraunces` and `JetBrains Mono` to the existing Google Fonts `@import` in
`src/styles.css`.

## Component Treatments

- **Top nav**: paper background (`--paper`), serif brand wordmark, unchanged
  structure/links.
- **Sidebar** (route form, city selector, results): `--paper-card` surface,
  hairline border, labels in mono uppercase, sliders with `--accent` thumb on
  a muted brown track.
- **Buttons** (primary/generate/CTA): `--accent` background, `--paper-card-light`
  text, small hard offset shadow (`3px 3px 0 rgba(60,46,30,.25)`) — the
  "stamped/letterpress" look validated in mockups — sharp-ish corners
  (2–4px radius), not the current large rounded pill style.
- **Route cards**: `--paper-card` surface, small stamp-style badge
  (route type, e.g. "HILLY LOOP") in dark ink pinned to the top-left corner,
  Fraunces title, mono stats row (distance / elevation / match %), rust
  progress bar for match score, secondary+primary action buttons.
- **Mode/chip selectors** (route type grid, etc.): paper-card chips, active
  state = dark ink fill (nav/sidebar context) or accent fill (on-map
  floating context).
- **Map**: Leaflet tile layer changes from
  `https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png` to
  `https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png`.
  Route polylines/markers drawn on the map switch from lime-green to
  `--accent` rust. Map area gets a hairline top/left border instead of
  padding + dark inset framing.
- **Decorative motifs** (used sparingly, not on every element): small
  uppercase mono "field note" labels (e.g. landing page eyebrow text), stamp
  badges on cards, ✕-mark list bullets on the landing feature list (replacing
  the current dot markers).

## Pages in scope

All pages get the same paper shell system (per the "full visual identity
overhaul, all pages" scope decision):

- Landing (`landing.component.ts`) — hero, feature list, Strava CTA, error
  states.
- Route builder shell (`route-builder.component.ts`) — nav, sidebar
  container.
- Route form (`route-form.component.ts`) — sliders, mode grid, generate
  button.
- Route results / route card (`route-results.component.ts`,
  `route-card.component.ts`).
- Map view (`map-view.component.ts`) — tile layer swap, route line color,
  floating controls if any.
- City selector, loading overlay, map-building banner (shared components).
- Rate rides, rate generated, coverage pages — same paper shell applied for
  visual consistency, using existing layouts (no mockups were produced for
  these specifically; apply the same color/typography/component tokens).

## Out of scope

- No change to app structure, routing, or functionality.
- No change to the Strava OAuth flow or backend.
- No light/dark mode toggle — single paper theme replaces the single dark
  theme.
- Coverage map's own data-visualization layer (heatmap/street coloring, if
  any) is not redesigned beyond taking the new base tile layer and accent
  colors — its specific rendering logic is out of scope.

## Testing

Visual/manual only — no new automated tests. After implementation, run the
dev server and check each route (landing, /app, /app/rate-rides,
/app/rate-generated, /app/coverage) for the new theme, plus the OAuth error
states on landing and the map tile swap rendering correctly.
