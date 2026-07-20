# Field Guide Frontend Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-theme the entire RouteMaker Angular frontend from its current dark "topo/cartographic" look to the approved "Field Guide / Trail Poster" paper aesthetic (warm cream/paper surfaces, rust accent, serif display type, mono data readouts, warm Leaflet basemap) across every page.

**Architecture:** The app's components almost universally read colors through CSS custom properties defined once in `src/styles.css` (`--bg-primary`, `--text-primary`, `--accent`, etc.) rather than hardcoding hex values. The redesign keeps every existing variable *name* and only changes its *value*, so most components re-theme automatically from a single file change. Each task after that fixes the handful of places that hardcode colors outside the variable system (map tile URL, polyline colors, gradients, a few inline hex fallbacks) and applies the new display/mono fonts to headings and data readouts.

**Tech Stack:** Angular 17 standalone components, inline component styles (no SCSS), Leaflet for maps, Google Fonts (Fraunces, Inter, JetBrains Mono).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-20-frontend-redesign-design.md` — follow its color table, typography rules, and component treatments exactly.
- No automated tests are added for this work — per the spec, verification is manual: run the dev server and visually check each affected route. Every task ends with a manual check step instead of a unit test.
- Do not change routing, component structure, services, or backend contracts — this is a visual-only pass.
- Do not touch `--strava-orange` (`#fc4c02`) or the Strava logo/button behavior.
- Reuse existing CSS custom property *names* wherever one already exists for the concept being changed; only introduce a new variable name when no existing one covers it (`--on-accent`, `--font-display`, `--font-mono`).
- New route/rank color palette (used identically in `map-view.component.ts` and `route-card.component.ts` so a given rank means the same color on the map and in the sidebar list):
  `['#a8471f', '#5f7a52', '#b8862c', '#5c6b73', '#8a4a5c']`
- "Stamped" button shape used on every primary/CTA button app-wide: `border-radius: 4px;` and `box-shadow: 3px 3px 0 rgba(60, 46, 30, 0.25);` (replaces the old large-radius pill + soft-glow-shadow look). Hover state: `transform: translate(-1px, -1px); box-shadow: 4px 4px 0 rgba(60, 46, 30, 0.3);`.

---

### Task 1: Global tokens, fonts, and base page styles

**Files:**
- Modify: `frontend/src/styles.css` (full `:root` block, font import, scrollbar, leaflet popup)

**Interfaces:**
- Produces: the full set of CSS custom properties every later task depends on:
  `--font-primary`, `--font-display`, `--font-mono`, `--bg-primary`, `--bg-secondary`, `--bg-surface`, `--text-primary`, `--text-secondary`, `--text-muted`, `--text-dim`, `--on-accent`, `--accent`, `--accent-hover`, `--accent-glow`, `--border`, `--border-subtle`, `--surface-hover`, `--surface-active`, `--surface-active-border`, `--strava-orange`.

- [ ] **Step 1: Replace the font import and `:root` token block**

Replace lines 1–39 of `frontend/src/styles.css` (the font `@import` through the end of the `:root { ... }` block) with:

```css
/* Global styles for RouteMaker Angular frontend */
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700;9..144,900&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* Leaflet map styles */
@import 'leaflet/dist/leaflet.css';

/* ── Design System: Field Guide / Trail Poster ─────────────────────── */
:root {
  /* Typography */
  --font-primary: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  --font-display: 'Fraunces', Georgia, serif;
  --font-mono: 'JetBrains Mono', 'SFMono-Regular', monospace;

  /* Backgrounds — warm paper/cream tones */
  --bg-primary: #e0d5b8;
  --bg-secondary: #efe7d8;
  --bg-surface: #efe7d8;

  /* Text hierarchy — ink on paper */
  --text-primary: #3c2e1e;
  --text-secondary: #6b5637;
  --text-muted: #8a6a3a;
  --text-dim: #a3906f;

  /* Text placed on top of --accent-colored surfaces (buttons, active chips) */
  --on-accent: #f4ead4;

  /* Accent — rust / terracotta */
  --accent: #a8471f;
  --accent-hover: #c05a2c;
  --accent-glow: rgba(168, 71, 31, 0.3);

  /* Borders — ink at low opacity */
  --border: rgba(60, 46, 30, 0.15);
  --border-subtle: rgba(60, 46, 30, 0.1);

  /* Interactive surfaces */
  --surface-hover: rgba(60, 46, 30, 0.08);
  --surface-active: rgba(168, 71, 31, 0.12);
  --surface-active-border: rgba(168, 71, 31, 0.35);

  /* Strava brand (only used on the login button) */
  --strava-orange: #fc4c02;
}
```

- [ ] **Step 2: Update the scrollbar and Leaflet popup rules to use ink-based tones**

Replace the scrollbar block:

```css
/* Custom scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(200, 190, 170, 0.03); }
::-webkit-scrollbar-thumb { background: rgba(200, 190, 170, 0.15); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(200, 190, 170, 0.25); }
```

with:

```css
/* Custom scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(60, 46, 30, 0.04); }
::-webkit-scrollbar-thumb { background: rgba(60, 46, 30, 0.18); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(60, 46, 30, 0.3); }
```

The Leaflet popup rules below the scrollbar block already use `var(--bg-secondary)`, `var(--border)`, and `var(--text-primary)` — leave them unchanged, they'll pick up the new paper tones automatically.

- [ ] **Step 3: Manual check**

Run `npm start` (or the project's existing dev-server command) from `frontend/`, open the app in a browser. Confirm: page background is now warm cream/tan instead of dark green-black, and no console errors about the Google Fonts request failing. Component text will look wrong/unstyled in places until later tasks land — that's expected at this point, just confirm the token change took effect (inspect `:root` in devtools and see the new `--bg-primary` value).

- [ ] **Step 4: Commit**

```bash
git add frontend/src/styles.css
git commit -m "style: retheme global tokens to Field Guide paper palette"
```

---

### Task 2: Landing page

**Files:**
- Modify: `frontend/src/app/features/landing/landing.component.ts`

**Interfaces:**
- Consumes: tokens from Task 1 (`--bg-primary`, `--text-primary`, `--text-muted`, `--bg-surface`, `--border`, `--accent`, `--font-primary`, `--font-display`, `--on-accent`, `--text-dim`).

- [ ] **Step 1: Replace the hardcoded dark gradient background**

In the `styles` array, replace:

```css
    .landing {
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      background: linear-gradient(145deg, var(--bg-primary) 0%, #182118 45%, #1a2418 100%);
      padding: 2rem;
    }
```

with:

```css
    .landing {
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      background: radial-gradient(ellipse at 70% 15%, #e8ddc0 0%, var(--bg-primary) 55%);
      padding: 2rem;
    }
```

- [ ] **Step 2: Switch the wordmark to the display serif and add the eyebrow label**

Replace:

```html
      <div class="hero">
        <div class="wordmark">RouteMaker</div>
        <p class="tagline">Custom cycling routes, built from your ride history.</p>
```

with:

```html
      <div class="hero">
        <div class="eyebrow">⟡ Field Notes · Custom Routes ⟡</div>
        <div class="wordmark">RouteMaker</div>
        <p class="tagline">Custom cycling routes, built from your ride history.</p>
```

Replace the `.wordmark` rule:

```css
    .wordmark {
      font-size: 3.5rem;
      font-weight: 800;
      color: var(--text-primary);
      letter-spacing: -1px;
      margin-bottom: 0.75rem;
      font-family: var(--font-primary);
    }
```

with:

```css
    .eyebrow {
      font-family: var(--font-mono);
      font-size: 0.7rem;
      font-weight: 600;
      letter-spacing: 0.15em;
      color: var(--text-muted);
      margin-bottom: 0.875rem;
    }
    .wordmark {
      font-size: 3.5rem;
      font-weight: 900;
      color: var(--text-primary);
      letter-spacing: -1px;
      margin-bottom: 0.75rem;
      font-family: var(--font-display);
    }
```

- [ ] **Step 3: Swap the feature-list dot markers for ✕ marks and re-theme the feature cards**

Replace:

```css
    .feature {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 0.875rem 1.25rem;
      color: var(--text-secondary);
      font-size: 0.95rem;
    }
    .feature-marker {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--accent);
      flex-shrink: 0;
    }
```

with:

```css
    .feature {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 3px;
      padding: 0.875rem 1.25rem;
      color: var(--text-secondary);
      font-size: 0.95rem;
    }
    .feature-marker {
      font-family: var(--font-mono);
      font-weight: 700;
      font-size: 0.7rem;
      color: var(--accent);
      flex-shrink: 0;
    }
    .feature-marker::before { content: '\2715'; }
```

- [ ] **Step 4: Give the Strava button the stamped shape**

Replace:

```css
    .strava-btn {
      display: inline-flex;
      align-items: center;
      gap: 0.625rem;
      background: var(--strava-orange);
      color: #fff;
      border: none;
      border-radius: 12px;
      padding: 1rem 2rem;
      font-size: 1.05rem;
      font-weight: 700;
      font-family: var(--font-primary);
      cursor: pointer;
      transition: transform 0.15s, box-shadow 0.15s;
      box-shadow: 0 4px 24px rgba(252, 76, 2, 0.35);
      width: 100%;
      justify-content: center;
    }
    .strava-btn:hover:not(:disabled) {
      transform: translateY(-2px);
      box-shadow: 0 8px 32px rgba(252, 76, 2, 0.5);
    }
```

with:

```css
    .strava-btn {
      display: inline-flex;
      align-items: center;
      gap: 0.625rem;
      background: var(--strava-orange);
      color: #fff;
      border: none;
      border-radius: 4px;
      padding: 1rem 2rem;
      font-size: 1.05rem;
      font-weight: 700;
      font-family: var(--font-primary);
      cursor: pointer;
      transition: transform 0.15s, box-shadow 0.15s;
      box-shadow: 3px 3px 0 rgba(60, 46, 30, 0.25);
      width: 100%;
      justify-content: center;
    }
    .strava-btn:hover:not(:disabled) {
      transform: translate(-1px, -1px);
      box-shadow: 4px 4px 0 rgba(60, 46, 30, 0.3);
    }
```

- [ ] **Step 5: Re-theme the auth error banner for a light background**

Replace:

```css
    .auth-error {
      background: rgba(200, 80, 60, 0.12);
      border: 1px solid rgba(200, 80, 60, 0.35);
      color: #e08a78;
      border-radius: 10px;
      padding: 0.75rem 1rem;
      margin-bottom: 1rem;
      font-size: 0.9rem;
    }
```

with:

```css
    .auth-error {
      background: rgba(178, 58, 46, 0.1);
      border: 1px solid rgba(178, 58, 46, 0.35);
      color: #8a3626;
      border-radius: 4px;
      padding: 0.75rem 1rem;
      margin-bottom: 1rem;
      font-size: 0.9rem;
    }
```

- [ ] **Step 6: Manual check**

Run the dev server, visit `/`. Confirm: warm paper background, serif "RouteMaker" wordmark, mono eyebrow label above it, ✕-mark feature list, stamped-shadow Strava button. Trigger an auth error (visit `/?auth_error=denied`) and confirm the error banner reads clearly on the light background.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/app/features/landing/landing.component.ts
git commit -m "style: retheme landing page to Field Guide paper look"
```

---

### Task 3: App shell — top nav and sidebar container

**Files:**
- Modify: `frontend/src/app/features/route-builder/route-builder.component.ts`

**Interfaces:**
- Consumes: tokens from Task 1.

- [ ] **Step 1: Re-theme the nav bar background and brand wordmark**

Replace:

```css
    .topnav {
      display: flex; align-items: center; justify-content: space-between;
      padding: 0 1.5rem; height: 56px; min-height: 56px;
      background: rgba(18, 26, 19, 0.95);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(12px);
      z-index: 100;
    }
    .nav-brand { display: flex; align-items: center; gap: 0.5rem; }
    .brand-name { font-size: 1.25rem; font-weight: 800; color: var(--text-primary); }
```

with:

```css
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
```

- [ ] **Step 2: Update the model-version badge and sign-out button shapes**

Replace:

```css
    .model-badge { font-size: 0.7rem; background: var(--surface-active); color: var(--accent); border-radius: 4px; padding: 2px 6px; font-weight: 600; }
    .logout-btn { background: var(--bg-surface); border: 1px solid var(--border); border-radius: 8px; color: var(--text-muted); padding: 0.375rem 0.75rem; font-size: 0.8rem; font-family: var(--font-primary); cursor: pointer; transition: all 0.15s; }
```

with:

```css
    .model-badge { font-size: 0.7rem; font-family: var(--font-mono); background: var(--surface-active); color: var(--accent); border-radius: 3px; padding: 2px 6px; font-weight: 600; }
    .logout-btn { background: var(--bg-surface); border: 1px solid var(--border); border-radius: 4px; color: var(--text-muted); padding: 0.375rem 0.75rem; font-size: 0.8rem; font-family: var(--font-primary); cursor: pointer; transition: all 0.15s; }
```

- [ ] **Step 3: Re-theme the sidebar surface**

Replace:

```css
    .sidebar {
      width: 320px; min-width: 320px;
      background: rgba(200, 190, 170, 0.03);
      border-right: 1px solid var(--border);
      display: flex; flex-direction: column; overflow-y: auto;
      padding: 1rem;
      gap: 1rem;
    }
```

with:

```css
    .sidebar {
      width: 320px; min-width: 320px;
      background: rgba(60, 46, 30, 0.03);
      border-right: 1px solid var(--border);
      display: flex; flex-direction: column; overflow-y: auto;
      padding: 1rem;
      gap: 1rem;
    }
```

- [ ] **Step 4: Manual check**

Log in and land on `/app`. Confirm: nav bar is paper-toned with a serif "RouteMaker" wordmark, sidebar background is a very light warm tint (not the old dark tint), model-version badge uses the mono font.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/app/features/route-builder/route-builder.component.ts
git commit -m "style: retheme app shell nav and sidebar to paper palette"
```

---

### Task 4: Route form (sliders, mode grid, generate button)

**Files:**
- Modify: `frontend/src/app/features/route-builder/route-form/route-form.component.ts`

**Interfaces:**
- Consumes: tokens from Task 1, including `--on-accent`.

- [ ] **Step 1: Switch the section title and labels to the mono font**

Replace:

```css
    .section-title { font-size: 0.875rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-dim); margin: 0 0 0.5rem; }
    .form-group { display: flex; flex-direction: column; gap: 0.375rem; }
    .label { font-size: 0.8rem; color: var(--text-muted); font-weight: 500; }
```

with:

```css
    .section-title { font-family: var(--font-mono); font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-dim); margin: 0 0 0.5rem; }
    .form-group { display: flex; flex-direction: column; gap: 0.375rem; }
    .label { font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); font-weight: 500; text-transform: uppercase; letter-spacing: 0.04em; }
```

- [ ] **Step 2: Square off the mode grid buttons**

Replace:

```css
    .mode-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.375rem; }
    .mode-btn {
      background: var(--bg-surface); border: 1px solid var(--border);
      border-radius: 8px; color: var(--text-muted); padding: 0.5rem; font-size: 0.8rem;
      font-family: var(--font-primary);
      cursor: pointer; transition: all 0.15s; text-align: center;
    }
```

with:

```css
    .mode-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.375rem; }
    .mode-btn {
      background: var(--bg-surface); border: 1px solid var(--border);
      border-radius: 3px; color: var(--text-muted); padding: 0.5rem; font-size: 0.8rem;
      font-family: var(--font-primary);
      cursor: pointer; transition: all 0.15s; text-align: center;
    }
```

- [ ] **Step 3: Give the generate button the stamped shape and correct on-accent text color**

Replace:

```css
    .generate-btn {
      width: 100%; padding: 0.875rem; background: var(--accent);
      border: none; border-radius: 10px; color: #fff; font-size: 0.95rem;
      font-weight: 700; font-family: var(--font-primary);
      cursor: pointer; transition: all 0.15s;
      box-shadow: 0 4px 16px var(--accent-glow);
      margin-top: 0.5rem;
    }
    .generate-btn:hover:not(:disabled) { background: var(--accent-hover); transform: translateY(-1px); box-shadow: 0 6px 24px var(--accent-glow); }
```

with:

```css
    .generate-btn {
      width: 100%; padding: 0.875rem; background: var(--accent);
      border: none; border-radius: 4px; color: var(--on-accent); font-size: 0.95rem;
      font-weight: 700; font-family: var(--font-primary);
      cursor: pointer; transition: all 0.15s;
      box-shadow: 3px 3px 0 rgba(60, 46, 30, 0.25);
      margin-top: 0.5rem;
    }
    .generate-btn:hover:not(:disabled) { background: var(--accent-hover); transform: translate(-1px, -1px); box-shadow: 4px 4px 0 rgba(60, 46, 30, 0.3); }
```

- [ ] **Step 4: Manual check**

On `/app`, confirm: "GENERATE ROUTE" section title and slider labels are in the mono font and uppercase, mode buttons have sharp corners, the Generate button has a hard offset shadow and cream (not white) text, and shifts on hover instead of just lifting.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/app/features/route-builder/route-form/route-form.component.ts
git commit -m "style: retheme route form controls to Field Guide look"
```

---

### Task 5: City selector

**Files:**
- Modify: `frontend/src/app/shared/components/city-selector/city-selector.component.ts`

**Interfaces:**
- Consumes: tokens from Task 1.

- [ ] **Step 1: Switch the label and city buttons to the mono font / square corners**

Replace:

```css
    .city-panel { display: flex; flex-direction: column; gap: 0.5rem; }
    .label { font-size: 0.875rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-dim); }
    .city-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.375rem; }
    .city-btn {
      display: flex; align-items: center; gap: 0.375rem;
      background: var(--bg-surface); border: 1px solid var(--border);
      border-radius: 8px; color: var(--text-muted); padding: 0.5rem 0.625rem; font-size: 0.78rem;
      font-family: var(--font-primary);
      cursor: pointer; transition: all 0.15s; text-align: left;
    }
```

with:

```css
    .city-panel { display: flex; flex-direction: column; gap: 0.5rem; }
    .label { font-family: var(--font-mono); font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-dim); }
    .city-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.375rem; }
    .city-btn {
      display: flex; align-items: center; gap: 0.375rem;
      background: var(--bg-surface); border: 1px solid var(--border);
      border-radius: 3px; color: var(--text-muted); padding: 0.5rem 0.625rem; font-size: 0.78rem;
      font-family: var(--font-primary);
      cursor: pointer; transition: all 0.15s; text-align: left;
    }
```

- [ ] **Step 2: Square off the coordinate inputs and "Set Location" button**

Replace:

```css
    .custom-inputs { display: flex; flex-direction: column; gap: 0.375rem; }
    .coord-input {
      width: 100%; padding: 0.5rem 0.75rem;
      background: var(--bg-surface); border: 1px solid var(--border);
      border-radius: 8px; color: var(--text-primary); font-size: 0.85rem;
      font-family: var(--font-primary);
    }
    .coord-input:focus { outline: none; border-color: var(--accent); }
    .set-btn {
      padding: 0.5rem; background: var(--surface-active); border: 1px solid var(--surface-active-border);
      border-radius: 8px; color: var(--accent); font-size: 0.82rem; font-weight: 600;
      font-family: var(--font-primary); cursor: pointer;
    }
```

with:

```css
    .custom-inputs { display: flex; flex-direction: column; gap: 0.375rem; }
    .coord-input {
      width: 100%; padding: 0.5rem 0.75rem;
      background: var(--bg-surface); border: 1px solid var(--border);
      border-radius: 3px; color: var(--text-primary); font-size: 0.85rem;
      font-family: var(--font-mono);
    }
    .coord-input:focus { outline: none; border-color: var(--accent); }
    .set-btn {
      padding: 0.5rem; background: var(--surface-active); border: 1px solid var(--surface-active-border);
      border-radius: 3px; color: var(--accent); font-size: 0.82rem; font-weight: 600;
      font-family: var(--font-primary); cursor: pointer;
    }
```

- [ ] **Step 3: Manual check**

On `/app`, confirm the "START LOCATION" label and city buttons use the new palette/mono label, click "Custom Pin" and confirm the lat/lng inputs render in mono font with square corners.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/app/shared/components/city-selector/city-selector.component.ts
git commit -m "style: retheme city selector to Field Guide look"
```

---

### Task 6: Route results list and route card

**Files:**
- Modify: `frontend/src/app/features/route-builder/route-results/route-results.component.ts`
- Modify: `frontend/src/app/shared/components/route-card/route-card.component.ts`

**Interfaces:**
- Consumes: tokens from Task 1.
- Produces: `RANK_COLORS` array `['#a8471f', '#5f7a52', '#b8862c', '#5c6b73', '#8a4a5c']` in `route-card.component.ts` — Task 7 (map-view) must use the identical array as `ROUTE_COLORS` so a route's rank color matches between the sidebar card and its map polyline.

- [ ] **Step 1: Mono-ize the results section title**

In `route-results.component.ts`, replace:

```css
    .section-title { font-size: 0.875rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-dim); margin: 0 0 0.25rem; }
```

with:

```css
    .section-title { font-family: var(--font-mono); font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-dim); margin: 0 0 0.25rem; }
```

- [ ] **Step 2: Update the route-card rank color palette**

In `route-card.component.ts`, replace:

```typescript
const RANK_COLORS = ['#c8915a', '#7a9a6d', '#a67c52', '#6b8f71', '#b8976a'];
```

with:

```typescript
const RANK_COLORS = ['#a8471f', '#5f7a52', '#b8862c', '#5c6b73', '#8a4a5c'];
```

- [ ] **Step 3: Give the card a paper-poster look — badge, serif score, square-ish corners**

Replace:

```css
    .card {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-left: 3px solid transparent;
      border-radius: 10px;
      padding: 0.75rem;
      cursor: pointer;
      transition: all 0.15s;
    }
    .card:hover { background: var(--surface-hover); }
    .card.selected { background: rgba(200, 190, 170, 0.1); border-color: rgba(200, 190, 170, 0.2); }

    .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
    .rank { font-weight: 800; font-size: 0.85rem; }
    .score { font-size: 1.25rem; font-weight: 800; color: var(--text-primary); }
    .score-max { font-size: 0.7rem; color: var(--text-dim); font-weight: 400; }

    .stats { display: flex; gap: 1rem; margin-bottom: 0.625rem; }
    .stat { display: flex; flex-direction: column; }
    .stat-val { font-size: 0.9rem; font-weight: 700; color: var(--text-primary); }
    .stat-label { font-size: 0.65rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; }
```

with:

```css
    .card {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-left: 3px solid transparent;
      border-radius: 4px;
      padding: 0.75rem;
      cursor: pointer;
      transition: all 0.15s;
    }
    .card:hover { background: var(--surface-hover); }
    .card.selected { background: rgba(60, 46, 30, 0.06); border-color: rgba(60, 46, 30, 0.25); }

    .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
    .rank { font-family: var(--font-mono); font-weight: 800; font-size: 0.85rem; }
    .score { font-family: var(--font-display); font-size: 1.3rem; font-weight: 700; color: var(--text-primary); }
    .score-max { font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-dim); font-weight: 400; }

    .stats { display: flex; gap: 1rem; margin-bottom: 0.625rem; }
    .stat { display: flex; flex-direction: column; }
    .stat-val { font-family: var(--font-mono); font-size: 0.9rem; font-weight: 700; color: var(--text-primary); }
    .stat-label { font-family: var(--font-mono); font-size: 0.65rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; }
```

- [ ] **Step 4: Re-theme the elevation chart fill and historic site tags for a light card**

Replace:

```css
    .elev-chart { width: 100%; height: 30px; display: block; margin-bottom: 0.625rem; overflow: visible; }
    .elev-line { fill: none; stroke-width: 1.25; vector-effect: non-scaling-stroke; opacity: 0.9; }
    .elev-area { fill: rgba(200, 145, 90, 0.12); stroke: none; }

    .historic-sites { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-bottom: 0.5rem; }
    .site-tag {
      font-size: 0.7rem; background: rgba(107, 143, 113, 0.15); color: #8fb896;
      border: 1px solid rgba(107, 143, 113, 0.3); border-radius: 6px; padding: 0.2rem 0.5rem;
      max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }

    .gpx-btn {
      width: 100%; padding: 0.375rem; background: var(--bg-surface);
      border: 1px solid var(--border); border-radius: 6px;
      color: var(--text-muted); font-size: 0.75rem; font-family: var(--font-primary);
      cursor: pointer; transition: all 0.15s;
    }
```

with:

```css
    .elev-chart { width: 100%; height: 30px; display: block; margin-bottom: 0.625rem; overflow: visible; }
    .elev-line { fill: none; stroke-width: 1.25; vector-effect: non-scaling-stroke; opacity: 0.9; }
    .elev-area { fill: rgba(168, 71, 31, 0.15); stroke: none; }

    .historic-sites { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-bottom: 0.5rem; }
    .site-tag {
      font-family: var(--font-mono); font-size: 0.68rem; background: rgba(95, 122, 82, 0.14); color: #45592f;
      border: 1px solid rgba(95, 122, 82, 0.35); border-radius: 3px; padding: 0.2rem 0.5rem;
      max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }

    .gpx-btn {
      width: 100%; padding: 0.375rem; background: var(--bg-surface);
      border: 1px solid var(--border); border-radius: 3px;
      color: var(--text-muted); font-size: 0.75rem; font-family: var(--font-mono);
      cursor: pointer; transition: all 0.15s;
    }
```

- [ ] **Step 5: Manual check**

Generate routes on `/app`. Confirm the sidebar route cards: serif score number, mono stat values/labels, rust-tinted elevation area fill, sage historic-site tags readable on the light card, and no more large rounded corners.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/app/features/route-builder/route-results/route-results.component.ts frontend/src/app/shared/components/route-card/route-card.component.ts
git commit -m "style: retheme route results list and route card to Field Guide look"
```

---

### Task 7: Map view — warm basemap and route colors

**Files:**
- Modify: `frontend/src/app/features/map-view/map-view.component.ts`

**Interfaces:**
- Consumes: `RANK_COLORS` palette produced in Task 6 — this task's `ROUTE_COLORS` must stay byte-identical to it.
- Produces: `ROUTE_COLORS = ['#a8471f', '#5f7a52', '#b8862c', '#5c6b73', '#8a4a5c']`.

- [ ] **Step 1: Update the route color palette**

Replace:

```typescript
const ROUTE_COLORS = ['#c8915a', '#7a9a6d', '#a67c52', '#6b8f71', '#b8976a'];
```

with:

```typescript
const ROUTE_COLORS = ['#a8471f', '#5f7a52', '#b8862c', '#5c6b73', '#8a4a5c'];
```

- [ ] **Step 2: Swap the dark CARTO tile layer for the warm Voyager tile layer**

Replace:

```typescript
    // Dark-style tile layer
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
      maxZoom: 19,
    }).addTo(this.map);
```

with:

```typescript
    // Warm paper-toned tile layer (Field Guide theme)
    L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
      maxZoom: 19,
    }).addTo(this.map);
```

- [ ] **Step 3: Re-theme the "already ridden" segment color and historic-site marker for the light basemap**

Replace:

```typescript
    const historicIcon = L.divIcon({
      className: 'historic-marker',
      html: '<div style="background:#6b8f71;color:#fff;border-radius:50%;width:24px;height:24px;display:flex;align-items:center;justify-content:center;font-size:14px;border:2px solid #c4bfb4;box-shadow:0 2px 6px rgba(0,0,0,0.4);">&#9733;</div>',
      iconSize: [24, 24],
      iconAnchor: [12, 12],
    });
```

with:

```typescript
    const historicIcon = L.divIcon({
      className: 'historic-marker',
      html: '<div style="background:#5f7a52;color:#efe7d8;border-radius:50%;width:24px;height:24px;display:flex;align-items:center;justify-content:center;font-size:14px;border:2px solid #efe7d8;box-shadow:0 2px 6px rgba(60,46,30,0.35);">&#9733;</div>',
      iconSize: [24, 24],
      iconAnchor: [12, 12],
    });
```

Replace:

```typescript
            const layer = L.polyline(points, {
              color: seg.is_new ? color : '#555',
              weight: seg.is_new ? 5 : 2,
              opacity: seg.is_new ? 0.9 : 0.5,
            }).addTo(this.map);
```

with:

```typescript
            const layer = L.polyline(points, {
              color: seg.is_new ? color : '#9a8a72',
              weight: seg.is_new ? 5 : 2,
              opacity: seg.is_new ? 0.9 : 0.5,
            }).addTo(this.map);
```

- [ ] **Step 4: Re-theme the idle-state overlay for the light map**

Replace:

```css
    .idle-state {
      position: absolute; inset: 0; z-index: 2;
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      background: rgba(18, 26, 19, 0.85); backdrop-filter: blur(4px);
      pointer-events: none;
      color: var(--text-muted); text-align: center;
    }
    .idle-state h2 { color: var(--text-primary); margin: 0 0 0.5rem; font-size: 1.5rem; }
    .idle-state p { margin: 0; font-size: 0.95rem; }
```

with:

```css
    .idle-state {
      position: absolute; inset: 0; z-index: 2;
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      background: rgba(224, 213, 184, 0.88); backdrop-filter: blur(4px);
      pointer-events: none;
      color: var(--text-muted); text-align: center;
    }
    .idle-state h2 { font-family: var(--font-display); color: var(--text-primary); margin: 0 0 0.5rem; font-size: 1.5rem; }
    .idle-state p { margin: 0; font-size: 0.95rem; }
```

- [ ] **Step 5: Manual check**

On `/app` with a city selected, confirm the map now shows warm/cream OSM tiles (not dark), the idle-state message overlay is paper-toned with a serif heading, and generated route polylines use the new rust/sage/ochre/slate/plum palette matching the sidebar cards' rank colors. Generate a novel-mode route and confirm the "already ridden" road segments render in the new muted warm-grey instead of dark grey.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/app/features/map-view/map-view.component.ts
git commit -m "style: swap map tiles to warm Voyager basemap and retheme route colors"
```

---

### Task 8: Loading overlay and map-building banner

**Files:**
- Modify: `frontend/src/app/shared/components/loading-overlay/loading-overlay.component.ts`
- Modify: `frontend/src/app/shared/components/map-building-banner/map-building-banner.component.ts`

**Interfaces:**
- Consumes: tokens from Task 1.

- [ ] **Step 1: Re-theme the loading overlay for the light map background**

In `loading-overlay.component.ts`, replace:

```css
    .overlay {
      position: absolute; inset: 0; z-index: 10;
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      background: rgba(18, 26, 19, 0.88); backdrop-filter: blur(6px);
      color: var(--text-primary); text-align: center; gap: 1rem;
    }
    .spinner-ring {
      width: 56px; height: 56px;
      border: 4px solid rgba(200, 145, 90, 0.2);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.9s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    h3 { margin: 0; font-size: 1.25rem; color: var(--text-primary); }
    p { margin: 0; color: var(--text-muted); font-size: 0.875rem; }
```

with:

```css
    .overlay {
      position: absolute; inset: 0; z-index: 10;
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      background: rgba(224, 213, 184, 0.92); backdrop-filter: blur(6px);
      color: var(--text-primary); text-align: center; gap: 1rem;
    }
    .spinner-ring {
      width: 56px; height: 56px;
      border: 4px solid rgba(168, 71, 31, 0.2);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.9s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    h3 { font-family: var(--font-display); margin: 0; font-size: 1.25rem; color: var(--text-primary); }
    p { margin: 0; color: var(--text-muted); font-size: 0.875rem; }
```

- [ ] **Step 2: Re-theme the map-building banner gradient and progress bar**

In `map-building-banner.component.ts`, replace:

```css
    .banner {
      background: linear-gradient(90deg, var(--bg-secondary), #1e2a1e);
      border-bottom: 2px solid var(--surface-active-border);
      padding: 0.625rem 1.5rem;
      z-index: 50;
    }
    .banner-content { display: flex; align-items: center; gap: 0.75rem; }
    .banner-text { flex: 1; }
    .banner-text strong { color: var(--text-primary); font-size: 0.875rem; margin-right: 0.5rem; }
    .banner-text span { color: var(--text-muted); font-size: 0.8rem; }
    .progress-bar-wrap { width: 120px; height: 4px; background: rgba(200, 190, 170, 0.1); border-radius: 2px; flex-shrink: 0; }
```

with:

```css
    .banner {
      background: linear-gradient(90deg, var(--bg-secondary), #e6dcc4);
      border-bottom: 2px solid var(--surface-active-border);
      padding: 0.625rem 1.5rem;
      z-index: 50;
    }
    .banner-content { display: flex; align-items: center; gap: 0.75rem; }
    .banner-text { flex: 1; }
    .banner-text strong { font-family: var(--font-display); color: var(--text-primary); font-size: 0.9rem; margin-right: 0.5rem; }
    .banner-text span { font-family: var(--font-mono); color: var(--text-muted); font-size: 0.75rem; }
    .progress-bar-wrap { width: 120px; height: 4px; background: rgba(60, 46, 30, 0.12); border-radius: 2px; flex-shrink: 0; }
```

- [ ] **Step 3: Manual check**

Trigger route generation and confirm the loading overlay is paper-toned with a serif "Generating Routes" heading and rust spinner. Add a custom pin to trigger graph building and confirm the banner reads clearly with a warm gradient and mono progress text.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/app/shared/components/loading-overlay/loading-overlay.component.ts frontend/src/app/shared/components/map-building-banner/map-building-banner.component.ts
git commit -m "style: retheme loading overlay and map-building banner to paper palette"
```

---

### Task 9: Rate Rides page

**Files:**
- Modify: `frontend/src/app/features/rate-rides/rate-rides.component.ts`

**Interfaces:**
- Consumes: tokens from Task 1. This page's `.page`, backgrounds, and most control colors already read from `var(--bg-primary)`, `var(--text-primary)`, `var(--accent)`, etc., so they re-theme automatically from Task 1 — this task only fixes the remaining hardcoded values and adds display/mono fonts.

- [ ] **Step 1: Give the page title the display serif font**

Replace:

```html
        <h1 class="page-title">Rate Your Rides</h1>
```

with no HTML change needed — instead add a font rule. Find the existing `.page` rule:

```css
    .page { min-height: 100vh; background: var(--bg-primary); color: var(--text-primary); padding: 2rem; font-family: var(--font-primary); }
```

and add a new rule immediately after it:

```css
    .page { min-height: 100vh; background: var(--bg-primary); color: var(--text-primary); padding: 2rem; font-family: var(--font-primary); }
    .page-title, .ride-name { font-family: var(--font-display); }
```

- [ ] **Step 2: Fix the hardcoded accent-tinted selected-state background**

Replace:

```css
    .rating-btn.selected { background: rgba(200, 145, 90, 0.2); border-color: var(--accent); color: var(--accent); }
```

with:

```css
    .rating-btn.selected { background: rgba(168, 71, 31, 0.16); border-color: var(--accent); color: var(--accent); }
```

- [ ] **Step 3: Fix the spinner border color and on-accent button text**

Replace:

```css
      padding: 0.625rem 1.5rem; background: var(--accent); color: #fff;
```

with:

```css
      padding: 0.625rem 1.5rem; background: var(--accent); color: var(--on-accent);
```

Replace:

```css
    .spinner { width: 32px; height: 32px; border: 3px solid rgba(200, 145, 90, 0.2); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; }
```

with:

```css
    .spinner { width: 32px; height: 32px; border: 3px solid rgba(168, 71, 31, 0.2); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; }
```

- [ ] **Step 4: Fix the hardcoded map polyline color**

Replace:

```typescript
    const line = L.polyline(points, { color: '#c8915a', weight: 3, opacity: 0.9 }).addTo(this.map);
```

with:

```typescript
    const line = L.polyline(points, { color: '#a8471f', weight: 3, opacity: 0.9 }).addTo(this.map);
```

- [ ] **Step 5: Manual check**

Visit `/app/rate-rides`. Confirm the page and controls look consistent with the rest of the app (paper background, serif page/ride title, rust accent on selected rating buttons, rust route polyline on the mini-map).

- [ ] **Step 6: Commit**

```bash
git add frontend/src/app/features/rate-rides/rate-rides.component.ts
git commit -m "style: retheme rate-rides page to Field Guide palette"
```

---

### Task 10: Rate Generated page

**Files:**
- Modify: `frontend/src/app/features/rate-generated/rate-generated.component.ts`

**Interfaces:**
- Consumes: tokens from Task 1. Same situation as Task 9 — most styling is already token-driven.

- [ ] **Step 1: Add the display serif to headings**

Find the existing `.page` rule:

```css
    .page { min-height: 100vh; background: var(--bg-primary); color: var(--text-primary); padding: 2rem; font-family: var(--font-primary); }
```

and add a new rule immediately after it:

```css
    .page { min-height: 100vh; background: var(--bg-primary); color: var(--text-primary); padding: 2rem; font-family: var(--font-primary); }
    .page-title, .route-label { font-family: var(--font-display); }
```

- [ ] **Step 2: Fix the hardcoded accent-tinted selected state and spinner**

Replace:

```css
    .rating-btn.selected { background: rgba(200, 145, 90, 0.2); border-color: var(--accent); color: var(--accent); }
```

with:

```css
    .rating-btn.selected { background: rgba(168, 71, 31, 0.16); border-color: var(--accent); color: var(--accent); }
```

Replace:

```css
    .spinner { width: 32px; height: 32px; border: 3px solid rgba(200, 145, 90, 0.2); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; }
```

with:

```css
    .spinner { width: 32px; height: 32px; border: 3px solid rgba(168, 71, 31, 0.2); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; }
```

- [ ] **Step 3: Fix on-accent button text color on the next/back buttons**

Replace:

```css
    .next-btn { padding: 0.625rem 1.5rem; background: var(--accent); color: #fff; border: none; border-radius: 8px; font-weight: 600; font-family: var(--font-primary); cursor: pointer; }
```

with:

```css
    .next-btn { padding: 0.625rem 1.5rem; background: var(--accent); color: var(--on-accent); border: none; border-radius: 4px; font-weight: 600; font-family: var(--font-primary); cursor: pointer; }
```

Replace:

```css
    .back-btn { padding: 0.625rem 1.5rem; background: var(--accent); color: #fff; border-radius: 8px; text-decoration: none; font-weight: 600; }
```

with:

```css
    .back-btn { padding: 0.625rem 1.5rem; background: var(--accent); color: var(--on-accent); border-radius: 4px; text-decoration: none; font-weight: 600; }
```

- [ ] **Step 4: Manual check**

Visit `/app/rate-generated`. Confirm headings use the serif font, selected rating buttons and buttons match the rust/paper palette used elsewhere, and button text is legible cream-on-rust rather than white-on-rust.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/app/features/rate-generated/rate-generated.component.ts
git commit -m "style: retheme rate-generated page to Field Guide palette"
```

---

### Task 11: Coverage page

**Files:**
- Modify: `frontend/src/app/features/coverage/coverage.component.ts`

**Interfaces:**
- Consumes: tokens from Task 1.

- [ ] **Step 1: Re-theme the nav bar background (matches Task 3's route-builder nav fix)**

Replace:

```css
    .topnav { display: flex; align-items: center; justify-content: space-between; padding: 0 1.5rem; height: 56px; min-height: 56px; background: rgba(18,26,19,0.95); border-bottom: 1px solid var(--border); }
```

with:

```css
    .topnav { display: flex; align-items: center; justify-content: space-between; padding: 0 1.5rem; height: 56px; min-height: 56px; background: rgba(224,213,184,0.95); border-bottom: 1px solid var(--border); }
    .brand { font-family: var(--font-display); }
```

- [ ] **Step 2: Add the display font to the sidebar heading**

Replace:

```css
    h1 { font-size: 1.1rem; margin: 0 0 1rem; }
```

with:

```css
    h1 { font-family: var(--font-display); font-size: 1.2rem; margin: 0 0 1rem; }
```

- [ ] **Step 3: Fix the hardcoded accent fallback hex values and on-accent text throughout**

Replace:

```css
    .pct { font-size: 2.4rem; font-weight: 800; color: var(--accent, #c8915a); }
```

with:

```css
    .pct { font-family: var(--font-mono); font-size: 2.4rem; font-weight: 800; color: var(--accent, #a8471f); }
```

Replace:

```css
    .toggle button.active { background: var(--accent, #c8915a); color: #1a1a1a; font-weight: 700; }
```

with:

```css
    .toggle button.active { background: var(--accent, #a8471f); color: var(--on-accent, #f4ead4); font-weight: 700; }
```

Replace:

```css
    .primary { background: var(--accent, #c8915a); border: none; border-radius: 8px; color: #1a1a1a; padding: 0.6rem 0.75rem; font-weight: 700; cursor: pointer; font-family: var(--font-primary); }
```

with:

```css
    .primary { background: var(--accent, #a8471f); border: none; border-radius: 4px; color: var(--on-accent, #f4ead4); padding: 0.6rem 0.75rem; font-weight: 700; cursor: pointer; font-family: var(--font-primary); box-shadow: 3px 3px 0 rgba(60, 46, 30, 0.25); }
```

Replace:

```css
    .error { color: #d9776c; font-size: 0.82rem; margin: 0; }
```

with:

```css
    .error { color: #8a3626; font-size: 0.82rem; margin: 0; }
```

Replace:

```css
    .link { background: none; border: none; color: var(--accent, #c8915a); cursor: pointer; font-size: 0.82rem; }
```

with:

```css
    .link { background: none; border: none; color: var(--accent, #a8471f); cursor: pointer; font-size: 0.82rem; }
```

Replace:

```css
    .bar-fill { height: 100%; background: var(--accent, #c8915a); transition: width 0.3s; }
```

with:

```css
    .bar-fill { height: 100%; background: var(--accent, #a8471f); transition: width 0.3s; }
```

- [ ] **Step 4: Re-theme the map "loading" pill for the light basemap**

Replace:

```css
    .loading { position: absolute; top: 12px; left: 50%; transform: translateX(-50%); background: rgba(18,26,19,0.9); padding: 0.4rem 0.9rem; border-radius: 8px; font-size: 0.85rem; z-index: 500; }
```

with:

```css
    .loading { position: absolute; top: 12px; left: 50%; transform: translateX(-50%); background: rgba(224,213,184,0.92); color: var(--text-primary); padding: 0.4rem 0.9rem; border-radius: 4px; font-size: 0.85rem; z-index: 500; border: 1px solid var(--border); }
```

- [ ] **Step 5: Update the ridden/unridden street colors and route-generation polyline colors**

Replace:

```typescript
const RIDDEN_COLOR = '#7a9a6d';    // green — streets you've ridden
const UNRIDDEN_COLOR = '#c0584e';  // red — streets you haven't
```

with:

```typescript
const RIDDEN_COLOR = '#5f7a52';    // sage green — streets you've ridden
const UNRIDDEN_COLOR = '#b23a2e';  // red — streets you haven't
```

Replace:

```typescript
      color: '#888', weight: 1, fill: false, dashArray: '4 6',
```

with:

```typescript
      color: '#9a8a72', weight: 1, fill: false, dashArray: '4 6',
```

Replace:

```typescript
          color: seg.is_new ? '#e0a458' : '#666',
```

with:

```typescript
          color: seg.is_new ? '#a8471f' : '#9a8a72',
```

Replace:

```typescript
      L.polyline(decodePolyline(r.polyline), { color: '#e0a458', weight: 5 }).addTo(this.routeLayer);
```

with:

```typescript
      L.polyline(decodePolyline(r.polyline), { color: '#a8471f', weight: 5 }).addTo(this.routeLayer);
```

- [ ] **Step 6: Manual check**

Visit `/app/coverage`. Confirm: nav bar and sidebar heading use paper/serif treatment, coverage percentage is in mono font, the ridden/unridden legend swatches and street coloring on the map use the new sage/red palette against the (now-light, from Task 7's global tile understanding — note this page has its own separate Leaflet map instance, confirm it isn't still using dark tiles; if the coverage map also hardcodes a dark tile URL, check `ngAfterViewInit` in this file and apply the same tile-layer swap as Task 7 Step 2), and the "Generate route to fill gaps" primary button has the stamped shadow with cream text.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/app/features/coverage/coverage.component.ts
git commit -m "style: retheme coverage page to Field Guide palette"
```

---

### Task 12: Final full-app visual pass

**Files:** none (verification only)

**Interfaces:** none.

- [ ] **Step 1: Walk every route in the running dev server**

With `npm start` running, visit in order: `/`, `/auth/callback` (via the normal Strava login redirect), `/app`, `/app/rate-rides`, `/app/rate-generated`, `/app/coverage`. On `/app`, exercise: selecting each preset city, dropping a custom pin (triggers the map-building banner), generating routes in each of the four modes (regular/hilly/historic/novel), selecting a route card, and downloading a GPX.

- [ ] **Step 2: Fix any remaining inconsistencies found**

If any element still shows the old dark-theme colors (a missed hardcoded hex/rgba, or a component not covered by Tasks 1–11), fix it in place following the same token/palette rules used throughout this plan (reuse `var(--bg-primary)`, `var(--accent)`, etc.; use the `RANK_COLORS`/`ROUTE_COLORS` palette for anything route-color-related; use `var(--font-display)` for headings and `var(--font-mono)` for data readouts and uppercase labels).

- [ ] **Step 3: Commit any fixes**

```bash
git add -A
git commit -m "style: fix remaining dark-theme leftovers found in full-app pass"
```

(Skip this step if Step 2 found nothing to fix.)
