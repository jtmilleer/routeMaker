// LAYER: Component — Street Coverage Page
// PURPOSE: The "100% coverage" feature. Shows which streets within a radius of the
//          user's saved home base they've ridden (green) vs. not (red), with a
//          "% covered" stat, and can generate loops that fill in the gaps.
//          Reuses the shared polyline decoder and the existing GPX download API.

import {
  Component, AfterViewInit, OnDestroy, ElementRef, ViewChild, signal,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import * as L from 'leaflet';

import { AuthService } from '../../core/services/auth.service';
import { RouteApiService, CoverageResponse, RouteResult, CoverageActivity } from '../../core/services/route-api.service';
import { PendingRatingsService } from '../../core/services/pending-ratings.service';
import { decodePolyline } from '../../shared/utils/polyline';

// Leaflet's default marker icons break under bundlers — point them at the CDN.
const iconDefault = L.icon({
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
  iconSize: [25, 41], iconAnchor: [12, 41],
});
L.Marker.prototype.options.icon = iconDefault;

const RIDDEN_COLOR = '#5f7a52';    // sage green — streets you've ridden
const UNRIDDEN_COLOR = '#b23a2e';  // red — streets you haven't
const DEFAULT_CENTER: [number, number] = [41.6543, -91.5267];  // Iowa City

@Component({
  selector: 'app-coverage',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="shell">
      <div class="body">
        <aside class="sidebar">
          <h1>Street Coverage</h1>

          <!-- No home base yet -->
          <div *ngIf="mode() === 'setHome'" class="panel">
            <p class="hint">Click the map to set your <strong>home base</strong> — the
              center of your coverage area.</p>
            <p class="coords" *ngIf="pendingLat() !== null">
              {{ pendingLat()! | number:'1.4-4' }}, {{ pendingLng()! | number:'1.4-4' }}
            </p>
            <button class="primary" [disabled]="pendingLat() === null || saving()"
                    (click)="saveHome()">
              {{ saving() ? 'Saving…' : 'Save home base' }}
            </button>
          </div>

          <!-- Building graph for the area -->
          <div *ngIf="mode() === 'building'" class="panel">
            <p class="hint">Building the map for your area… {{ buildPct() }}%</p>
            <div class="bar"><div class="bar-fill" [style.width.%]="buildPct()"></div></div>
          </div>

          <!-- Ready: show coverage -->
          <div *ngIf="mode() === 'ready'" class="panel">
            <div class="toggle">
              <button [class.active]="activity() === 'ride'" (click)="setActivity('ride')">Ride</button>
              <button [class.active]="activity() === 'walk'" (click)="setActivity('walk')">Walk</button>
              <button [class.active]="activity() === 'both'" (click)="setActivity('both')">Both</button>
            </div>
            <button class="ghost sync" [disabled]="syncing()" (click)="syncFromStrava()">
              {{ syncing() ? 'Syncing…' : 'Sync from Strava' }}
            </button>
            <p class="hint" *ngIf="syncMsg()">{{ syncMsg() }}</p>
            <div class="stat">
              <span class="pct">{{ coverage()?.coverage_pct ?? 0 }}%</span>
              <span class="stat-label">covered</span>
            </div>
            <p class="miles">{{ coverage()?.ridden_mi ?? 0 }} / {{ coverage()?.total_mi ?? 0 }} mi covered</p>

            <label class="field">
              <span>Radius: {{ radiusMi() }} mi</span>
              <input type="range" min="0.25" max="5" step="0.25"
                     [ngModel]="radiusMi()" (ngModelChange)="onRadiusChange($event)">
            </label>

            <div class="legend">
              <span><i class="swatch" [style.background]="RIDDEN_COLOR"></i> covered</span>
              <span><i class="swatch" [style.background]="UNRIDDEN_COLOR"></i> not yet</span>
            </div>

            <hr>

            <label class="field">
              <span>Route length: {{ routeDistanceMi() }} mi</span>
              <input type="range" min="1" max="40" step="1"
                     [ngModel]="routeDistanceMi()" (ngModelChange)="routeDistanceMi.set($event)">
            </label>
            <button class="primary" [disabled]="generating()" (click)="generateRoute()">
              {{ generating() ? 'Generating…' : 'Generate route to fill gaps' }}
            </button>
            <p class="error" *ngIf="error()">{{ error() }}</p>

            <div class="routes" *ngIf="routes().length">
              <div class="route-row" *ngFor="let r of routes(); let i = index">
                <span>{{ r.distance_mi }} mi · {{ r.novelty_pct }}% new</span>
                <button class="link" (click)="downloadGpx(r)">GPX</button>
              </div>
            </div>

            <button class="ghost" (click)="changeHome()">Change home base</button>
          </div>
        </aside>

        <main class="map-area">
          <div #mapEl class="map"></div>
          <div class="loading" *ngIf="loading()">Loading coverage…</div>
        </main>
      </div>
    </div>
  `,
  styles: [`
    :host { display: flex; flex: 1; min-height: 0; overflow: hidden; }
    .shell { display: flex; flex-direction: column; height: 100%; width: 100%; background: var(--bg-primary); color: var(--text-primary); font-family: var(--font-primary); }
    .body { display: flex; flex: 1; overflow: hidden; }
    .sidebar { width: 320px; min-width: 320px; padding: 1.25rem; border-right: 1px solid var(--border); overflow-y: auto; }
    h1 { font-family: var(--font-display); font-size: 1.2rem; margin: 0 0 1rem; }
    .panel { display: flex; flex-direction: column; gap: 0.85rem; }
    .hint { color: var(--text-muted); font-size: 0.85rem; line-height: 1.4; margin: 0; }
    .coords { font-family: monospace; font-size: 0.8rem; color: var(--text-secondary); margin: 0; }
    .stat { display: flex; align-items: baseline; gap: 0.5rem; }
    .pct { font-family: var(--font-mono); font-size: 2.4rem; font-weight: 800; color: var(--accent, #a8471f); }
    .stat-label { color: var(--text-muted); }
    .miles { margin: 0; color: var(--text-secondary); font-size: 0.9rem; }
    .field { display: flex; flex-direction: column; gap: 0.35rem; font-size: 0.85rem; color: var(--text-secondary); }
    .field input[type=range] { width: 100%; }
    .toggle { display: flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
    .toggle button { flex: 1; background: transparent; border: none; color: var(--text-muted); padding: 0.5rem; cursor: pointer; font-family: var(--font-primary); font-size: 0.85rem; }
    .toggle button.active { background: var(--accent, #a8471f); color: var(--on-accent, #f4ead4); font-weight: 700; }
    .legend { display: flex; gap: 1rem; font-size: 0.8rem; color: var(--text-muted); }
    .swatch { display: inline-block; width: 12px; height: 12px; border-radius: 3px; vertical-align: middle; margin-right: 4px; }
    hr { border: none; border-top: 1px solid var(--border); width: 100%; margin: 0.25rem 0; }
    .primary { background: var(--accent, #a8471f); border: none; border-radius: 4px; color: var(--on-accent, #f4ead4); padding: 0.6rem 0.75rem; font-weight: 700; cursor: pointer; font-family: var(--font-primary); box-shadow: 3px 3px 0 rgba(60, 46, 30, 0.25); }
    .primary:hover:not(:disabled) { background: var(--accent-hover); transform: translate(-1px, -1px); box-shadow: 4px 4px 0 rgba(60, 46, 30, 0.3); }
    .primary:disabled { opacity: 0.6; cursor: default; }
    .ghost { background: transparent; border: 1px solid var(--border); border-radius: 8px; color: var(--text-muted); padding: 0.5rem; cursor: pointer; font-family: var(--font-primary); }
    .error { color: #8a3626; font-size: 0.82rem; margin: 0; }
    .routes { display: flex; flex-direction: column; gap: 0.4rem; }
    .route-row { display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem; color: var(--text-secondary); }
    .link { background: none; border: none; color: var(--accent, #a8471f); cursor: pointer; font-size: 0.82rem; }
    .bar { width: 100%; height: 6px; background: var(--bg-surface); border-radius: 3px; overflow: hidden; }
    .bar-fill { height: 100%; background: var(--accent, #a8471f); transition: width 0.3s; }
    .map-area { flex: 1; position: relative; }
    .map { width: 100%; height: 100%; }
    .loading { position: absolute; top: 12px; left: 50%; transform: translateX(-50%); background: rgba(224,213,184,0.92); color: var(--text-primary); padding: 0.4rem 0.9rem; border-radius: 4px; font-size: 0.85rem; z-index: 500; border: 1px solid var(--border); }
  `],
})
export class CoverageComponent implements AfterViewInit, OnDestroy {
  @ViewChild('mapEl', { static: true }) mapEl!: ElementRef<HTMLDivElement>;

  // Expose constants to the template
  readonly RIDDEN_COLOR = RIDDEN_COLOR;
  readonly UNRIDDEN_COLOR = UNRIDDEN_COLOR;

  mode = signal<'setHome' | 'building' | 'ready'>('setHome');
  loading = signal(false);
  saving = signal(false);
  generating = signal(false);
  syncing = signal(false);
  syncMsg = signal<string | null>(null);
  buildPct = signal(0);
  error = signal<string | null>(null);

  radiusMi = signal(1);
  routeDistanceMi = signal(15);
  activity = signal<CoverageActivity>('ride');
  coverage = signal<CoverageResponse | null>(null);
  routes = signal<RouteResult[]>([]);

  pendingLat = signal<number | null>(null);
  pendingLng = signal<number | null>(null);

  private map!: L.Map;
  private coverageLayer = L.layerGroup();
  private routeLayer = L.layerGroup();
  private homeMarker?: L.Marker;
  private radiusCircle?: L.Circle;
  private cityKey = '';
  private pollActive = false;
  private pollTimer?: ReturnType<typeof setTimeout>;
  private radiusDebounce?: ReturnType<typeof setTimeout>;

  constructor(public auth: AuthService, private api: RouteApiService, private pending: PendingRatingsService) {}

  ngAfterViewInit(): void {
    const user = this.auth.currentUser();
    const hasHome = user?.home_lat != null && user?.home_lng != null;
    const center: [number, number] = hasHome
      ? [user!.home_lat!, user!.home_lng!]
      : DEFAULT_CENTER;

    this.map = L.map(this.mapEl.nativeElement, { center, zoom: hasHome ? 14 : 12 });
    L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; OpenStreetMap &copy; CARTO', maxZoom: 19,
    }).addTo(this.map);
    this.coverageLayer.addTo(this.map);
    this.routeLayer.addTo(this.map);

    this.map.on('click', (e: L.LeafletMouseEvent) => {
      if (this.mode() !== 'setHome') return;
      this.pendingLat.set(e.latlng.lat);
      this.pendingLng.set(e.latlng.lng);
      if (this.homeMarker) this.homeMarker.setLatLng(e.latlng);
      else this.homeMarker = L.marker(e.latlng).addTo(this.map);
    });

    if (hasHome) {
      this.mode.set('ready');
      this.cityKey = this.makeCityKey(user!.home_lat!, user!.home_lng!);
      this.placeHome(user!.home_lat!, user!.home_lng!);
      this.loadCoverage();
    }
  }

  ngOnDestroy(): void {
    this.stopPolling();
    if (this.radiusDebounce) clearTimeout(this.radiusDebounce);
    this.map?.remove();
  }

  changeHome(): void {
    this.stopPolling();
    this.mode.set('setHome');
    this.coverageLayer.clearLayers();
    this.routeLayer.clearLayers();
    this.routes.set([]);
  }

  /** Replicate the backend's city_key (Python round(x, 2)) so polling targets the
   *  same key the server stores. Authoritative keys from API responses override it. */
  private makeCityKey(lat: number, lng: number): string {
    const r = (x: number) => {
      const v = Math.round(x * 100) / 100;
      return Number.isInteger(v) ? `${v}.0` : `${v}`;
    };
    return `${r(lat)}_${r(lng)}`;
  }

  saveHome(): void {
    const lat = this.pendingLat(), lng = this.pendingLng();
    if (lat == null || lng == null) return;
    this.saving.set(true);
    this.error.set(null);
    this.api.setHomeBase(lat, lng).subscribe({
      next: res => {
        this.saving.set(false);
        this.cityKey = res.city_key;  // authoritative key from the server
        // Reflect the new home in the shared auth user.
        const u = this.auth.currentUser();
        if (u) this.auth.currentUser.set({ ...u, home_lat: lat, home_lng: lng });
        this.placeHome(lat, lng);
        this.mode.set('building');
        this.startPolling();
      },
      error: () => { this.saving.set(false); this.error.set('Could not save home base.'); },
    });
  }

  private placeHome(lat: number, lng: number): void {
    if (this.homeMarker) this.homeMarker.setLatLng([lat, lng]);
    else this.homeMarker = L.marker([lat, lng]).addTo(this.map);
    this.map.setView([lat, lng], 14);
  }

  /** Start polling the graph build status (no-op if already polling). */
  private startPolling(): void {
    if (this.pollActive || !this.cityKey) return;
    this.pollActive = true;
    this.pollGraph();
  }

  private stopPolling(): void {
    this.pollActive = false;
    if (this.pollTimer) { clearTimeout(this.pollTimer); this.pollTimer = undefined; }
  }

  /** Poll until the area is ready, then load coverage. Stops on ready/error so it
   *  never loops forever. Re-entry is guarded by pollActive. */
  private pollGraph(): void {
    if (!this.pollActive || !this.cityKey) return;
    this.api.getGraphStatus(this.cityKey).subscribe({
      next: s => {
        if (!this.pollActive) return;
        this.buildPct.set(s.progress_pct ?? 0);
        if (s.status === 'ready') {
          this.stopPolling();
          this.mode.set('ready');
          this.loadCoverage();
        } else if (s.status === 'error') {
          this.stopPolling();
          this.error.set('Map build failed for this area. Try a different home base.');
        } else {
          // "building" (or transient "not_found" just after queueing) — keep waiting.
          this.pollTimer = setTimeout(() => this.pollGraph(), 3000);
        }
      },
      error: () => {
        if (this.pollActive) this.pollTimer = setTimeout(() => this.pollGraph(), 4000);
      },
    });
  }

  onRadiusChange(value: number): void {
    this.radiusMi.set(value);
    if (this.radiusDebounce) clearTimeout(this.radiusDebounce);
    this.radiusDebounce = setTimeout(() => this.loadCoverage(), 300);
  }

  setActivity(a: CoverageActivity): void {
    if (this.activity() === a) return;
    this.activity.set(a);
    this.routeLayer.clearLayers();
    this.routes.set([]);
    this.loadCoverage();
  }

  /** Pull the latest activities from Strava, then refresh coverage so newly
   *  imported walks/runs/rides show up without leaving this page. */
  syncFromStrava(): void {
    this.syncing.set(true);
    this.syncMsg.set(null);
    this.api.syncRides().subscribe({
      next: s => {
        this.syncing.set(false);
        this.syncMsg.set(`${s.new_rides} new (${s.total_rides} total)`);
        this.loadCoverage();
      },
      error: () => {
        this.syncing.set(false);
        this.syncMsg.set('Sync failed — try again.');
      },
    });
  }

  loadCoverage(): void {
    this.loading.set(true);
    this.error.set(null);
    this.api.getCoverage(this.radiusMi(), this.activity()).subscribe({
      next: resp => { this.loading.set(false); this.renderCoverage(resp); },
      error: err => {
        this.loading.set(false);
        if (err?.status === 503) { this.mode.set('building'); this.startPolling(); }
        else this.error.set('Could not load coverage.');
      },
    });
  }

  private renderCoverage(resp: CoverageResponse): void {
    this.coverage.set(resp);
    this.cityKey = resp.city_key;  // authoritative key from the server
    this.coverageLayer.clearLayers();

    for (const seg of resp.segments) {
      L.polyline(decodePolyline(seg.polyline), {
        color: seg.ridden ? RIDDEN_COLOR : UNRIDDEN_COLOR,
        weight: seg.ridden ? 3 : 4,
        opacity: seg.ridden ? 0.7 : 0.9,
      }).addTo(this.coverageLayer);
    }

    if (this.radiusCircle) this.map.removeLayer(this.radiusCircle);
    this.radiusCircle = L.circle([resp.center_lat, resp.center_lng], {
      radius: resp.radius_mi * 1609.34,
      color: '#9a8a72', weight: 1, fill: false, dashArray: '4 6',
    }).addTo(this.map);
    this.map.fitBounds(this.radiusCircle.getBounds(), { padding: [20, 20] });
  }

  generateRoute(): void {
    this.generating.set(true);
    this.error.set(null);
    this.routeLayer.clearLayers();
    this.api.generateCoverageRoute(this.radiusMi(), this.activity(), this.routeDistanceMi()).subscribe({
      next: routes => { this.generating.set(false); this.routes.set(routes); this.renderRoutes(routes); this.pending.refresh(); },
      error: err => {
        this.generating.set(false);
        this.error.set(err?.error?.detail ?? 'Could not generate a route.');
      },
    });
  }

  private renderRoutes(routes: RouteResult[]): void {
    // Draw the first (best) candidate; its segments mark new vs. already-ridden.
    const r = routes[0];
    if (!r) return;
    if (r.route_segments) {
      const segs: { polyline: string; is_new: boolean }[] = JSON.parse(r.route_segments);
      for (const seg of segs) {
        L.polyline(decodePolyline(seg.polyline), {
          color: seg.is_new ? '#a8471f' : '#9a8a72',
          weight: seg.is_new ? 5 : 3,
          opacity: seg.is_new ? 0.95 : 0.6,
        }).addTo(this.routeLayer);
      }
    } else if (r.polyline) {
      L.polyline(decodePolyline(r.polyline), { color: '#a8471f', weight: 5 }).addTo(this.routeLayer);
    }
  }

  downloadGpx(r: RouteResult): void {
    this.api.downloadGpx(r.id).subscribe(blob => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `coverage_route_${r.distance_mi}mi.gpx`;
      a.click();
      URL.revokeObjectURL(url);
    });
  }
}
