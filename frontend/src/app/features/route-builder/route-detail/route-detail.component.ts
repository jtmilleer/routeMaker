// LAYER: Component — Selected Route Detail Panel
// PURPOSE: Panel docked across the top of the map showing the full stats for
//          the selected route: score, distance/climb/time, a large elevation
//          profile with a hover readout, historic sites, and GPX download.
//          The sidebar cards stay compact; this is where the detail lives.

import { Component, computed, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { toSignal } from '@angular/core/rxjs-interop';
import { RouteStateService } from '../../../core/services/route-state.service';
import { RouteApiService } from '../../../core/services/route-api.service';

const RANK_COLORS = ['#a8471f', '#5f7a52', '#b8862c', '#5c6b73', '#8a4a5c'];
// Same flat-ground speed the backend assumes when it trains on generated routes.
const EST_SPEED_MPH = 15;
const CHART_W = 300, CHART_H = 70;

const TYPE_LABELS: Record<string, string> = {
  regular: 'Regular', hilly: 'Hilly', historic: 'Historic', novel: 'Novel', coverage: 'Coverage',
};

@Component({
  selector: 'app-route-detail',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="panel" *ngIf="route() as r" [style.border-top-color]="rankColor()">
      <div class="head">
        <span class="rank" [style.color]="rankColor()">#{{ rank() }}</span>
        <span class="type">{{ typeLabel() }} route</span>
        <span class="score" *ngIf="r.predicted_score != null">
          {{ r.predicted_score }}<span class="score-max">/10 predicted</span>
        </span>
        <span class="spacer"></span>
        <button class="gpx-btn" [id]="'gpx-btn-' + r.id" (click)="downloadGpx()" [disabled]="downloading()">
          {{ downloading() ? 'Downloading...' : 'Download GPX' }}
        </button>
        <button class="toggle" (click)="collapsed.set(!collapsed())"
          [attr.aria-label]="collapsed() ? 'Show route details' : 'Hide route details'">
          {{ collapsed() ? '▾' : '▴' }}
        </button>
      </div>

      <div class="body" *ngIf="!collapsed()">
        <div class="stats">
          <div class="stat"><span class="stat-val">{{ r.distance_mi }}</span><span class="stat-label">mi</span></div>
          <div class="stat"><span class="stat-val">{{ r.elevation_ft | number:'1.0-0' }}</span><span class="stat-label">ft gain</span></div>
          <div class="stat" title="At 15 mph average">
            <span class="stat-val">{{ estTime() }}</span><span class="stat-label">est. time</span>
          </div>
          <div class="stat"><span class="stat-val">{{ climbRate() | number:'1.0-0' }}</span><span class="stat-label">ft / mi</span></div>
          <ng-container *ngIf="elevRange() as e">
            <div class="stat"><span class="stat-val">{{ e.max | number:'1.0-0' }}</span><span class="stat-label">high pt ft</span></div>
            <div class="stat"><span class="stat-val">{{ e.min | number:'1.0-0' }}</span><span class="stat-label">low pt ft</span></div>
          </ng-container>
          <div class="stat" *ngIf="r.novelty_pct != null">
            <span class="stat-val">{{ r.novelty_pct | number:'1.0-0' }}%</span><span class="stat-label">new roads</span>
          </div>
        </div>

        <div class="chart" *ngIf="chart() as c" (mousemove)="onHover($event)" (mouseleave)="hoverIdx.set(null)">
          <div class="y-labels">
            <span>{{ c.max | number:'1.0-0' }} ft</span>
            <span>{{ c.min | number:'1.0-0' }} ft</span>
          </div>
          <div class="plot">
            <svg [attr.viewBox]="'0 0 ' + CHART_W + ' ' + CHART_H" preserveAspectRatio="none" aria-label="Elevation profile">
              <path [attr.d]="c.area" class="elev-area" />
              <path [attr.d]="c.line" class="elev-line" [attr.stroke]="rankColor()" />
              <line *ngIf="hover() as h" [attr.x1]="h.x" [attr.x2]="h.x" y1="0" [attr.y2]="CHART_H" class="cursor" />
            </svg>
            <div class="x-labels"><span>0 mi</span><span>{{ c.maxD | number:'1.1-1' }} mi</span></div>
            <div class="readout" *ngIf="hover() as h" [style.left.%]="h.x / CHART_W * 100">
              {{ h.d | number:'1.1-1' }} mi · {{ h.e | number:'1.0-0' }} ft
            </div>
          </div>
        </div>

        <div class="sites" *ngIf="r.historic_sites?.length">
          <span class="site-tag" *ngFor="let site of r.historic_sites">{{ site.name }}</span>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .panel {
      position: absolute; top: 12px; left: 12px; right: 12px; z-index: 500;
      background: var(--bg-surface);
      border: 1px solid var(--border); border-top: 3px solid transparent; border-radius: 4px;
      box-shadow: 3px 3px 0 rgba(60, 46, 30, 0.18);
      padding: 0.6rem 0.9rem;
    }
    .head { display: flex; align-items: baseline; gap: 0.75rem; }
    .rank { font-family: var(--font-mono); font-weight: 800; font-size: 0.95rem; }
    .type { font-family: var(--font-display); font-size: 1.1rem; font-weight: 700; color: var(--text-primary); }
    .score { font-family: var(--font-display); font-size: 1.1rem; font-weight: 700; color: var(--text-primary); }
    .score-max { font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-dim); font-weight: 400; margin-left: 0.15rem; }
    .spacer { flex: 1; }
    .gpx-btn, .toggle {
      background: var(--bg-surface); border: 1px solid var(--border); border-radius: 3px;
      color: var(--text-muted); font-size: 0.75rem; font-family: var(--font-mono);
      padding: 0.3rem 0.6rem; cursor: pointer; transition: all 0.15s; align-self: center;
    }
    .gpx-btn:hover:not(:disabled), .toggle:hover { background: var(--surface-hover); color: var(--text-primary); }
    .gpx-btn:disabled { opacity: 0.5; cursor: not-allowed; }

    .body { display: flex; flex-wrap: wrap; gap: 0.75rem 1.5rem; margin-top: 0.6rem; align-items: flex-start; }
    .stats { display: grid; grid-template-columns: repeat(4, auto); gap: 0.5rem 1.25rem; }
    .stat { display: flex; flex-direction: column; }
    .stat-val { font-family: var(--font-mono); font-size: 0.95rem; font-weight: 700; color: var(--text-primary); }
    .stat-label { font-family: var(--font-mono); font-size: 0.65rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; }

    .chart { flex: 1; min-width: 240px; display: flex; gap: 0.4rem; cursor: crosshair; }
    .y-labels {
      display: flex; flex-direction: column; justify-content: space-between;
      height: 70px; font-family: var(--font-mono); font-size: 0.62rem; color: var(--text-dim); text-align: right;
    }
    .plot { flex: 1; position: relative; }
    .plot svg { width: 100%; height: 70px; display: block; overflow: visible; }
    .elev-line { fill: none; stroke-width: 1.5; vector-effect: non-scaling-stroke; }
    .elev-area { fill: rgba(168, 71, 31, 0.15); stroke: none; }
    .cursor { stroke: var(--text-muted); stroke-width: 1; vector-effect: non-scaling-stroke; stroke-dasharray: 2 2; }
    .x-labels { display: flex; justify-content: space-between; font-family: var(--font-mono); font-size: 0.62rem; color: var(--text-dim); }
    .readout {
      position: absolute; top: -4px; transform: translate(-50%, -100%); white-space: nowrap; pointer-events: none;
      font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-primary);
      background: var(--bg-surface); border: 1px solid var(--border); border-radius: 3px; padding: 0.1rem 0.4rem;
    }

    .sites { display: flex; flex-wrap: wrap; gap: 0.3rem; width: 100%; }
    .site-tag {
      font-family: var(--font-mono); font-size: 0.68rem; background: rgba(95, 122, 82, 0.14); color: #45592f;
      border: 1px solid rgba(95, 122, 82, 0.35); border-radius: 3px; padding: 0.2rem 0.5rem;
    }
  `]
})
export class RouteDetailComponent {
  readonly CHART_W = CHART_W;
  readonly CHART_H = CHART_H;

  private state = inject(RouteStateService);
  private api = inject(RouteApiService);
  private routes = toSignal(this.state.routes$, { initialValue: [] });
  route = toSignal(this.state.selectedRoute$, { initialValue: null });
  collapsed = signal(false);
  downloading = signal(false);
  hoverIdx = signal<number | null>(null);

  rank = computed(() => {
    const r = this.route();
    return r ? this.routes().findIndex(x => x.id === r.id) + 1 : 0;
  });
  rankColor = computed(() => RANK_COLORS[(Math.max(this.rank(), 1) - 1) % RANK_COLORS.length]);
  typeLabel = computed(() => TYPE_LABELS[this.route()?.route_type ?? ''] ?? 'Generated');

  estTime = computed(() => {
    const mins = Math.round(((this.route()?.distance_mi ?? 0) / EST_SPEED_MPH) * 60);
    return `${Math.floor(mins / 60)}:${String(mins % 60).padStart(2, '0')}`;
  });
  climbRate = computed(() => {
    const r = this.route();
    return r && r.distance_mi > 0 ? r.elevation_ft / r.distance_mi : 0;
  });

  private profile = computed(() => {
    const prof = this.route()?.elevation_profile;
    return prof && prof.length >= 2 ? prof : null;
  });
  elevRange = computed(() => {
    const prof = this.profile();
    if (!prof) return null;
    const elevs = prof.map(p => p[1]);
    return { min: Math.min(...elevs), max: Math.max(...elevs) };
  });

  /** Elevation line/area paths mapped into the chart's viewBox, plus axis bounds. */
  chart = computed(() => {
    const prof = this.profile();
    const range = this.elevRange();
    if (!prof || !range) return null;
    const maxD = prof[prof.length - 1][0] || 1;
    const span = range.max - range.min || 1;
    const pts = prof.map(p => ({
      x: (p[0] / maxD) * CHART_W,
      y: CHART_H - ((p[1] - range.min) / span) * CHART_H,
      d: p[0],
      e: p[1],
    }));
    const line = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
    return { pts, line, area: `${line} L${CHART_W},${CHART_H} L0,${CHART_H} Z`, maxD, ...range };
  });

  hover = computed(() => {
    const i = this.hoverIdx();
    const c = this.chart();
    return i != null && c ? c.pts[i] : null;
  });

  onHover(event: MouseEvent): void {
    const c = this.chart();
    const plot = (event.currentTarget as HTMLElement).querySelector('.plot');
    if (!c || !plot) return;
    const rect = plot.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * CHART_W;
    let best = 0;
    c.pts.forEach((p, i) => { if (Math.abs(p.x - x) < Math.abs(c.pts[best].x - x)) best = i; });
    this.hoverIdx.set(best);
  }

  downloadGpx(): void {
    const route = this.route();
    if (!route) return;
    this.downloading.set(true);
    this.api.downloadGpx(route.id).subscribe({
      next: blob => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `route_${route.distance_mi}mi_${this.rank()}.gpx`;
        a.click();
        URL.revokeObjectURL(url);
        this.downloading.set(false);
      },
      error: () => this.downloading.set(false),
    });
  }
}
