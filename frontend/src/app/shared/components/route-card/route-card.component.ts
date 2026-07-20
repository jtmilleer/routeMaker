// LAYER: Component — Route Card
// PURPOSE: Displays stats for a single generated route. Highlights when selected.
//          Includes a GPX download button that streams the file from the backend.

import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouteResult, RouteApiService } from '../../../core/services/route-api.service';

const RANK_COLORS = ['#a8471f', '#5f7a52', '#b8862c', '#5c6b73', '#8a4a5c'];

@Component({
  selector: 'app-route-card',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="card" [class.selected]="selected" [style.border-left-color]="rankColor">
      <div class="card-header">
        <span class="rank" [style.color]="rankColor">#{{ rank }}</span>
        <span class="score">{{ route.predicted_score }}<span class="score-max">/10</span></span>
      </div>
      <div class="stats">
        <div class="stat">
          <span class="stat-val">{{ route.distance_mi }}</span>
          <span class="stat-label">mi</span>
        </div>
        <div class="stat">
          <span class="stat-val">{{ route.elevation_ft | number:'1.0-0' }}</span>
          <span class="stat-label">ft gain</span>
        </div>
        <div class="stat" *ngIf="route.novelty_pct != null">
          <span class="stat-val">{{ route.novelty_pct | number:'1.0-0' }}%</span>
          <span class="stat-label">new roads</span>
        </div>
      </div>
      <svg
        *ngIf="elevationLine"
        class="elev-chart"
        viewBox="0 0 100 28"
        preserveAspectRatio="none"
        [attr.aria-label]="'Elevation profile'"
      >
        <path [attr.d]="elevationArea" class="elev-area" />
        <path [attr.d]="elevationLine" class="elev-line" [attr.stroke]="rankColor" />
      </svg>
      <div class="historic-sites" *ngIf="route.historic_sites?.length">
        <span class="site-tag" *ngFor="let site of route.historic_sites">{{ site.name }}</span>
      </div>
      <button
        class="gpx-btn"
        [id]="'gpx-btn-' + route.id"
        (click)="downloadGpx($event)"
        [disabled]="downloading"
      >
        {{ downloading ? 'Downloading...' : 'Download GPX' }}
      </button>
    </div>
  `,
  styles: [`
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
    .gpx-btn:hover:not(:disabled) { background: var(--surface-hover); color: var(--text-primary); }
    .gpx-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  `]
})
export class RouteCardComponent {
  @Input() route!: RouteResult;
  @Input() rank = 1;
  @Input() selected = false;
  downloading = false;

  get rankColor(): string {
    return RANK_COLORS[(this.rank - 1) % RANK_COLORS.length];
  }

  /** SVG path for the elevation line, mapped into the 100x28 viewBox. */
  get elevationLine(): string | null {
    const prof = this.route.elevation_profile;
    if (!prof || prof.length < 2) return null;

    const maxD = prof[prof.length - 1][0] || 1;
    const elevs = prof.map(p => p[1]);
    const minE = Math.min(...elevs);
    const range = Math.max(...elevs) - minE || 1;
    const W = 100, H = 28;

    return prof
      .map((p, i) => {
        const x = (p[0] / maxD) * W;
        const y = H - ((p[1] - minE) / range) * H;
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  }

  /** Closed path filling the area under the elevation line. */
  get elevationArea(): string | null {
    const line = this.elevationLine;
    return line ? `${line} L100,28 L0,28 Z` : null;
  }

  constructor(private api: RouteApiService) {}

  downloadGpx(event: Event): void {
    event.stopPropagation();
    this.downloading = true;
    this.api.downloadGpx(this.route.id).subscribe({
      next: blob => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `route_${this.route.distance_mi}mi_${this.rank}.gpx`;
        a.click();
        URL.revokeObjectURL(url);
        this.downloading = false;
      },
      error: () => { this.downloading = false; }
    });
  }
}
