// LAYER: Component — Route Card
// PURPOSE: Displays stats for a single generated route. Highlights when selected.
//          Includes a GPX download button that streams the file from the backend.

import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouteResult, RouteApiService } from '../../../core/services/route-api.service';

const RANK_COLORS = ['#fc4c02', '#3b82f6', '#8b5cf6', '#10b981', '#f59e0b'];

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
      <button
        class="gpx-btn"
        [id]="'gpx-btn-' + route.id"
        (click)="downloadGpx($event)"
        [disabled]="downloading"
      >
        {{ downloading ? 'Downloading...' : '↓ GPX' }}
      </button>
    </div>
  `,
  styles: [`
    .card {
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08);
      border-left: 3px solid transparent;
      border-radius: 10px;
      padding: 0.75rem;
      cursor: pointer;
      transition: all 0.15s;
    }
    .card:hover { background: rgba(255,255,255,0.07); }
    .card.selected { background: rgba(255,255,255,0.08); border-color: rgba(255,255,255,0.15); }

    .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
    .rank { font-weight: 800; font-size: 0.85rem; }
    .score { font-size: 1.25rem; font-weight: 800; color: #fff; }
    .score-max { font-size: 0.7rem; color: #6b7280; font-weight: 400; }

    .stats { display: flex; gap: 1rem; margin-bottom: 0.625rem; }
    .stat { display: flex; flex-direction: column; }
    .stat-val { font-size: 0.9rem; font-weight: 700; color: #e5e7eb; }
    .stat-label { font-size: 0.65rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; }

    .gpx-btn {
      width: 100%; padding: 0.375rem; background: rgba(255,255,255,0.07);
      border: 1px solid rgba(255,255,255,0.12); border-radius: 6px;
      color: #9ca3af; font-size: 0.75rem; cursor: pointer; transition: all 0.15s;
    }
    .gpx-btn:hover:not(:disabled) { background: rgba(255,255,255,0.12); color: #fff; }
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
