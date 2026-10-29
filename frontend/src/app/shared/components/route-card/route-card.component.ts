// LAYER: Component — Route Card
// PURPOSE: Compact summary of a single generated route in the sidebar list.
//          Highlights when selected; the full detail (elevation profile, GPX
//          download, historic sites) lives in RouteDetailComponent over the map.

import { Component, computed, input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouteResult } from '../../../core/services/route-api.service';

const RANK_COLORS = ['#a8471f', '#5f7a52', '#b8862c', '#5c6b73', '#8a4a5c'];

@Component({
  selector: 'app-route-card',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="card" [class.selected]="selected()" [style.border-left-color]="rankColor()">
      <div class="card-header">
        <span class="rank" [style.color]="rankColor()">#{{ rank() }}</span>
        <span class="score" *ngIf="route().predicted_score != null">{{ route().predicted_score }}<span class="score-max">/10</span></span>
      </div>
      <div class="stats">
        <div class="stat">
          <span class="stat-val">{{ route().distance_mi }}</span>
          <span class="stat-label">mi</span>
        </div>
        <div class="stat">
          <span class="stat-val">{{ route().elevation_ft | number:'1.0-0' }}</span>
          <span class="stat-label">ft gain</span>
        </div>
        <div class="stat" *ngIf="route().novelty_pct != null">
          <span class="stat-val">{{ route().novelty_pct | number:'1.0-0' }}%</span>
          <span class="stat-label">new roads</span>
        </div>
      </div>
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

    .stats { display: flex; gap: 1rem; }
    .stat { display: flex; flex-direction: column; }
    .stat-val { font-family: var(--font-mono); font-size: 0.9rem; font-weight: 700; color: var(--text-primary); }
    .stat-label { font-family: var(--font-mono); font-size: 0.65rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; }
  `]
})
export class RouteCardComponent {
  route = input.required<RouteResult>();
  rank = input(1);
  selected = input(false);

  rankColor = computed(() => RANK_COLORS[(this.rank() - 1) % RANK_COLORS.length]);
}
