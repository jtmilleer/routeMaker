// LAYER: Component — Route Configuration Form
// PURPOSE: Sidebar form with sliders for distance, tolerance, route type,
//          and conditional type-specific options. Submits to the API via
//          RouteApiService and updates state via RouteStateService.

import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RouteApiService, GenerateRequest } from '../../../core/services/route-api.service';
import { RouteStateService } from '../../../core/services/route-state.service';
import { PendingRatingsService } from '../../../core/services/pending-ratings.service';

type RouteType = 'regular' | 'hilly' | 'historic' | 'novel';

@Component({
  selector: 'app-route-form',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="form-panel">
      <h3 class="section-title">Generate Route</h3>

      <!-- Route Type -->
      <div class="form-group">
        <label class="label">Route Mode</label>
        <div class="mode-grid">
          <button *ngFor="let t of routeTypes" class="mode-btn"
            [class.active]="routeType === t.value"
            (click)="routeType = t.value"
            [id]="'mode-' + t.value">
            {{ t.label }}
          </button>
        </div>
      </div>

      <!-- Distance -->
      <div class="form-group">
        <label class="label">Distance: <span class="val accent">{{ distance }} mi</span></label>
        <input id="distance-slider" type="range" class="slider" min="10" max="100" step="1" [(ngModel)]="distance">
        <div class="slider-range"><span>10</span><span>100 mi</span></div>
      </div>

      <!-- Tolerance -->
      <div class="form-group">
        <label class="label">Tolerance: <span class="val accent">&plusmn;{{ (tolerance * 100).toFixed(0) }}%</span></label>
        <input id="tolerance-slider" type="range" class="slider" min="0.05" max="0.5" step="0.05" [(ngModel)]="tolerance">
        <div class="slider-range"><span>5%</span><span>50%</span></div>
      </div>

      <!-- Hilly factor (hilly mode only) -->
      <div class="form-group" *ngIf="routeType === 'hilly'">
        <label class="label">Hilly Factor: <span class="val accent">{{ hillyFactor }}</span></label>
        <input id="hilly-factor-slider" type="range" class="slider" min="0" max="500" step="10" [(ngModel)]="hillyFactor">
        <div class="slider-range"><span>Flat</span><span>Very hilly</span></div>
      </div>

      <!-- Downtown radius (historic mode only) -->
      <div class="form-group" *ngIf="routeType === 'historic'">
        <label class="label">Historic Site Radius: <span class="val accent">{{ downtownRadius }}m</span></label>
        <input id="radius-slider" type="range" class="slider" min="500" max="5000" step="100" [(ngModel)]="downtownRadius">
        <div class="slider-range"><span>500m</span><span>5km</span></div>
      </div>

      <!-- Novelty factor (novel mode only) -->
      <div class="form-group" *ngIf="routeType === 'novel'">
        <label class="label">Novelty Factor: <span class="val accent">{{ noveltyFactor }}</span></label>
        <input id="novelty-slider" type="range" class="slider" min="0.5" max="20" step="0.5" [(ngModel)]="noveltyFactor">
        <div class="slider-range"><span>Slight</span><span>Max new roads</span></div>
      </div>

      <!-- Generate button -->
      <button
        id="generate-btn"
        class="generate-btn"
        (click)="generate()"
        [disabled]="(state.loading$ | async) || !(state.activeCity$ | async)"
      >
        <span *ngIf="!(state.loading$ | async)">Generate Routes</span>
        <span *ngIf="state.loading$ | async">Generating...</span>
      </button>

      <p class="hint" *ngIf="!(state.activeCity$ | async)">
        Select a city or drop a pin above to begin.
      </p>
      <p class="error" *ngIf="error">{{ error }}</p>
    </div>
  `,
  styles: [`
    .form-panel { display: flex; flex-direction: column; gap: 1rem; }
    .section-title { font-family: var(--font-mono); font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-dim); margin: 0 0 0.5rem; }
    .form-group { display: flex; flex-direction: column; gap: 0.375rem; }
    .label { font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); font-weight: 500; text-transform: uppercase; letter-spacing: 0.04em; }
    .val { font-weight: 700; }
    .accent { color: var(--accent); }

    .mode-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.375rem; }
    .mode-btn {
      background: var(--bg-surface); border: 1px solid var(--border);
      border-radius: 3px; color: var(--text-muted); padding: 0.5rem; font-size: 0.8rem;
      font-family: var(--font-primary);
      cursor: pointer; transition: all 0.15s; text-align: center;
    }
    .mode-btn:hover { background: var(--surface-hover); color: var(--text-secondary); }
    .mode-btn.active { background: var(--surface-active); border-color: var(--surface-active-border); color: var(--accent); font-weight: 600; }

    .slider { width: 100%; accent-color: var(--accent); cursor: pointer; }
    .slider-range { display: flex; justify-content: space-between; font-size: 0.7rem; color: var(--text-dim); margin-top: -0.25rem; }

    .generate-btn {
      width: 100%; padding: 0.875rem; background: var(--accent);
      border: none; border-radius: 4px; color: var(--on-accent); font-size: 0.95rem;
      font-weight: 700; font-family: var(--font-primary);
      cursor: pointer; transition: all 0.15s;
      box-shadow: 3px 3px 0 rgba(60, 46, 30, 0.25);
      margin-top: 0.5rem;
    }
    .generate-btn:hover:not(:disabled) { background: var(--accent-hover); transform: translate(-1px, -1px); box-shadow: 4px 4px 0 rgba(60, 46, 30, 0.3); }
    .generate-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
    .hint { font-size: 0.75rem; color: var(--text-dim); text-align: center; margin: 0; }
    .error { font-size: 0.75rem; color: #8a3626; margin: 0; }
  `]
})
export class RouteFormComponent {
  routeType: RouteType = 'regular';
  distance = 35;
  tolerance = 0.25;
  hillyFactor = 100;
  downtownRadius = 1000;
  noveltyFactor = 5.0;
  error = '';

  routeTypes = [
    { value: 'regular' as RouteType, label: 'Regular' },
    { value: 'hilly' as RouteType, label: 'Hilly' },
    { value: 'historic' as RouteType, label: 'Historic' },
    { value: 'novel' as RouteType, label: 'Novel' },
  ];

  constructor(private api: RouteApiService, public state: RouteStateService, private pending: PendingRatingsService) {}

  generate(): void {
    const city = this.state['_activeCity'].value;
    if (!city) return;

    const params: GenerateRequest = {
      route_type: this.routeType,
      distance_mi: this.distance,
      tolerance: this.tolerance,
      start_lat: city.lat,
      start_lng: city.lng,
      hilly_factor: this.hillyFactor,
      downtown_radius: this.downtownRadius,
      novelty_factor: this.noveltyFactor,
    };

    this.error = '';
    this.state.setLoading(true);
    this.state.clearRoutes();

    this.api.generateRoutes(params).subscribe({
      next: routes => {
        this.state.setRoutes(routes);
        this.state.setLoading(false);
        this.pending.refresh();
      },
      error: err => {
        this.error = err.error?.detail ?? 'Generation failed. Please try again.';
        this.state.setLoading(false);
      }
    });
  }
}
