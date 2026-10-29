// LAYER: Component — Route Builder (Main App Page)
// PURPOSE: Assembles the city-selector, route-form sidebar, map-view, and
//          route-results panel for the main route generation experience.
//          Until the user has rated enough rides to train a personal model,
//          the form is replaced by a locked panel pointing them to Rate.
//          Nav/chrome now lives in AppShellComponent — this component only
//          owns the builder's own layout.

import { Component, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { RouteApiService, RatingStats } from '../../core/services/route-api.service';
import { RouteStateService } from '../../core/services/route-state.service';
import { CitySelectorComponent } from '../../shared/components/city-selector/city-selector.component';
import { RouteFormComponent } from './route-form/route-form.component';
import { RouteResultsComponent } from './route-results/route-results.component';
import { RouteDetailComponent } from './route-detail/route-detail.component';
import { MapViewComponent } from '../map-view/map-view.component';
import { LoadingOverlayComponent } from '../../shared/components/loading-overlay/loading-overlay.component';

@Component({
  selector: 'app-route-builder',
  standalone: true,
  imports: [
    CommonModule,
    RouterLink,
    CitySelectorComponent,
    RouteFormComponent,
    RouteResultsComponent,
    RouteDetailComponent,
    MapViewComponent,
    LoadingOverlayComponent,
  ],
  template: `
    <div class="main-layout">
      <aside class="sidebar">
        <app-city-selector></app-city-selector>
        <ng-container *ngIf="loaded()">
          <ng-container *ngIf="lockedStats() == null; else locked">
            <app-route-form></app-route-form>
            <app-route-results></app-route-results>
          </ng-container>
          <ng-template #locked>
            <ng-container *ngIf="lockedStats() as s">
              <div class="locked" id="generation-locked">
                <h2 class="locked-title">Rate your rides first</h2>
                <p class="locked-text">
                  RouteMaker ranks routes with a model trained on your own ratings.
                  Rate <strong>{{ s.ratings_until_next_retrain }}</strong> more
                  {{ s.ratings_until_next_retrain === 1 ? 'ride' : 'rides' }} to unlock route generation.
                </p>
                <div class="progress">
                  <div class="progress-fill" [style.width.%]="progressPct(s)"></div>
                </div>
                <a routerLink="/app/rate/rides" class="locked-btn" id="go-rate-btn">Rate rides</a>
              </div>
            </ng-container>
          </ng-template>
        </ng-container>
      </aside>

      <main class="map-area">
        <app-loading-overlay *ngIf="state.loading$ | async"></app-loading-overlay>
        <app-map-view></app-map-view>
        <app-route-detail></app-route-detail>
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

    .locked {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 1rem;
      display: flex; flex-direction: column; gap: 0.75rem;
    }
    .locked-title { font-family: var(--font-display); font-size: 1.15rem; color: var(--text-primary); margin: 0; }
    .locked-text { font-size: 0.85rem; color: var(--text-secondary); line-height: 1.5; margin: 0; }
    .progress { height: 6px; background: var(--border-subtle); border-radius: 3px; overflow: hidden; }
    .progress-fill { height: 100%; background: var(--accent); transition: width 0.3s; }
    .locked-btn {
      display: block; text-align: center; padding: 0.75rem;
      background: var(--accent); color: var(--on-accent);
      border-radius: 4px; font-weight: 700; text-decoration: none;
      box-shadow: 3px 3px 0 rgba(60, 46, 30, 0.25);
      transition: all 0.15s;
    }
    .locked-btn:hover { background: var(--accent-hover); transform: translate(-1px, -1px); box-shadow: 4px 4px 0 rgba(60, 46, 30, 0.3); }
  `]
})
export class RouteBuilderComponent implements OnInit {
  readonly loaded = signal(false);
  // Set only while the user has no model yet; null means generation is unlocked.
  readonly lockedStats = signal<RatingStats | null>(null);

  constructor(public state: RouteStateService, private api: RouteApiService) {}

  ngOnInit(): void {
    this.api.getRatingStats().subscribe({
      next: s => {
        if (!s.has_model) this.lockedStats.set(s);
        this.loaded.set(true);
      },
      // Fail open: if stats can't load, show the form; the backend still enforces the lock.
      error: () => this.loaded.set(true),
    });
  }

  progressPct(s: RatingStats): number {
    const threshold = s.total_ratings + s.ratings_until_next_retrain;
    return threshold > 0 ? (s.total_ratings / threshold) * 100 : 0;
  }
}
