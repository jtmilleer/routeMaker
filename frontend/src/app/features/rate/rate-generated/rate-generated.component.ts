// LAYER: Component — Rate Generated Routes (queue + rated history)
// PURPOSE: Rate un-rated generated routes one at a time, after riding them.
//          Shows predicted score vs actual rating after submission. A
//          "View rated" toggle switches to a read-only predicted-vs-actual
//          list of routes already rated. Nav now lives in AppShellComponent +
//          RateShellComponent's sub-nav.

import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { RouteApiService, RouteResult, RatingStats } from '../../../core/services/route-api.service';
import { PendingRatingsService } from '../../../core/services/pending-ratings.service';

@Component({
  selector: 'app-rate-generated',
  standalone: true,
  imports: [CommonModule, RouterLink],
  template: `
    <div class="page">
      <div class="toolbar">
        <button class="toggle-view-btn" (click)="toggleView()" id="toggle-rated-view-gen">
          {{ viewMode === 'pending' ? 'View rated (' + ratedRoutes.length + ')' : '← Back to rating' }}
        </button>
      </div>

      <div class="model-bar" *ngIf="stats">
        <span>Preference model v{{ stats.model_version }}</span>
        <span>{{ stats.total_ratings }} total ratings</span>
        <span *ngIf="stats.ratings_until_next_retrain > 0">
          {{ stats.ratings_until_next_retrain }} until next retrain
        </span>
      </div>

      <ng-container *ngIf="viewMode === 'pending'">
        <div class="route-card" *ngIf="current && !done && !submitted">
          <div class="card-header">
            <div>
              <h2 class="route-label">{{ current.route_type | titlecase }} Route</h2>
              <span class="predicted">Predicted score: <strong>{{ current.predicted_score }}/10</strong></span>
            </div>
            <span class="route-count">{{ currentIndex + 1 }} / {{ routes.length }}</span>
          </div>
          <div class="route-stats">
            <div class="stat-box">
              <span class="stat-val">{{ current.distance_mi | number:'1.1-1' }}</span>
              <span class="stat-label">miles</span>
            </div>
            <div class="stat-box">
              <span class="stat-val">{{ current.elevation_ft | number:'1.0-0' }}</span>
              <span class="stat-label">ft gain</span>
            </div>
            <div class="stat-box" *ngIf="current.novelty_pct != null">
              <span class="stat-val">{{ current.novelty_pct | number:'1.0-0' }}%</span>
              <span class="stat-label">new roads</span>
            </div>
          </div>
          <div class="rating-row">
            <span class="rating-label">How was this route after actually riding it?</span>
            <div class="rating-btns">
              <button *ngFor="let r of ratings"
                class="rating-btn"
                [class.selected]="selectedRating === r"
                [id]="'gen-rating-' + r"
                (click)="rate(r)">{{ r }}</button>
            </div>
          </div>
          <button class="skip-btn" (click)="skip()" id="skip-gen-btn">Skip</button>
        </div>

        <div class="comparison-card" *ngIf="submitted && lastRating != null">
          <h3>Rating Saved</h3>
          <div class="compare-row">
            <div class="compare-item">
              <span class="compare-val">{{ lastPredicted }}</span>
              <span class="compare-label">Predicted</span>
            </div>
            <div class="compare-arrow">&rarr;</div>
            <div class="compare-item">
              <span class="compare-val actual">{{ lastRating }}</span>
              <span class="compare-label">Your Rating</span>
            </div>
          </div>
          <p class="compare-note" *ngIf="diff > 1.5">This difference will improve future routes.</p>
          <button class="next-btn" (click)="nextAfterSubmit()" id="next-route-btn">Next Route &rarr;</button>
        </div>

        <div class="done-state" *ngIf="done && !submitted">
          <h2>All routes reviewed</h2>
          <a routerLink="/app" class="back-btn">Back to Route Builder</a>
        </div>

        <div class="loading-state" *ngIf="loading">
          <div class="spinner"></div>
          <span>Loading route history...</span>
        </div>
      </ng-container>

      <div class="rated-list" *ngIf="viewMode === 'rated'">
        <div class="rated-empty" *ngIf="ratedRoutes.length === 0">No rated routes yet.</div>
        <div class="rated-row" *ngFor="let r of ratedRoutes">
          <span class="rated-type">{{ r.route_type | titlecase }} · {{ r.distance_mi | number:'1.1-1' }} mi</span>
          <span class="rated-compare">
            <span class="rated-predicted">{{ r.predicted_score }}</span>
            <span class="rated-arrow">&rarr;</span>
            <span class="rated-actual">{{ r.user_rating }}</span>
          </span>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .page { min-height: 100%; background: var(--bg-primary); color: var(--text-primary); padding: 2rem; font-family: var(--font-primary); }
    .route-label, .comparison-card h3, .done-state h2 { font-family: var(--font-display); }

    .toolbar { display: flex; justify-content: flex-end; margin-bottom: 1rem; }
    .toggle-view-btn { background: none; border: 1px solid var(--border); border-radius: 6px; color: var(--text-muted); padding: 0.35rem 0.75rem; font-size: 0.8rem; font-family: var(--font-primary); cursor: pointer; }
    .toggle-view-btn:hover { color: var(--text-primary); border-color: var(--surface-active-border); }

    .model-bar {
      display: flex; gap: 1.5rem;
      background: var(--bg-surface); border: 1px solid var(--border);
      border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 1.5rem;
      font-size: 0.82rem; color: var(--text-muted);
    }

    .route-card {
      max-width: 580px; margin: 0 auto;
      background: var(--bg-surface); border: 1px solid var(--border);
      border-radius: 16px; padding: 1.75rem; display: flex; flex-direction: column; gap: 1.25rem;
    }
    .card-header { display: flex; justify-content: space-between; align-items: flex-start; }
    .route-label { margin: 0 0 0.25rem; font-size: 1.15rem; font-weight: 700; color: var(--text-primary); }
    .predicted { font-size: 0.82rem; color: var(--text-muted); }
    .predicted strong { color: var(--accent); }
    .route-count { font-size: 0.8rem; color: var(--text-dim); }

    .route-stats { display: flex; gap: 1.25rem; }
    .stat-box { display: flex; flex-direction: column; }
    .stat-val { font-size: 1.25rem; font-weight: 800; color: var(--text-primary); }
    .stat-label { font-size: 0.65rem; color: var(--text-dim); text-transform: uppercase; }

    .rating-label { font-size: 0.875rem; color: var(--text-muted); display: block; margin-bottom: 0.5rem; }
    .rating-btns { display: flex; gap: 0.4rem; flex-wrap: wrap; }
    .rating-btn {
      width: 42px; height: 42px; border-radius: 10px; border: 1px solid var(--border);
      background: var(--bg-surface); color: var(--text-secondary); font-size: 0.9rem; font-weight: 600;
      font-family: var(--font-primary); cursor: pointer; transition: all 0.12s;
    }
    .rating-btn:hover { background: var(--surface-active); border-color: var(--surface-active-border); color: var(--accent); }
    .rating-btn.selected { background: rgba(168, 71, 31, 0.16); border-color: var(--accent); color: var(--accent); }

    .skip-btn { background: none; border: none; color: var(--text-dim); font-size: 0.8rem; font-family: var(--font-primary); cursor: pointer; align-self: center; }

    .comparison-card {
      max-width: 400px; margin: 2rem auto; text-align: center;
      background: var(--bg-surface); border: 1px solid var(--border);
      border-radius: 16px; padding: 2rem; display: flex; flex-direction: column; gap: 1.25rem; align-items: center;
    }
    .comparison-card h3 { margin: 0; color: var(--text-primary); }
    .compare-row { display: flex; align-items: center; gap: 1.5rem; }
    .compare-item { display: flex; flex-direction: column; align-items: center; }
    .compare-val { font-size: 2.5rem; font-weight: 800; color: var(--text-muted); }
    .compare-val.actual { color: var(--accent); }
    .compare-arrow { font-size: 1.5rem; color: var(--text-dim); }
    .compare-label { font-size: 0.7rem; color: var(--text-dim); text-transform: uppercase; }
    .compare-note { color: var(--text-muted); font-size: 0.85rem; margin: 0; }
    .next-btn { padding: 0.625rem 1.5rem; background: var(--accent); color: var(--on-accent); border: none; border-radius: 4px; font-weight: 600; font-family: var(--font-primary); cursor: pointer; box-shadow: 3px 3px 0 rgba(60, 46, 30, 0.25); transition: all 0.15s; }
    .next-btn:hover { background: var(--accent-hover); transform: translate(-1px, -1px); box-shadow: 4px 4px 0 rgba(60, 46, 30, 0.3); }

    .done-state, .loading-state {
      max-width: 400px; margin: 4rem auto; text-align: center;
      display: flex; flex-direction: column; align-items: center; gap: 1rem;
    }
    .done-state h2 { margin: 0; color: var(--text-primary); }
    .back-btn { padding: 0.625rem 1.5rem; background: var(--accent); color: var(--on-accent); border-radius: 4px; text-decoration: none; font-weight: 600; display: inline-block; box-shadow: 3px 3px 0 rgba(60, 46, 30, 0.25); transition: all 0.15s; }
    .back-btn:hover { background: var(--accent-hover); transform: translate(-1px, -1px); box-shadow: 4px 4px 0 rgba(60, 46, 30, 0.3); }
    .spinner { width: 32px; height: 32px; border: 3px solid rgba(168, 71, 31, 0.2); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }

    .rated-list { max-width: 580px; margin: 0 auto; display: flex; flex-direction: column; gap: 0.5rem; }
    .rated-empty { color: var(--text-muted); text-align: center; padding: 2rem; }
    .rated-row {
      display: flex; justify-content: space-between; align-items: center;
      background: var(--bg-surface); border: 1px solid var(--border); border-radius: 10px;
      padding: 0.75rem 1rem; font-size: 0.85rem;
    }
    .rated-type { color: var(--text-primary); font-weight: 600; }
    .rated-compare { display: flex; align-items: center; gap: 0.5rem; font-family: var(--font-mono); }
    .rated-predicted { color: var(--text-muted); }
    .rated-arrow { color: var(--text-dim); }
    .rated-actual { color: var(--accent); font-weight: 700; }
  `]
})
export class RateGeneratedComponent implements OnInit {
  allRoutes: RouteResult[] = [];
  routes: RouteResult[] = [];
  ratedRoutes: RouteResult[] = [];
  currentIndex = 0;
  selectedRating: number | null = null;
  stats: RatingStats | null = null;
  loading = true;
  done = false;
  submitted = false;
  lastRating: number | null = null;
  lastPredicted = 0;
  viewMode: 'pending' | 'rated' = 'pending';
  ratings = [1,2,3,4,5,6,7,8,9,10];

  get current(): RouteResult | null { return this.routes[this.currentIndex] ?? null; }
  get diff(): number { return Math.abs((this.lastRating ?? 0) - this.lastPredicted); }

  constructor(private api: RouteApiService, private pending: PendingRatingsService) {}

  ngOnInit(): void {
    this.viewMode = 'pending';
    this.api.getRatingStats().subscribe(s => this.stats = s);
    this.api.getRouteHistory().subscribe({
      next: routes => {
        this.allRoutes = routes;
        this.routes = routes.filter(r => r.user_rating == null);
        this.ratedRoutes = routes.filter(r => r.user_rating != null);
        this.loading = false;
        this.done = this.routes.length === 0;
      },
      error: () => { this.loading = false; }
    });
  }

  rate(r: number): void {
    const route = this.current;
    if (!route) return;
    this.lastPredicted = route.predicted_score;
    this.lastRating = r;
    this.api.rateGeneratedRoute(route.id, r).subscribe({
      next: stats => {
        this.stats = stats;
        this.submitted = true;
        this.pending.decrement();
        this.ratedRoutes = [{ ...route, user_rating: r }, ...this.ratedRoutes];
      }
    });
  }

  skip(): void { this.advance(); }

  nextAfterSubmit(): void {
    this.submitted = false;
    this.lastRating = null;
    this.advance();
  }

  toggleView(): void {
    this.viewMode = this.viewMode === 'pending' ? 'rated' : 'pending';
  }

  private advance(): void {
    this.selectedRating = null;
    if (this.currentIndex + 1 >= this.routes.length) {
      this.done = true;
    } else {
      this.currentIndex++;
    }
  }
}
