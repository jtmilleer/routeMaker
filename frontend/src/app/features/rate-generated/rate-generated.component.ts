// LAYER: Component — Rate Generated Routes (port of feedback.html)
// PURPOSE: Shows the user's generated route history one at a time for rating.
//          Shows predicted score vs actual user rating after submission.
//          Same retrain trigger logic as rate-rides (combined counter).

import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { RouteApiService, RouteResult, RatingStats } from '../../core/services/route-api.service';

@Component({
  selector: 'app-rate-generated',
  standalone: true,
  imports: [CommonModule, RouterLink],
  template: `
    <div class="page">
      <nav class="page-nav">
        <a routerLink="/app" class="back-link" id="back-to-app-gen">← Back to Route Builder</a>
        <h1 class="page-title">Rate Generated Routes</h1>
      </nav>

      <div class="model-bar" *ngIf="stats">
        <span>🤖 Model v{{ stats.model_version }}</span>
        <span>{{ stats.total_ratings }} total ratings</span>
        <span *ngIf="stats.ratings_until_next_retrain > 0">
          {{ stats.ratings_until_next_retrain }} until next retrain
        </span>
      </div>

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

      <!-- After rating: show predicted vs actual comparison -->
      <div class="comparison-card" *ngIf="submitted && lastRating != null">
        <h3>Rating Saved!</h3>
        <div class="compare-row">
          <div class="compare-item">
            <span class="compare-val">{{ lastPredicted }}</span>
            <span class="compare-label">Predicted</span>
          </div>
          <div class="compare-arrow">→</div>
          <div class="compare-item">
            <span class="compare-val actual">{{ lastRating }}</span>
            <span class="compare-label">Your Rating</span>
          </div>
        </div>
        <p class="compare-note" *ngIf="diff > 1.5">The model will learn from this difference!</p>
        <button class="next-btn" (click)="nextAfterSubmit()" id="next-route-btn">Next Route →</button>
      </div>

      <div class="done-state" *ngIf="done && !submitted">
        <div class="done-icon">✅</div>
        <h2>All routes reviewed!</h2>
        <a routerLink="/app" class="back-btn">Back to Route Builder</a>
      </div>

      <div class="loading-state" *ngIf="loading">
        <div class="spinner"></div>
        <span>Loading route history...</span>
      </div>
    </div>
  `,
  styles: [`
    .page { min-height: 100vh; background: #0f0f1a; color: #e5e7eb; padding: 2rem; font-family: 'Inter', sans-serif; }
    .page-nav { display: flex; align-items: center; gap: 1.5rem; margin-bottom: 1.5rem; }
    .back-link { color: #6b7280; text-decoration: none; font-size: 0.875rem; }
    .back-link:hover { color: #fff; }
    .page-title { margin: 0; font-size: 1.5rem; font-weight: 800; color: #fff; }

    .model-bar {
      display: flex; gap: 1.5rem;
      background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
      border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 1.5rem;
      font-size: 0.82rem; color: #9ca3af;
    }

    .route-card {
      max-width: 580px; margin: 0 auto;
      background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1);
      border-radius: 16px; padding: 1.75rem; display: flex; flex-direction: column; gap: 1.25rem;
    }
    .card-header { display: flex; justify-content: space-between; align-items: flex-start; }
    .route-label { margin: 0 0 0.25rem; font-size: 1.15rem; font-weight: 700; color: #fff; }
    .predicted { font-size: 0.82rem; color: #9ca3af; }
    .predicted strong { color: #fc4c02; }
    .route-count { font-size: 0.8rem; color: #6b7280; }

    .route-stats { display: flex; gap: 1.25rem; }
    .stat-box { display: flex; flex-direction: column; }
    .stat-val { font-size: 1.25rem; font-weight: 800; color: #fff; }
    .stat-label { font-size: 0.65rem; color: #6b7280; text-transform: uppercase; }

    .rating-label { font-size: 0.875rem; color: #9ca3af; display: block; margin-bottom: 0.5rem; }
    .rating-btns { display: flex; gap: 0.4rem; flex-wrap: wrap; }
    .rating-btn {
      width: 42px; height: 42px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.06); color: #d1d5db; font-size: 0.9rem; font-weight: 600;
      cursor: pointer; transition: all 0.12s;
    }
    .rating-btn:hover { background: rgba(252,76,2,0.15); border-color: rgba(252,76,2,0.4); color: #fc4c02; }
    .rating-btn.selected { background: rgba(252,76,2,0.2); border-color: #fc4c02; color: #fc4c02; }

    .skip-btn { background: none; border: none; color: #6b7280; font-size: 0.8rem; cursor: pointer; align-self: center; }

    .comparison-card {
      max-width: 400px; margin: 2rem auto; text-align: center;
      background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1);
      border-radius: 16px; padding: 2rem; display: flex; flex-direction: column; gap: 1.25rem; align-items: center;
    }
    .comparison-card h3 { margin: 0; color: #fff; }
    .compare-row { display: flex; align-items: center; gap: 1.5rem; }
    .compare-item { display: flex; flex-direction: column; align-items: center; }
    .compare-val { font-size: 2.5rem; font-weight: 800; color: #9ca3af; }
    .compare-val.actual { color: #fc4c02; }
    .compare-arrow { font-size: 1.5rem; color: #6b7280; }
    .compare-label { font-size: 0.7rem; color: #6b7280; text-transform: uppercase; }
    .compare-note { color: #9ca3af; font-size: 0.85rem; margin: 0; }
    .next-btn { padding: 0.625rem 1.5rem; background: #fc4c02; color: #fff; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; }

    .done-state, .loading-state {
      max-width: 400px; margin: 4rem auto; text-align: center;
      display: flex; flex-direction: column; align-items: center; gap: 1rem;
    }
    .done-icon { font-size: 3rem; }
    .done-state h2 { margin: 0; color: #fff; }
    .back-btn { padding: 0.625rem 1.5rem; background: #fc4c02; color: #fff; border-radius: 8px; text-decoration: none; font-weight: 600; }
    .spinner { width: 32px; height: 32px; border: 3px solid rgba(252,76,2,0.2); border-top-color: #fc4c02; border-radius: 50%; animation: spin 0.8s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
  `]
})
export class RateGeneratedComponent implements OnInit {
  routes: RouteResult[] = [];
  currentIndex = 0;
  selectedRating: number | null = null;
  stats: RatingStats | null = null;
  loading = true;
  done = false;
  submitted = false;
  lastRating: number | null = null;
  lastPredicted = 0;
  ratings = [1,2,3,4,5,6,7,8,9,10];

  get current(): RouteResult | null { return this.routes[this.currentIndex] ?? null; }
  get diff(): number { return Math.abs((this.lastRating ?? 0) - this.lastPredicted); }

  constructor(private api: RouteApiService) {}

  ngOnInit(): void {
    this.api.getRatingStats().subscribe(s => this.stats = s);
    this.api.getRouteHistory().subscribe({
      next: routes => {
        this.routes = routes;
        this.loading = false;
        this.done = routes.length === 0;
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
      }
    });
  }

  skip(): void { this.advance(); }

  nextAfterSubmit(): void {
    this.submitted = false;
    this.lastRating = null;
    this.advance();
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
