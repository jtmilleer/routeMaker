// LAYER: Component — Rate Strava Rides (port of rating.html)
// PURPOSE: Shows the user's Strava rides one at a time with stats and a 1-10
//          rating row. After rating, advances to the next unrated ride.
//          Shows model training progress (ratings remaining until next retrain).

import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { RouteApiService, Ride, RatingStats } from '../../core/services/route-api.service';

@Component({
  selector: 'app-rate-rides',
  standalone: true,
  imports: [CommonModule, RouterLink],
  template: `
    <div class="page">
      <nav class="page-nav">
        <a routerLink="/app" class="back-link" id="back-to-app">← Back to Route Builder</a>
        <h1 class="page-title">Rate Your Rides</h1>
      </nav>

      <!-- Model stats bar -->
      <div class="model-bar" *ngIf="stats">
        <span>🤖 Model v{{ stats.model_version }}</span>
        <span>{{ stats.total_ratings }} total ratings</span>
        <span *ngIf="stats.ratings_until_next_retrain > 0">
          {{ stats.ratings_until_next_retrain }} ratings until next retrain
        </span>
        <span *ngIf="stats.ratings_until_next_retrain === 0" class="retrain-soon">
          Retraining triggered!
        </span>
      </div>

      <!-- Syncing -->
      <div class="sync-row">
        <button class="sync-btn" (click)="syncRides()" [disabled]="syncing" id="sync-rides-btn">
          {{ syncing ? 'Syncing...' : '↻ Sync from Strava' }}
        </button>
        <span class="hint" *ngIf="syncMsg">{{ syncMsg }}</span>
      </div>

      <!-- Ride card -->
      <div class="ride-card" *ngIf="currentRide && !done">
        <div class="ride-header">
          <h2 class="ride-name">{{ currentRide.name }}</h2>
          <span class="ride-count">{{ currentIndex + 1 }} / {{ rides.length }}</span>
        </div>
        <div class="ride-stats">
          <div class="stat-box">
            <span class="stat-val">{{ currentRide.distance_mi | number:'1.1-1' }}</span>
            <span class="stat-label">miles</span>
          </div>
          <div class="stat-box">
            <span class="stat-val">{{ currentRide.elevation_ft | number:'1.0-0' }}</span>
            <span class="stat-label">ft gain</span>
          </div>
          <div class="stat-box">
            <span class="stat-val">{{ currentRide.moving_time_min | number:'1.0-0' }}</span>
            <span class="stat-label">min</span>
          </div>
          <div class="stat-box">
            <span class="stat-val">{{ currentRide.avg_speed_mph | number:'1.1-1' }}</span>
            <span class="stat-label">mph avg</span>
          </div>
        </div>
        <div class="rating-row">
          <span class="rating-label">How enjoyable was this ride?</span>
          <div class="rating-btns">
            <button *ngFor="let r of ratings"
              class="rating-btn"
              [class.selected]="selectedRating === r"
              [id]="'rating-' + r"
              (click)="rate(r)">
              {{ r }}
            </button>
          </div>
        </div>
        <div class="skip-row">
          <button class="skip-btn" (click)="skip()" id="skip-ride-btn">Skip (no rating)</button>
        </div>
      </div>

      <div class="done-state" *ngIf="done">
        <div class="done-icon">🎉</div>
        <h2>All rides rated!</h2>
        <p>Your model will personalize over time as you add more data.</p>
        <a routerLink="/app" class="back-btn">Back to Route Builder</a>
      </div>

      <div class="loading-state" *ngIf="loading">
        <div class="spinner"></div>
        <span>Loading rides...</span>
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
      display: flex; gap: 1.5rem; align-items: center;
      background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
      border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 1rem; font-size: 0.82rem; color: #9ca3af;
    }
    .retrain-soon { color: #fc4c02; font-weight: 600; }

    .sync-row { display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem; }
    .sync-btn {
      padding: 0.5rem 1rem; background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.12);
      border-radius: 8px; color: #d1d5db; font-size: 0.85rem; cursor: pointer;
    }
    .hint { font-size: 0.8rem; color: #6b7280; }

    .ride-card {
      max-width: 580px; margin: 0 auto;
      background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1);
      border-radius: 16px; padding: 1.75rem; display: flex; flex-direction: column; gap: 1.25rem;
    }
    .ride-header { display: flex; justify-content: space-between; align-items: flex-start; }
    .ride-name { margin: 0; font-size: 1.15rem; font-weight: 700; color: #fff; }
    .ride-count { font-size: 0.8rem; color: #6b7280; flex-shrink: 0; }

    .ride-stats { display: flex; gap: 1.25rem; }
    .stat-box { display: flex; flex-direction: column; }
    .stat-val { font-size: 1.25rem; font-weight: 800; color: #fff; }
    .stat-label { font-size: 0.65rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; }

    .rating-label { font-size: 0.875rem; color: #9ca3af; margin-bottom: 0.5rem; display: block; }
    .rating-btns { display: flex; gap: 0.4rem; flex-wrap: wrap; }
    .rating-btn {
      width: 42px; height: 42px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.06); color: #d1d5db; font-size: 0.9rem; font-weight: 600;
      cursor: pointer; transition: all 0.12s;
    }
    .rating-btn:hover { background: rgba(252,76,2,0.15); border-color: rgba(252,76,2,0.4); color: #fc4c02; }
    .rating-btn.selected { background: rgba(252,76,2,0.2); border-color: #fc4c02; color: #fc4c02; }

    .skip-row { text-align: center; }
    .skip-btn { background: none; border: none; color: #6b7280; font-size: 0.8rem; cursor: pointer; }
    .skip-btn:hover { color: #9ca3af; }

    .done-state, .loading-state {
      max-width: 400px; margin: 4rem auto; text-align: center;
      display: flex; flex-direction: column; align-items: center; gap: 1rem;
    }
    .done-icon { font-size: 3rem; }
    .done-state h2 { margin: 0; color: #fff; }
    .done-state p { color: #9ca3af; margin: 0; }
    .back-btn {
      padding: 0.625rem 1.5rem; background: #fc4c02; color: #fff;
      border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 0.9rem;
    }
    .spinner { width: 32px; height: 32px; border: 3px solid rgba(252,76,2,0.2); border-top-color: #fc4c02; border-radius: 50%; animation: spin 0.8s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
  `]
})
export class RateRidesComponent implements OnInit {
  rides: Ride[] = [];
  currentIndex = 0;
  selectedRating: number | null = null;
  stats: RatingStats | null = null;
  loading = true;
  syncing = false;
  syncMsg = '';
  done = false;
  ratings = [1,2,3,4,5,6,7,8,9,10];

  get currentRide(): Ride | null {
    return this.rides[this.currentIndex] ?? null;
  }

  constructor(private api: RouteApiService) {}

  ngOnInit(): void {
    this.api.getRatingStats().subscribe(s => this.stats = s);
    this.api.getRides().subscribe({
      next: rides => {
        this.rides = rides.filter(r => r.user_rating == null);
        this.loading = false;
        this.done = this.rides.length === 0;
      },
      error: () => { this.loading = false; }
    });
  }

  syncRides(): void {
    this.syncing = true;
    this.api.syncRides().subscribe({
      next: s => {
        this.syncMsg = `${s.new_rides} new rides synced (${s.total_rides} total)`;
        this.syncing = false;
        this.ngOnInit();
      },
      error: () => { this.syncing = false; }
    });
  }

  rate(r: number): void {
    this.selectedRating = r;
    const ride = this.currentRide;
    if (!ride) return;

    this.api.rateRide(ride.id, r).subscribe({
      next: stats => {
        this.stats = stats;
        this.advance();
      }
    });
  }

  skip(): void {
    this.advance();
  }

  private advance(): void {
    this.selectedRating = null;
    if (this.currentIndex + 1 >= this.rides.length) {
      this.done = true;
    } else {
      this.currentIndex++;
    }
  }
}
