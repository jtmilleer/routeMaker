import { Component, OnInit, AfterViewChecked, ElementRef, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import * as L from 'leaflet';
import { RouteApiService, Ride, RatingStats } from '../../core/services/route-api.service';

function decodePolyline(encoded: string): [number, number][] {
  const points: [number, number][] = [];
  let idx = 0, lat = 0, lng = 0;
  while (idx < encoded.length) {
    let b, shift = 0, result = 0;
    do { b = encoded.charCodeAt(idx++) - 63; result |= (b & 0x1f) << shift; shift += 5; } while (b >= 0x20);
    lat += (result & 1) ? ~(result >> 1) : (result >> 1);
    shift = 0; result = 0;
    do { b = encoded.charCodeAt(idx++) - 63; result |= (b & 0x1f) << shift; shift += 5; } while (b >= 0x20);
    lng += (result & 1) ? ~(result >> 1) : (result >> 1);
    points.push([lat / 1e5, lng / 1e5]);
  }
  return points;
}

@Component({
  selector: 'app-rate-rides',
  standalone: true,
  imports: [CommonModule, RouterLink],
  template: `
    <div class="page">
      <nav class="page-nav">
        <a routerLink="/app" class="back-link" id="back-to-app">&larr; Back to Route Builder</a>
        <h1 class="page-title">Rate Your Rides</h1>
      </nav>

      <div class="model-bar" *ngIf="stats">
        <span>Preference model v{{ stats.model_version }}</span>
        <span>{{ stats.total_ratings }} total ratings</span>
        <span *ngIf="stats.ratings_until_next_retrain > 0">
          {{ stats.ratings_until_next_retrain }} ratings until next retrain
        </span>
        <span *ngIf="stats.ratings_until_next_retrain === 0" class="retrain-soon">
          Retraining triggered!
        </span>
      </div>

      <div class="sync-row">
        <button class="sync-btn" (click)="syncRides()" [disabled]="syncing" id="sync-rides-btn">
          {{ syncing ? 'Syncing...' : 'Sync from Strava' }}
        </button>
        <span class="hint" *ngIf="syncMsg">{{ syncMsg }}</span>
      </div>

      <div class="ride-card" *ngIf="currentRide && !done">
        <div class="ride-header">
          <h2 class="ride-name">{{ currentRide.name }}</h2>
          <span class="ride-count">{{ currentIndex + 1 }} / {{ rides.length }}</span>
        </div>
        <div #rideMap class="ride-map" *ngIf="currentRide.polyline"></div>
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
            <button *ngFor="let r of ratingValues"
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

      <div class="done-state" *ngIf="!loading && rides.length === 0 && !done">
        <h2>No rides yet</h2>
        <p>Hit "Sync from Strava" to import your ride history.</p>
      </div>

      <div class="done-state" *ngIf="done && rides.length === 0 && totalRides > 0">
        <h2>All rides rated!</h2>
        <p>Your routes will personalize as you add more data.</p>
        <a routerLink="/app" class="back-btn">Back to Route Builder</a>
      </div>

      <div class="done-state" *ngIf="done && rides.length > 0">
        <h2>All done!</h2>
        <p>Your routes will personalize as you add more data.</p>
        <a routerLink="/app" class="back-btn">Back to Route Builder</a>
      </div>

      <div class="loading-state" *ngIf="loading">
        <div class="spinner"></div>
        <span>Loading rides...</span>
      </div>
    </div>
  `,
  styles: [`
    .page { min-height: 100vh; background: var(--bg-primary); color: var(--text-primary); padding: 2rem; font-family: var(--font-primary); }
    .page-title, .ride-name, .done-state h2 { font-family: var(--font-display); }
    .page-nav { display: flex; align-items: center; gap: 1.5rem; margin-bottom: 1.5rem; }
    .back-link { color: var(--text-dim); text-decoration: none; font-size: 0.875rem; }
    .back-link:hover { color: var(--text-primary); }
    .page-title { margin: 0; font-size: 1.5rem; font-weight: 800; color: var(--text-primary); }

    .model-bar {
      display: flex; gap: 1.5rem; align-items: center;
      background: var(--bg-surface); border: 1px solid var(--border);
      border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 1rem; font-size: 0.82rem; color: var(--text-muted);
    }
    .retrain-soon { color: var(--accent); font-weight: 600; }

    .sync-row { display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem; }
    .sync-btn {
      padding: 0.5rem 1rem; background: var(--bg-surface); border: 1px solid var(--border);
      border-radius: 8px; color: var(--text-secondary); font-size: 0.85rem; font-family: var(--font-primary); cursor: pointer;
    }
    .hint { font-size: 0.8rem; color: var(--text-dim); }

    .ride-card {
      max-width: 580px; margin: 0 auto;
      background: var(--bg-surface); border: 1px solid var(--border);
      border-radius: 16px; padding: 1.75rem; display: flex; flex-direction: column; gap: 1.25rem;
    }
    .ride-header { display: flex; justify-content: space-between; align-items: flex-start; }
    .ride-name { margin: 0; font-size: 1.15rem; font-weight: 700; color: var(--text-primary); }
    .ride-count { font-size: 0.8rem; color: var(--text-dim); flex-shrink: 0; }

    .ride-map {
      width: 100%; height: 250px; border-radius: 10px; overflow: hidden;
      border: 1px solid var(--border);
    }

    .ride-stats { display: flex; gap: 1.25rem; }
    .stat-box { display: flex; flex-direction: column; }
    .stat-val { font-size: 1.25rem; font-weight: 800; color: var(--text-primary); }
    .stat-label { font-size: 0.65rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; }

    .rating-label { font-size: 0.875rem; color: var(--text-muted); margin-bottom: 0.5rem; display: block; }
    .rating-btns { display: flex; gap: 0.4rem; flex-wrap: wrap; }
    .rating-btn {
      width: 42px; height: 42px; border-radius: 10px; border: 1px solid var(--border);
      background: var(--bg-surface); color: var(--text-secondary); font-size: 0.9rem; font-weight: 600;
      font-family: var(--font-primary); cursor: pointer; transition: all 0.12s;
    }
    .rating-btn:hover { background: var(--surface-active); border-color: var(--surface-active-border); color: var(--accent); }
    .rating-btn.selected { background: rgba(168, 71, 31, 0.16); border-color: var(--accent); color: var(--accent); }

    .skip-row { text-align: center; }
    .skip-btn { background: none; border: none; color: var(--text-dim); font-size: 0.8rem; font-family: var(--font-primary); cursor: pointer; }
    .skip-btn:hover { color: var(--text-muted); }

    .done-state, .loading-state {
      max-width: 400px; margin: 4rem auto; text-align: center;
      display: flex; flex-direction: column; align-items: center; gap: 1rem;
    }
    .done-state h2 { margin: 0; color: var(--text-primary); }
    .done-state p { color: var(--text-muted); margin: 0; }
    .back-btn {
      padding: 0.625rem 1.5rem; background: var(--accent); color: var(--on-accent);
      border-radius: 4px; text-decoration: none; font-weight: 600; font-size: 0.9rem;
      display: inline-block; box-shadow: 3px 3px 0 rgba(60, 46, 30, 0.25); transition: all 0.15s;
    }
    .back-btn:hover { background: var(--accent-hover); transform: translate(-1px, -1px); box-shadow: 4px 4px 0 rgba(60, 46, 30, 0.3); }
    .spinner { width: 32px; height: 32px; border: 3px solid rgba(168, 71, 31, 0.2); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
  `]
})
export class RateRidesComponent implements OnInit, AfterViewChecked {
  @ViewChild('rideMap') mapEl!: ElementRef;

  rides: Ride[] = [];
  totalRides = 0;
  currentIndex = 0;
  selectedRating: number | null = null;
  stats: RatingStats | null = null;
  loading = true;
  syncing = false;
  syncMsg = '';
  done = false;
  ratingValues = [1,2,3,4,5,6,7,8,9,10];

  private map: L.Map | null = null;
  private renderedRideId: number | null = null;
  private submitting = false;

  get currentRide(): Ride | null {
    return this.rides[this.currentIndex] ?? null;
  }

  constructor(private api: RouteApiService) {}

  ngOnInit(): void {
    this.loading = true;
    this.done = false;
    this.currentIndex = 0;
    this.api.getRatingStats().subscribe(s => this.stats = s);
    this.api.getRides().subscribe({
      next: rides => {
        this.totalRides = rides.length;
        this.rides = rides.filter(r => r.user_rating == null);
        this.loading = false;
        if (this.rides.length === 0 && this.totalRides > 0) {
          this.done = true;
        }
      },
      error: () => { this.loading = false; }
    });
  }

  ngAfterViewChecked(): void {
    const ride = this.currentRide;
    if (!ride || !ride.polyline || !this.mapEl) return;
    if (this.renderedRideId === ride.id) return;
    this.renderedRideId = ride.id;
    this.renderMap(ride.polyline);
  }

  private renderMap(polyline: string): void {
    if (this.map) {
      this.map.remove();
      this.map = null;
    }
    const points = decodePolyline(polyline);
    if (points.length === 0) return;

    this.map = L.map(this.mapEl.nativeElement, {
      zoomControl: false,
      attributionControl: false,
      dragging: false,
      scrollWheelZoom: false,
      doubleClickZoom: false,
      touchZoom: false,
    });

    L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
      maxZoom: 18,
    }).addTo(this.map);

    const line = L.polyline(points, { color: '#a8471f', weight: 3, opacity: 0.9 }).addTo(this.map);
    this.map.fitBounds(line.getBounds(), { padding: [20, 20] });
  }

  syncRides(): void {
    this.syncing = true;
    this.api.syncRides().subscribe({
      next: s => {
        this.syncMsg = `${s.new_rides} new rides synced (${s.total_rides} total)`;
        this.syncing = false;
        this.renderedRideId = null;
        this.ngOnInit();
      },
      error: () => { this.syncing = false; }
    });
  }

  rate(r: number): void {
    if (this.submitting) return;
    this.submitting = true;
    this.selectedRating = r;
    const ride = this.currentRide;
    if (!ride) return;

    this.api.rateRide(ride.id, r).subscribe({
      next: stats => {
        this.stats = stats;
        setTimeout(() => {
          this.advance();
          this.submitting = false;
        }, 300);
      },
      error: () => { this.submitting = false; }
    });
  }

  skip(): void {
    this.advance();
  }

  private advance(): void {
    this.selectedRating = null;
    this.renderedRideId = null;
    if (this.currentIndex + 1 >= this.rides.length) {
      this.done = true;
    } else {
      this.currentIndex++;
    }
  }
}
