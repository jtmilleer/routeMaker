// LAYER: Component — Graph Build Progress Banner
// PURPOSE: Shown when a new city graph is being downloaded in the background.
//          Polls GET /api/graph/status/{city_key} every 3 seconds.
//          Displays a progress bar and disappears when status === "ready".

import { Component, OnDestroy, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { interval, Subscription, switchMap, takeWhile } from 'rxjs';
import { RouteApiService } from '../../../core/services/route-api.service';
import { RouteStateService } from '../../../core/services/route-state.service';

@Component({
  selector: 'app-map-building-banner',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="banner" *ngIf="(state.graphBuilding$ | async) && progress < 100">
      <div class="banner-content">
        <div class="banner-text">
          <strong>Building your map</strong>
          <span>Downloading road network from OpenStreetMap — {{ progress }}% complete</span>
        </div>
        <div class="progress-bar-wrap">
          <div class="progress-bar" [style.width.%]="progress"></div>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .banner {
      background: linear-gradient(90deg, var(--bg-secondary), #1e2a1e);
      border-bottom: 2px solid var(--surface-active-border);
      padding: 0.625rem 1.5rem;
      z-index: 50;
    }
    .banner-content { display: flex; align-items: center; gap: 0.75rem; }
    .banner-text { flex: 1; }
    .banner-text strong { color: var(--text-primary); font-size: 0.875rem; margin-right: 0.5rem; }
    .banner-text span { color: var(--text-muted); font-size: 0.8rem; }
    .progress-bar-wrap { width: 120px; height: 4px; background: rgba(200, 190, 170, 0.1); border-radius: 2px; flex-shrink: 0; }
    .progress-bar { height: 100%; background: var(--accent); border-radius: 2px; transition: width 0.5s; }
  `]
})
export class MapBuildingBannerComponent implements OnInit, OnDestroy {
  progress = 0;
  private pollSub?: Subscription;
  private cityKey = '';

  constructor(public state: RouteStateService, private api: RouteApiService) {}

  ngOnInit(): void {
    // Watch for graph building state + current city key
    this.state.graphBuilding$.subscribe(building => {
      if (building) {
        const city = (this.state as any)['_activeCity'].value;
        if (city) {
          this.cityKey = city.city_key;
          this.startPolling();
        }
      } else {
        this.stopPolling();
      }
    });
  }

  ngOnDestroy(): void {
    this.stopPolling();
  }

  private startPolling(): void {
    this.stopPolling();
    this.pollSub = interval(3000).pipe(
      switchMap(() => this.api.getGraphStatus(this.cityKey)),
      takeWhile(s => s.status !== 'ready' && s.status !== 'error', true),
    ).subscribe(status => {
      this.progress = status.progress_pct;
      if (status.status === 'ready') {
        this.state.setGraphBuilding(false);
      }
    });
  }

  private stopPolling(): void {
    this.pollSub?.unsubscribe();
  }
}
