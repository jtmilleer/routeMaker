// LAYER: Component — Route Builder (Main App Page)
// PURPOSE: Parent layout component for the main app experience.
//          Assembles the city-selector, route-form sidebar, map-view,
//          and route-results panel. Coordinates child components via
//          route-state.service. Handles the top navigation bar.

import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { AuthService } from '../../core/services/auth.service';
import { RouteStateService } from '../../core/services/route-state.service';
import { CitySelectorComponent } from '../../shared/components/city-selector/city-selector.component';
import { RouteFormComponent } from './route-form/route-form.component';
import { RouteResultsComponent } from './route-results/route-results.component';
import { MapViewComponent } from '../map-view/map-view.component';
import { MapBuildingBannerComponent } from '../../shared/components/map-building-banner/map-building-banner.component';
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
    MapViewComponent,
    MapBuildingBannerComponent,
    LoadingOverlayComponent,
  ],
  template: `
    <div class="app-shell">
      <!-- Top Navigation -->
      <nav class="topnav">
        <div class="nav-brand">
          <span class="brand-name">RouteMaker</span>
        </div>
        <div class="nav-links">
          <a routerLink="/app/rate-rides" class="nav-link" id="nav-rate-rides">Rate Rides</a>
          <a routerLink="/app/rate-generated" class="nav-link" id="nav-rate-generated">Rate Generated</a>
          <a routerLink="/app/coverage" class="nav-link" id="nav-coverage">Coverage</a>
          <div class="nav-user" *ngIf="auth.currentUser() as user">
            <img *ngIf="user.profile_pic_url" [src]="user.profile_pic_url" class="avatar" alt="Profile">
            <span class="user-name">{{ user.name }}</span>
            <span class="model-badge" title="Preference model version">v{{ user.model_version }}</span>
          </div>
          <button class="logout-btn" (click)="auth.logout()" id="logout-btn">Sign Out</button>
        </div>
      </nav>

      <!-- Graph building banner (shown when map is downloading) -->
      <app-map-building-banner></app-map-building-banner>

      <div class="main-layout">
        <!-- Sidebar -->
        <aside class="sidebar">
          <app-city-selector></app-city-selector>
          <app-route-form></app-route-form>
          <app-route-results></app-route-results>
        </aside>

        <!-- Map area -->
        <main class="map-area">
          <app-loading-overlay *ngIf="state.loading$ | async"></app-loading-overlay>
          <app-map-view></app-map-view>
        </main>
      </div>
    </div>
  `,
  styles: [`
    :host { display: block; height: 100vh; overflow: hidden; }
    .app-shell { display: flex; flex-direction: column; height: 100vh; background: var(--bg-primary); color: var(--text-primary); font-family: var(--font-primary); }

    .topnav {
      display: flex; align-items: center; justify-content: space-between;
      padding: 0 1.5rem; height: 56px; min-height: 56px;
      background: rgba(224, 213, 184, 0.95);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(12px);
      z-index: 100;
    }
    .nav-brand { display: flex; align-items: center; gap: 0.5rem; }
    .brand-name { font-size: 1.25rem; font-weight: 700; color: var(--text-primary); font-family: var(--font-display); }

    .nav-links { display: flex; align-items: center; gap: 1rem; }
    .nav-link { color: var(--text-muted); text-decoration: none; font-size: 0.875rem; font-weight: 500; transition: color 0.15s; }
    .nav-link:hover { color: var(--text-primary); }
    .nav-user { display: flex; align-items: center; gap: 0.5rem; }
    .avatar { width: 28px; height: 28px; border-radius: 50%; }
    .user-name { font-size: 0.875rem; color: var(--text-secondary); }
    .model-badge { font-size: 0.7rem; font-family: var(--font-mono); background: var(--surface-active); color: var(--accent); border-radius: 3px; padding: 2px 6px; font-weight: 600; }
    .logout-btn { background: var(--bg-surface); border: 1px solid var(--border); border-radius: 4px; color: var(--text-muted); padding: 0.375rem 0.75rem; font-size: 0.8rem; font-family: var(--font-primary); cursor: pointer; transition: all 0.15s; }
    .logout-btn:hover { background: var(--surface-hover); color: var(--text-primary); }

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
  `]
})
export class RouteBuilderComponent {
  constructor(public auth: AuthService, public state: RouteStateService) {}
}
