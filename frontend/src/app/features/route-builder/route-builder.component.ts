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
          <span class="brand-icon">🚴</span>
          <span class="brand-name">Route<span class="accent">Maker</span></span>
        </div>
        <div class="nav-links">
          <a routerLink="/app/rate-rides" class="nav-link" id="nav-rate-rides">Rate Rides</a>
          <a routerLink="/app/rate-generated" class="nav-link" id="nav-rate-generated">Rate Generated</a>
          <div class="nav-user" *ngIf="auth.currentUser() as user">
            <img *ngIf="user.profile_pic_url" [src]="user.profile_pic_url" class="avatar" alt="Profile">
            <span class="user-name">{{ user.name }}</span>
            <span class="model-badge" title="Model version">v{{ user.model_version }}</span>
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
    .app-shell { display: flex; flex-direction: column; height: 100vh; background: #0f0f1a; color: #e5e7eb; font-family: 'Inter', sans-serif; }

    .topnav {
      display: flex; align-items: center; justify-content: space-between;
      padding: 0 1.5rem; height: 56px; min-height: 56px;
      background: rgba(15,15,26,0.95);
      border-bottom: 1px solid rgba(255,255,255,0.08);
      backdrop-filter: blur(12px);
      z-index: 100;
    }
    .nav-brand { display: flex; align-items: center; gap: 0.5rem; }
    .brand-icon { font-size: 1.4rem; }
    .brand-name { font-size: 1.25rem; font-weight: 800; color: #fff; }
    .accent { color: #fc4c02; }

    .nav-links { display: flex; align-items: center; gap: 1rem; }
    .nav-link { color: #9ca3af; text-decoration: none; font-size: 0.875rem; font-weight: 500; transition: color 0.15s; }
    .nav-link:hover { color: #fff; }
    .nav-user { display: flex; align-items: center; gap: 0.5rem; }
    .avatar { width: 28px; height: 28px; border-radius: 50%; }
    .user-name { font-size: 0.875rem; color: #d1d5db; }
    .model-badge { font-size: 0.7rem; background: rgba(252,76,2,0.2); color: #fc4c02; border-radius: 4px; padding: 2px 6px; font-weight: 600; }
    .logout-btn { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12); border-radius: 8px; color: #9ca3af; padding: 0.375rem 0.75rem; font-size: 0.8rem; cursor: pointer; transition: all 0.15s; }
    .logout-btn:hover { background: rgba(255,255,255,0.1); color: #fff; }

    .main-layout { display: flex; flex: 1; overflow: hidden; }

    .sidebar {
      width: 320px; min-width: 320px;
      background: rgba(255,255,255,0.03);
      border-right: 1px solid rgba(255,255,255,0.08);
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
